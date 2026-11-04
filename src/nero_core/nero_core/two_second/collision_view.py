"""Shared offline geometry/projection used by PNG exports and the camera UI."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from .collision_capture import frame_metadata
from .diagnostics import json_value


def edges(vertices):
    vertices=np.asarray(vertices)
    if len(vertices)<4:
        return np.empty((0,2,3))
    faces=ConvexHull(vertices).simplices
    pairs=np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]))
    pairs=np.unique(np.sort(pairs,axis=1),axis=0)
    return vertices[pairs]


def prepare_view(packet):
    """Convex hull work happens off the planner and never alters certification."""
    a=packet['arrays']; transform=a['geometry_to_base']
    original=a['geometry_vertices_local_m']
    padded=a['geometry_halfspaces_local'].copy()
    padded[:,3]-=packet['metadata']['collision_margin_m']
    expanded=HalfspaceIntersection(padded,original.mean(axis=0)).intersections
    a['collision_hull_edges_base_m']=edges(original)@transform[:3,:3].T+transform[:3,3]
    a['margin_hull_edges_base_m']=edges(expanded)@transform[:3,:3].T+transform[:3,3]
    robot=[]
    for lo,hi in zip(a['robot_geometry_offsets'][:-1],a['robot_geometry_offsets'][1:]):
        segment=edges(a['robot_vertices_base_m'][lo:hi])
        robot.append(segment[::max(1,len(segment)//180)])
    a['robot_edges_base_m']=np.concatenate(robot) if robot else np.empty((0,2,3))
    return packet


def project_base(points, frame, d2c):
    from .measured_rgbd import project_to_rgb
    points=np.asarray(points).reshape(-1,3)
    if not len(points):return np.empty((0,2)),np.empty(0,dtype=bool)
    optical=(points-frame.camera_to_world_cv[:3,3])@frame.camera_to_world_cv[:3,:3]
    uv,positive=project_to_rgb(optical,frame,d2c)
    h,w=frame.rgb.shape[:2]
    visible=positive & np.isfinite(uv).all(axis=1)
    visible &= (uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    return uv,visible


def overlay_collision(rgb, frame, d2c, arrays):
    """Future predicted points projected on the original exposure, not a photo of impact."""
    result=rgb.copy()
    for key,color in [('collision_hull_edges_base_m',(255,140,70)),('margin_hull_edges_base_m',(255,210,70))]:
        for segment in arrays.get(key,()):
            uv,visible=project_base(segment,frame,d2c)
            if visible.all():
                cv2.line(result,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),color,1)
    for key,color,radius in [('object_points_exposure_base_m',(70,190,255),1),
                             ('hit_points_base_m',(255,50,60),4)]:
        points=arrays.get(key,np.empty((0,3)))
        uv,visible=project_base(points[::max(1,len(points)//1500)],frame,d2c)
        for point in uv[visible]:cv2.circle(result,tuple(np.rint(point).astype(int)),radius,color,-1)
    cv2.putText(result,'PREDICTION on frozen exposure / NOT measured impact',(12,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,235,150),1)
    if 'synthetic' in frame.source.lower():
        cv2.putText(result,'SYNTHETIC OFFLINE FIXTURE',(12,44),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,235,150),1)
    return result


def restore_frame(metadata, arrays):
    from mujoco_conveyor_sorting.runtime.types import RGBDFrame,CameraIntrinsics,SampleHeader
    return RGBDFrame(header=SampleHeader(**metadata['header']),depth_m=arrays['frame_depth_m'],
        rgb=arrays.get('frame_rgb'),depth_intrinsics=CameraIntrinsics(**metadata['depth_intrinsics']),
        color_intrinsics=None if metadata['color_intrinsics'] is None else CameraIntrinsics(**metadata['color_intrinsics']),
        camera_to_world_cv=np.asarray(metadata['depth_to_base_cv']),serial_number=metadata['serial_number'],
        calibration_id=metadata['calibration_id'],source=metadata['source'],timestamp_metadata=metadata.get('timestamp_metadata'))


def read_packet(path):
    path=Path(path)
    metadata=json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:
        arrays={key:data[key] for key in data.files}
    frame=None if metadata.get('camera') is None else restore_frame(metadata['camera'],arrays)
    return {'metadata':metadata,'arrays':arrays,'frame':frame,
            'd2c':np.eye(4) if frame is None else np.asarray(metadata['camera']['depth_to_color_cv'])}


def save_packet(directory, packet):
    directory=Path(directory); directory.mkdir(exist_ok=True,parents=True)
    meta=dict(packet['metadata']); arrays=dict(packet['arrays'])
    meta['camera']=frame_metadata(packet['frame'],packet['d2c'])
    meta['robot_feedback_at_exposure']=packet.get('robot_feedback_at_exposure')
    frame=packet['frame']
    if frame is not None:
        arrays.update(frame_depth_m=frame.depth_m)
        if frame.rgb is not None:arrays['frame_rgb']=frame.rgb
    stem=directory/meta['collision_id']
    with stem.with_suffix('.npz.tmp').open('wb') as stream:np.savez(stream,**arrays)
    stem.with_suffix('.npz.tmp').replace(stem.with_suffix('.npz'))
    if frame is not None and frame.rgb is not None:
        image=overlay_collision(frame.rgb,frame,packet['d2c'],arrays)
        if not cv2.imwrite(str(stem.with_suffix('.png')),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):
            raise OSError('碰撞叠加图保存失败')
    temporary=stem.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(json_value(meta),ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(stem.with_suffix('.json'))
    return stem.with_suffix('.json')
