"""Single-exposure depth trigger for the independent camera test interface.

No tracker, device, CAN adapter or robot command is owned by this module.
The locator uses a wide search crop; the 2 cm band selects a component's
centre and never slices the workpiece cloud into a narrow strip.
"""
from collections import deque
from dataclasses import dataclass,replace
from threading import Condition
import time

import cv2
import numpy as np

from mujoco_conveyor_sorting.perception import DepthGeometryVisionPipeline, GeometryPrototype
from .reference import EntryNotReady
from .synchronized_reference import admit_synchronized_observation, freeze_window_for_speed


class ExposureBuffer:
    """Bounded original-frame FIFO, independent of expensive geometry/planning."""
    def __init__(self, capacity=16):
        if isinstance(capacity,bool) or not isinstance(capacity,int) or not 3<=capacity<=90:
            raise ValueError('buffer frames must be an integer from 3 to 90')
        self.capacity=capacity
        self.frames=deque()
        self.condition=Condition()
        self.closed=False
        self.received=self.dropped=self.sequence_skips=0
        self.previous=None

    def put(self,frame):
        key=(frame.header.clock_epoch,frame.header.sequence)
        with self.condition:
            if self.closed:return
            if self.previous is not None and key[0]==self.previous[0]:
                if key[1]<=self.previous[1]:return
                self.sequence_skips+=max(0,key[1]-self.previous[1]-1)
            if len(self.frames)==self.capacity:
                self.frames.popleft();self.dropped+=1
            self.frames.append(frame);self.previous=key;self.received+=1
            self.condition.notify_all()

    def take(self,timeout=.2):
        end=time.monotonic()+timeout
        with self.condition:
            while not self.frames and not self.closed:
                left=end-time.monotonic()
                if left<=0:return None
                self.condition.wait(left)
            return self.frames.popleft() if self.frames else None

    def close(self):
        with self.condition:
            self.closed=True;self.condition.notify_all()

    def diagnostics(self):
        with self.condition:
            return dict(capacity=self.capacity,queued=len(self.frames),received=self.received,
                        dropped=self.dropped,capture_sequence_skips=self.sequence_skips)


def capture_admission(config,capture_s,position,velocity,now_s,*,belt_region=None):
    """Check the true saved exposure, without projecting or retimestamping it."""
    if not 0<=now_s-capture_s<=config.raw['maximum_observation_age_s']:
        raise EntryNotReady('原始曝光已过期或来自未来','stale_exposure')
    window=freeze_window_for_speed(config,float(velocity@config.direction),belt_region=belt_region)
    minimum = window.get('minimum_distance_m',config.raw.get('tracked_freeze_publish_before_upstream_m'))
    return admit_synchronized_observation(capture_s,float(position@config.direction),
        float(velocity@config.direction),config.follow_bounds,now_s,window['distance_m'],
        minimum_upstream_distance_m=minimum)


def _clip_polygon(polygon,axis,limit,keep_greater):
    result=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        da,db=float(a@axis-limit),float(b@axis-limit)
        ina=da>=0 if keep_greater else da<=0
        inb=db>=0 if keep_greater else db<=0
        if ina:result.append(a)
        if ina!=inb:result.append(a+(b-a)*(da/(da-db)))
    return np.asarray(result).reshape(-1,3)


@dataclass(frozen=True)
class DepthCandidate:
    position: np.ndarray
    point_count: int
    seed_pixel_indices: np.ndarray
    along_bounds_m: tuple | None = None
    prepared: object | None = None


def fixed_freeze_detection_surface(surface):
    """Use the visible belt outline for detection, with native hull fallback.

    The native-depth 2/98% quantiles are useful for the original tracker but
    can exclude a configured freeze line that still has measured belt depth.
    The visible outline avoids inheriting the plane fit mask's erosion.
    This private detection copy adds no points and leaves the original
    plane, native hull, scene boundary and collision geometry untouched.
    """
    visible=getattr(surface,'detection_hull_xy',None)
    detection_hull=np.asarray(surface.hull_xy if visible is None else visible,dtype=np.float32)
    hull=detection_hull.reshape(-1,2)
    bounds=np.column_stack((hull.min(axis=0),hull.max(axis=0)))
    bounds[:,0]=np.minimum(bounds[:,0],surface.bounds_xy_m[:,0])
    bounds[:,1]=np.maximum(bounds[:,1],surface.bounds_xy_m[:,1])
    return replace(surface,bounds_xy_m=bounds,hull_xy=detection_hull,
        scene_bounds_xy_m=(surface.bounds_xy_m.copy() if surface.scene_bounds_xy_m is None
                           else surface.scene_bounds_xy_m))


class NativeDepthLocator:
    """Fast single-frame native-depth components around the freezing line.

    This is a trigger only. The selected exposure subsequently passes the
    existing complete workpiece/colour/geometry detector before planning.
    """
    def __init__(self,config,surface,frame,*,capture_band_m=None):
        self.original_bounds=surface.bounds_xy_m.copy()
        surface=fixed_freeze_detection_surface(surface)
        self.config,self.surface=config,surface
        if capture_band_m is None:
            opening = config.raw['freeze_position_upstream_m']
            width = config.raw['freeze_position_tolerance_m']
            self.crop_downstream_m = .22
        else:
            cutoff, opening = capture_band_m
            width = opening-cutoff
            self.crop_downstream_m = .20+width
        self.line = float(config.follow_bounds[0]-opening)
        self.cutoff = self.line+width
        if np.any(np.abs(frame.depth_intrinsics.distortion)>1.e-10):
            raise ValueError('固定冻结测试需要零畸变原生深度流')
        self.crop=self._crop(frame)
        y0,y1,x0,x1=self.crop
        y,x=np.mgrid[y0:y1,x0:x1]
        k=frame.depth_intrinsics
        rays=np.stack(((x-k.cx)/k.fx,(y-k.cy)/k.fy,np.ones_like(x)),axis=-1)
        self.rays_base=rays@frame.camera_to_world_cv[:3,:3].T
        self.origin=np.asarray(frame.camera_to_world_cv[:3,3])
        from .tracked_freeze import native_exposure_window
        seed_length=config.raw.get('maximum_target_length_m', .16)
        if capture_band_m is not None and native_exposure_window(config):
            # This rough component is measured before RGB background removal and
            # may be diagonal to the cleaned footprint. Only seed selection uses
            # the diagonal bound; the complete detector still checks actual size.
            seed_length=float(np.hypot(seed_length, config.settings.open_width_m))
        self.pipeline=DepthGeometryVisionPipeline(
            [GeometryPrototype(category_id=0,dimensions=(.05,.05,.05),symmetric=True)],
            support_plane_height=0.,roi_x=surface.bounds_xy_m[0],roi_y=surface.bounds_xy_m[1],
            minimum_height=.012,maximum_height=config.raw['maximum_target_height_m'],
            minimum_mask_pixels=70,minimum_footprint=.012,
            maximum_footprint=seed_length,
            classification_enabled=False)
        self.intrinsics=k.matrix.copy();self.intrinsics[:2,2]-=[x0,y0]

    def _crop(self,frame):
        plane=self.surface.plane_to_base
        bounds=self.surface.bounds_xy_m
        rectangle=np.array([[bounds[0,0],bounds[1,0],0.],
                            [bounds[0,1],bounds[1,0],0.],
                            [bounds[0,1],bounds[1,1],0.],
                            [bounds[0,0],bounds[1,1],0.]])
        rectangle=rectangle@plane[:3,:3].T+plane[:3,3]
        coordinate=self.line
        # Preserve the legacy crop at 16cm; larger supported parts need enough
        # room for their complete native-depth component, never a sliced cloud.
        extra=max(0., self.config.raw.get('maximum_target_length_m', .16)-.16)
        polygon=_clip_polygon(rectangle,self.config.direction,coordinate-.20-extra,True)
        if len(polygon):polygon=_clip_polygon(polygon,self.config.direction,coordinate+self.crop_downstream_m+extra,False)
        if not len(polygon):raise ValueError('冻结线及搜索范围没有落在实测传送带视野内')
        prism=np.vstack((polygon,polygon+plane[:3,2]*self.config.raw['maximum_target_height_m']))
        optical=(prism-frame.camera_to_world_cv[:3,3])@frame.camera_to_world_cv[:3,:3]
        if np.any(optical[:,2]<=0):raise ValueError('冻结搜索范围位于相机后方')
        uv=optical[:,:2]/optical[:,2,None]
        k=frame.depth_intrinsics
        uv=uv*[k.fx,k.fy]+[k.cx,k.cy]
        lo=np.floor(uv.min(axis=0)-4).astype(int);hi=np.ceil(uv.max(axis=0)+5).astype(int)
        x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[k.width,k.height])
        if x1<=x0 or y1<=y0:raise ValueError('冻结搜索范围不在原生深度画面内')
        return int(y0),int(y1),int(x0),int(x1)

    def locate(self,frame,*,robot_model=None,robot_state=None):
        started=time.perf_counter()
        y0,y1,x0,x1=self.crop
        z=frame.depth_m[y0:y1,x0:x1]
        base=self.rays_base*z[...,None]+self.origin
        plane=self.surface.plane_to_base
        local=(base-plane[:3,3])@plane[:3,:3]
        foreground=np.isfinite(z)&(z>0)&np.isfinite(local).all(axis=2)
        foreground&=(local[:,:,2]>=.012)&(local[:,:,2]<=self.config.raw['maximum_target_height_m'])
        projected=time.perf_counter()
        if (robot_model is None)!=(robot_state is None):
            raise ValueError('自身点云过滤需要真实模型和配对反馈同时提供')
        removed=0
        if robot_model is not None and foreground.any():
            filtered=robot_model.classify_robot_points(base[foreground],robot_state,
                camera_origin=frame.camera_to_world_cv[:3,3])
            removed=int(np.count_nonzero(filtered.removed_mask))
            clean=foreground.copy();clean[foreground]=~filtered.removed_mask;foreground=clean
        filtered_at=time.perf_counter()
        components=self.pipeline.extract_components(None,z,self.intrinsics,np.eye(4),
            points_world=local,foreground_mask=foreground,roi_selects_components=True)
        extracted=time.perf_counter()
        candidates=[];partial=0;near_zone_components=0;size_decisions=[]
        for component in components:
            decision={'estimated_footprint_m':component.feature_vector[:2].tolist(),
                      'maximum_length_m':self.pipeline.maximum_footprint, 'accepted':False}
            size_decisions.append(decision)
            mask=component.mask
            edge_distance=cv2.pointPolygonTest(self.surface.hull_xy,tuple(map(float,component.position[:2])),True)
            edge_margin=self.config.raw.get('perception_belt_edge_margin_m', .025)
            decision.update(belt_edge_distance_m=float(edge_distance),belt_edge_margin_m=edge_margin)
            if edge_distance<=edge_margin:
                decision['reason']='belt_edge'
                continue
            near_margin=self.config.raw.get('perception_belt_near_camera_margin_m', 0.)
            if near_margin>0:
                from mujoco_conveyor_sorting.runtime.belt_surface import near_camera_belt_clearance
                near_distance=near_camera_belt_clearance(self.surface,component.position[:2],
                    frame.camera_to_world_cv[:3,3],self.config.direction)
                decision.update(near_camera_edge_distance_m=near_distance,near_camera_edge_margin_m=near_margin)
                if near_distance<=near_margin+1e-9:
                    decision['reason']='near_camera_belt_edge'
                    continue
            rows,cols=np.nonzero(mask)
            position=np.median(base[mask],axis=0)
            if self.line-.08<=position@self.config.direction<=self.cutoff+.08:near_zone_components+=1
            if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
                decision['reason']='partial_component_at_crop'
                partial+=1;continue
            decision.update(accepted=True,reason='size_and_visibility_passed',
                            distance_to_entry_m=float(self.config.follow_bounds[0]-position@self.config.direction))
            along = base[mask] @ self.config.direction
            candidates.append(DepthCandidate(position,len(rows),
                (rows+y0)*frame.depth_intrinsics.width+cols+x0, (float(along.min()), float(along.max()))))
        finished=time.perf_counter()
        return candidates,dict(search_crop_yxyx=self.crop,candidate_count=len(candidates),
            timings_s=dict(depth_projection=projected-started,
                robot_self_filter=filtered_at-projected,connected_components=extracted-filtered_at,
                candidate_selection=finished-extracted,total=finished-started),
            detection_bounds_xy_m=self.surface.bounds_xy_m.tolist(),
            original_quantile_bounds_xy_m=self.original_bounds.tolist(),
            detection_bounds_source=('visible_green_outline_on_measured_plane'
                if self.surface.detection_hull_xy is not None else 'measured_native_belt_hull'),
            detection_boundary=self.surface.detection_boundary_diagnostics,
            partial_components=partial,evaluated_depth_pixels=int(z.size),
            native_depth_pixels=int(frame.depth_m.size),
            native_valid_depth_ratio=float(np.mean(np.isfinite(z)&(z>0))),
            near_freeze_zone_components=near_zone_components,robot_points_removed=removed,
            maximum_target_length_m=self.pipeline.maximum_footprint,
            size_decisions=size_decisions[:32],
            component_extraction=self.pipeline.last_component_workload)


def along_belt_quantile_grasps(grasps, box, points_base, plane, direction):
    """Move the candidate TCPs along base XY only, keeping the measured OBB as object frame."""
    heading = np.array(direction[:2], dtype=float)
    length = np.linalg.norm(heading)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError('沿带抓取中心需要有效的基座 XY 带向')
    heading /= length
    local_xy = (points_base - plane[:3, 3]) @ plane[:3, :2]
    axes = np.asarray(box['axes_world_columns'])[:2, :2]
    bounds = np.quantile(local_xy @ axes, [.02, .98], axis=0)
    robust_xy = bounds.mean(axis=0) @ axes.T
    delta_plane = robust_xy - np.asarray(box['center_world_m'])[:2]
    shift = float(heading @ (plane[:2, :2] @ delta_plane))
    offset = np.r_[shift * heading, 0.]
    adjusted = [g.copy() for g in grasps]
    for g in adjusted:
        g[:3, 3] += offset
    return adjusted, dict(box,
        grasp_tcp_offset_base_m=(np.asarray(box['grasp_tcp_offset_base_m']) + offset).tolist(),
        along_belt_center=dict(quantiles=[.02, .98], point_count=len(points_base),
            signed_shift_m=shift, original_tcp_base_m=grasps[0][:3, 3].tolist(),
            adjusted_tcp_base_m=adjusted[0][:3, 3].tolist()))


def frozen_geometry(config,frame,d2c,surface,candidate,capture_s,*,clock=time.monotonic,
                    robot_model=None,robot_state=None,detect_fn=None,grasp_fn=None,fast_stereo=None):
    """Shared complete single-exposure geometry for camera tests and hardware.

    Hardware passes measured feedback and its private model; camera-only
    callers never supply an assumed state as an actual observation.
    """
    from .measured_rgbd import fit_plane,green_belt_points
    from .velocity_source import prediction_velocity
    from .vision import obb_grasps, offset_grasp_tcp_z, record_grasp_box
    from .tracked_freeze import native_exposure_window
    from mujoco_conveyor_sorting.runtime.single_pick import detect_target,PickRejected
    if (robot_model is None)!=(robot_state is None):
        raise ValueError('完整工件自身过滤需要配对反馈')
    if config.raw.get('fast_stereo_enabled', False) and fast_stereo is None:
        raise PickRejected('Fast冻结缺少已预热的重建后端', details={'reason':'fast_stereo_worker_unavailable'})
    detection_surface=fixed_freeze_detection_surface(surface)
    prepared=candidate.prepared
    detector=detect_fn or detect_target
    if prepared is None and detect_fn is None and config.raw.get('local_target_geometry_enabled', True):
        from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
        original_detector=detector
        detector=lambda *args,**kwargs: detect_seed_local(*args,detector=original_detector,**kwargs)
    if prepared is not None:
        if (prepared.frame is not frame or prepared.surface is not surface or prepared.state is not robot_state
                or prepared.model is not robot_model or prepared.config is not config
                or not np.array_equal(prepared.d2c, d2c)
                or config.raw.get('rgb_cloud_completion_enabled',False)):
            raise PickRejected('原生候选与曝光、标定、反馈或几何模式不一致')
        detection=prepared.geometry
    else:
        detection=detector(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
            retain_native_seed=config.raw.get('depth_separation_enabled',False),
            rgb_cloud_completion=config.raw.get('rgb_cloud_completion_enabled',False),
            **({'minimum_complete_height_m':config.raw['native_single_min_height_m']}
               if native_exposure_window(config) else {}),
            target_shape='simple',measured_surface=detection_surface,robot_model=robot_model,robot_state=robot_state,
            target_seed_pixel_indices=candidate.seed_pixel_indices,
            maximum_target_height_m=config.raw['maximum_target_height_m'],
            maximum_target_length_m=config.raw.get('maximum_target_length_m', .16),
            belt_edge_margin_m=config.raw.get('perception_belt_edge_margin_m', .025),
            belt_near_camera_margin_m=config.raw.get('perception_belt_near_camera_margin_m', 0.),
            belt_direction_base=config.direction,
            scene_scope=config.raw['environment_collision_scope'])
    if fast_stereo is not None:
        detection=fast_stereo.refine(detection,frame,d2c,robot_model=robot_model,robot_state=robot_state,
                                     surface=detection_surface)
    from .tracked_freeze import check_near_camera_footprint
    check_near_camera_footprint(config, surface, frame, detection)
    overlap=None
    if prepared is None or fast_stereo is not None:
        overlap=float(np.mean(detection['mask'].ravel()[candidate.seed_pixel_indices]))
        if overlap<.5:
            raise PickRejected('完整工件分割与本帧触发区域不一致',details={'seed_overlap_ratio':overlap})
    # Downstream planning retains the original measured scene geometry. Only
    # target search used the wider native-depth hull, never inferred depth.
    detection['belt_region']={**surface.region_metadata(),
        'target_search_bounds_xy_m':detection_surface.bounds_xy_m.copy(),
        'target_search_bounds_source':('visible_green_outline_on_measured_plane'
            if detection_surface.detection_hull_xy is not None else 'measured_native_belt_hull')}
    position=(candidate.position if prepared is not None and fast_stereo is None
              else np.median(detection['object_points'],axis=0))
    velocity=prediction_velocity(config,None,surface.normal_base)
    capture_admission(config,capture_s,position,velocity,clock(),belt_region=detection['belt_region'])
    from .depth_separation import grasp_points
    obb_points=grasp_points(detection,frame,enabled=config.raw.get('depth_separation_enabled',False))
    grasps,box=(grasp_fn or obb_grasps)(obb_points,surface.plane_to_base,config.canonical_tcp)
    grasps,box=offset_grasp_tcp_z(grasps,box,config.raw.get('grasp_tcp_z_offset_m', 0.))
    if (config.raw.get('freeze_source') == 'single_exposure_native_depth'
            and config.raw.get('native_single_along_belt_quantile_center', False)):
        grasps,box=along_belt_quantile_grasps(grasps,box,obb_points,surface.plane_to_base,config.direction)
    if prepared is not None:
        scene,ids,valid=prepared.scene()
        detection.update(scene_points=scene,scene_pixel_indices=ids,
            scene_input_count_before_prefilter=valid-len(prepared.geometry['object_points'])-detection['excluded_depth_pixels'])
    record_grasp_box(detection, box, grasps, position)
    detection.update(frame=frame,depth_to_color_cv=d2c,scene_state=robot_state)
    return detection,position,velocity,tuple(grasps),overlap
