#!/usr/bin/env python3
"""Recover one frozen RGB-D exposure and audit logged hits offline; no devices.

Only preview/target.npz is accepted, paired against qualified_target. Never use
preview/latest.npz as a substitute: it may belong to a later exposure.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

# Reuse the plotting runtime fix for mixed pip/system mpl_toolkits installs.
from render_saved_scene_collision import plt, FontProperties, fontManager, equal_limits
from nero_core.two_second.collision_view import restore_frame, project_base
from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.single_pick import frame_points, PickRejected


def frozen_input(run):
    path = run / 'preview/target.npz'
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        frame = restore_frame(meta, dict(frame_rgb=data['frame_rgb'], frame_depth_m=data['frame_depth_m']))
        target = data['points'].copy()
    events = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    qualified = [e['data'] for e in events if e.get('event') == 'qualified_target'
                 and e['data']['identity'] == meta['identity']]
    if len(qualified) != 1:
        raise ValueError('无法唯一关联冻结目标，拒绝用其他帧替代')
    q = qualified[0]
    if (q['sequence'] != frame.header.sequence or q['capture_monotonic_s'] != meta['capture_s']
            or q['source_device_s'] != frame.header.source_time
            or meta['identity'][0] != frame.header.clock_epoch):
        raise ValueError('冻结帧时间/序号/时钟代次不匹配')
    return path, meta, frame, target


def write_ply(path, xyz, colors, *, pixels=None):
    # Preserve double precision and original depth pixel provenance. Projection
    # colours are for viewing only and cannot identify a point through occlusion.
    fields = [('x','<f8'),('y','<f8'),('z','<f8'),('red','u1'),('green','u1'),('blue','u1')]
    if pixels is not None: fields += [('depth_u','<i4'),('depth_v','<i4')]
    records = np.empty(len(xyz), dtype=fields)
    for i, name in enumerate(('x','y','z')): records[name] = xyz[:,i]
    for i, name in enumerate(('red','green','blue')): records[name] = colors[:,i]
    header = ['ply','format binary_little_endian 1.0','comment robot_base; coordinates in metres',
              'comment Native positive finite depth only; no denoising, filling or self filter',
              'comment RGB by calibrated projection; occlusion can cause colour mismatch',
              f'element vertex {len(xyz)}','property double x','property double y','property double z',
              'property uchar red','property uchar green','property uchar blue']
    if pixels is not None:
        records['depth_u'] = pixels[:,1]; records['depth_v'] = pixels[:,0]
        header += ['property int depth_u','property int depth_v']
    with path.open('wb') as stream:
        stream.write(('\n'.join(header+['end_header',''])).encode('ascii'))
        stream.write(records.tobytes())


def belt_only_saved_states(config, frame, d2c, failures):
    """Explicit offline ablation, not a live planner policy or path certificate."""
    surface = BeltSurface.from_frame(frame, d2c, fit_plane=fit_plane, green_belt_points=green_belt_points)
    model = calibrated_model(config)
    model.scene = np.empty((0,3)); model.tree = cKDTree(model.scene)
    model.target = np.empty((0,3))
    results = []
    for failure in failures:
        d = failure['details']
        model_check = {'passed':True}
        try:
            model.check_state(d['joints_rad'], d['width_m'], allow_finger_contact=True, check_scene=False)
        except PickRejected as error:
            model_check = {'passed':False, 'error':str(error)}
        minima, hits = {}, []
        # Same gripper-vertex/support-plane predicate as MeasuredPlanner.valid.
        # This does NOT claim continuous mesh/belt intersection certification.
        for g in model.gripper_geometries if model_check['passed'] else ():
            world = model.mesh_vertices[g] @ model.data.geom_xmat[g].reshape(3,3).T + model.data.geom_xpos[g]
            local = (world-surface.plane_to_base[:3,3]) @ surface.plane_to_base[:3,:3]
            over = ((local[:,:2] >= surface.bounds_xy_m[:,0]) &
                    (local[:,:2] <= surface.bounds_xy_m[:,1])).all(axis=1)
            if over.any(): minima[model.model.geom(g).name] = float(local[over,2].min())
            if (over & (local[:,2] < -1e-6)).any(): hits.append(model.model.geom(g).name)
        results.append({'candidate_number':failure['candidate_index']+1,
            'saved_trajectory_time_s':failure['last_progress']['trajectory_time_s'],
            'original_scene_alarm_retained':d, 'remaining_model_checks':model_check,
            'belt_gripper_vertex_check_passed':not hits if model_check['passed'] else None,'belt_violations':hits,
            'minimum_gripper_vertex_height_over_belt_m':minima})
    return {'scope':'offline_saved_states_only','obstacle_scope':'recomputed_measured_belt_plane',
        'plane_source':'recomputed_from_frozen_frame_not_original_cached_plane',
        'belt_region':surface.region_metadata(), 'cases':results,
        'continuous_trajectory_checked':False,'belt_frame_rails_modelled':False,
        'hardware_configuration_changed':False,'hardware_execution_authorized':False}


def figures(out, frame, point, xyz, colors, uv, pixels, distances, report):
    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        fontManager.addfont(str(font)); plt.rcParams['font.family'] = FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11})
    xy = uv[0]; u,v = report['depth_pixel_uv']
    fig = plt.figure(figsize=(16,10),facecolor='#f7f9fc')
    grid = fig.add_gridspec(2,2,left=.055,right=.96,top=.88,bottom=.16,hspace=.30,wspace=.18)
    rgb = fig.add_subplot(grid[0,0]); raw = fig.add_subplot(grid[0,1],projection='3d',computed_zorder=False)
    depth = fig.add_subplot(grid[1,0]); zoom = fig.add_subplot(grid[1,1],projection='3d',computed_zorder=False)
    rgb.imshow(frame.rgb); rgb.scatter(*xy,s=170,facecolors='none',edgecolors='#ff253a',linewidths=2)
    rgb.set_title('① 同一冻结帧 RGB：红圈为告警点的标定投影'); rgb.set_axis_off()
    view = ((xyz >= [-.75,-.7,-.30]) & (xyz <= [.25,.85,.65])).all(axis=1)
    displayed = np.flatnonzero(view)[::max(1,int(view.sum())//60000)]
    raw.scatter(*xyz[displayed].T,s=.7,c=colors[displayed]/255.,depthshade=False,rasterized=True)
    raw.scatter(*point,c='red',s=55,depthshade=False,edgecolor='white',zorder=50)
    raw.set_xlim(-.75,.25);raw.set_ylim(-.7,.85);raw.set_zlim(-.30,.65);raw.set_box_aspect((1,1.55,.95))
    raw.view_init(elev=25,azim=-52);raw.set_title('② 原始点云：作业区视图（仅显示抽样）')
    window = 18
    image = np.ma.masked_less_equal(frame.depth_m[max(0,v-window):v+window+1,max(0,u-window):u+window+1]*1000,0)
    cmap = plt.get_cmap('viridis').copy();cmap.set_bad('#dadee6')
    plot = depth.imshow(image,origin='upper',cmap=cmap,interpolation='nearest',
        extent=(u-window-.5,u+window+.5,v+window+.5,v-window-.5))
    depth.scatter(u,v,s=160,facecolors='none',edgecolors='#ff253a',linewidths=2)
    depth.set_xlabel('深度图 u / pixel');depth.set_ylabel('深度图 v / pixel')
    depth.set_title('③ 原始深度局部：灰色为无效深度（未补点）')
    fig.colorbar(plot,ax=depth,label='原生深度 / mm',fraction=.035,pad=.02)
    local = distances <= .04
    zoom.scatter(*xyz[local].T*1000,c=colors[local]/255.,s=15,depthshade=False)
    zoom.scatter(*point*1000,c='red',s=95,depthshade=False,edgecolor='white',zorder=50)
    equal_limits(zoom,xyz[local]*1000,1.1);zoom.view_init(elev=27,azim=-52)
    zoom.set_title(f'④ 告警点周围 40 mm：{int(local.sum())} 个原始点')
    for ax,unit in ((raw,'m'),(zoom,'mm')):
        ax.set_xlabel('X / '+unit);ax.set_ylabel('Y / '+unit);ax.set_zlabel('Z / '+unit)
        ax.tick_params(labelsize=8,pad=0)
    fig.suptitle(f'当时的真实点云：第 {frame.header.sequence} 帧，非模型补全、非仿真点云',fontsize=21,y=.965)
    fig.text(.055,.917,f'目标 {report["target_identity"]} · 设备深度时间 {frame.header.source_time:.6f} s · 红点来自首个场景告警',fontsize=11,color='#4e6174')
    fig.text(.055,.062,f'有效原生深度 {len(xyz):,} 点；目标 {report["target_points"]:,} 点；5 mm 内 {report["neighbors_including_self"]["0.005"]} 点。整帧 PLY 保留所有有效点。',fontsize=11)
    fig.text(.055,.024,'颜色由真实标定投影，边缘处可能有遮挡错色；单帧不能证明噪声或真实障碍。未连接设备，未修改真机检查。',fontsize=10,color='#4e6174')
    fig.savefig(out/'pointcloud_overview.png',dpi=160);plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,6))
    ax.imshow(frame.rgb);ax.scatter(*xy,s=330,facecolors='none',edgecolors='red',linewidths=2)
    ax.set_xlim(xy[0]-140,xy[0]+140);ax.set_ylim(xy[1]+110,xy[1]-110)
    ax.set_title('原始 RGB 局部与告警坐标投影（未去噪）');ax.set_axis_off()
    fig.tight_layout();fig.savefig(out/'rgb_trigger_crop.png',dpi=160);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    source,meta,frame,target = frozen_input(args.run)
    summary_path = args.run/'summary.json'; summary = json.loads(summary_path.read_text())
    failures = [f for f in summary['details']['failures'] if (f.get('details') or {}).get('kind') == 'scene_collision']
    if not failures: raise ValueError('没有可匹配的场景告警')
    point = np.asarray(failures[0]['details']['points_base_m'][0])
    all_points = frame_points(frame)
    valid = (frame.depth_m>0) & np.isfinite(all_points).all(axis=2)
    xyz = all_points[valid]; pixels = np.argwhere(valid)
    distances = np.linalg.norm(xyz-point,axis=1); index = int(distances.argmin())
    if distances[index] > 1e-9:
        raise ValueError('告警点不在冻结帧原生点云中，不能将这帧标为告警来源')
    v,u = pixels[index]
    d2c = np.asarray(meta['depth_to_color_cv'])
    uv,visible = project_base(xyz,frame,d2c)
    colors = np.full((len(xyz),3),150,dtype=np.uint8)
    # Round only after clipping; project_base's continuous bounds can round
    # the last fractional pixel beyond the image edge.
    projected = np.rint(uv[visible]).astype(int)
    projected[:,0] = np.clip(projected[:,0],0,frame.rgb.shape[1]-1)
    projected[:,1] = np.clip(projected[:,1],0,frame.rgb.shape[0]-1)
    colors[visible] = frame.rgb[projected[:,1],projected[:,0]]
    hit_uv,hit_visible = project_base(point[None,:],frame,d2c)
    if not hit_visible[0]: raise ValueError('告警点不在原始RGB有效投影范围')
    patch = frame.depth_m[v-2:v+3,u-2:u+3]; positive = patch[patch>0]
    report = {'scope':'offline_frozen_rgbd_recovery','source_run':str(args.run.resolve()),
        'source_frame':str(source.resolve()),'source_sha256':sha256(source.read_bytes()).hexdigest(),
        'source_summary_sha256':sha256(summary_path.read_bytes()).hexdigest(),
        'header':meta['header'],'target_identity':meta['identity'],'capture_monotonic_s':meta['capture_s'],
        'raw_valid_points':len(xyz),'target_points':len(target),'point_base_m':point.tolist(),
        'depth_pixel_uv':[int(u),int(v)],'rgb_projection_uv':hit_uv[0].tolist(),
        'match_distance_m':float(distances[index]),'depth_m':float(frame.depth_m[v,u]),
        'native_depth_5x5_m':patch.tolist(),'native_depth_5x5_valid_count':len(positive),
        'native_depth_5x5_valid_range_m':[float(positive.min()),float(positive.max())],
        'neighbors_including_self':{str(r):int((distances<=r).sum()) for r in [.003,.005,.01,.02,.03,.04]},
        'noise_classification':'undetermined_single_exposure','original_filtered_scene_saved':False,
        'hardware_connections':0,'commands_sent':0,'hardware_configuration_changed':False}
    config = CellConfig.load(args.config)
    report['cell_config_sha256'] = config.source_sha256
    report['belt_only_offline_comparison'] = belt_only_saved_states(config,frame,d2c,failures)
    args.output.mkdir(parents=True,exist_ok=False)
    write_ply(args.output/'native_pointcloud.ply',xyz,colors,pixels=pixels)
    local = distances <= .04
    write_ply(args.output/'trigger_neighborhood.ply',xyz[local],colors[local],pixels=pixels[local])
    cv2.imwrite(str(args.output/'raw_rgb.png'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
    figures(args.output,frame,point,xyz,colors,hit_uv,pixels,distances,report)
    from nero_core.two_second.diagnostics import json_value
    (args.output/'report.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2,allow_nan=False))
    print(f'[原始点云] {args.output.resolve()}/pointcloud_overview.png')
    print(f'[离线核对] {len(xyz)} 点；深度像素 ({u}, {v})；匹配误差 {distances[index]:g} m；设备/命令 0')


if __name__ == '__main__': main()
