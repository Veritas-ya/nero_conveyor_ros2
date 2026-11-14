"""Experimental model-based 20 cm drop clearance, not physical certification."""
from hashlib import sha256
from pathlib import Path
import numpy as np
import yaml

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def load_guard(path, config, taught):
    path=Path(path).expanduser().resolve(strict=True)
    content=path.read_bytes(); data=yaml.safe_load(content)
    if not isinstance(data,dict) or data.get('source')!='operator_fingertip_model_estimate':
        raise ValueError('料框高度文件必须明确说明指尖名义估计来源')
    if data.get('base_frame') != config.raw['base_frame']:
        raise ValueError('料框参考坐标系与当前基座坐标系不一致')
    from .taught import model_digest
    if data.get('robot_model_sha256') != model_digest():
        raise ValueError('料框指尖模型与当前机械臂模型不一致')
    if data.get('taught_transfer_sha256') != taught.sha256:
        raise ValueError('料框参考投放点与投放示教文件不一致')
    tcp_hash=sha256(Path(config.raw['tcp_calibration_file']).read_bytes()).hexdigest()
    if data.get('tcp_sha256') != tcp_hash:
        raise ValueError('料框参考指尖模型与当前 TCP 不一致')
    center=np.asarray(data.get('drop_tcp_xyz_base_m'),float)
    if center.shape!=(3,) or not np.isfinite(center).all():
        raise ValueError('料框参考投放点必须是三个有限坐标')
    values=[]
    for name in ('radius_m','rim_z_m','margin_m'):
        value=data.get(name)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value):
            raise ValueError(f'料框参考 {name} 必须是有限数值')
        values.append(float(value))
    radius,rim,margin=values
    if radius<=0 or margin<=0:
        raise ValueError('料框水平半径和预留间隙必须为正')
    from .planning import calibrated_model
    model=calibrated_model(config)
    model.set_state(taught.joints[-1],config.settings.open_width_m)
    if np.linalg.norm(model.tcp()[:3,3]-center)>1e-6:
        raise ValueError('名义投放点与当前机械臂模型及 TCP 不匹配')
    return {'source_file':str(path),'source_sha256':sha256(content).hexdigest(),
            'source':data['source'],'drop_tcp_xyz_base_m':center.tolist(),
            'radius_m':radius,'rim_z_m':rim,'margin_m':margin,
            'required_min_z_m':rim+margin,'taught_transfer_sha256':taught.sha256,
            'tcp_sha256':tcp_hash,'physical_calibration_verified':False}


def check_guard(planner, tcp, *, payload):
    guard=planner.config.raw.get('experimental_direct_drop_guard')
    if not guard or not payload:
        return
    center=np.asarray(guard['drop_tcp_xyz_base_m'])
    distance=float(np.linalg.norm(tcp[:2,3]-center[:2]))
    if distance>guard['radius_m']:
        return
    planner.stats['bin_clearance_checks'] = planner.stats.get('bin_clearance_checks',0)+1
    if planner.payload_local is None or len(planner.payload_local)==0:
        raise PickRejected('料框高度检查缺少本次载物点云',details={'kind':'bin_clearance_missing_payload'})
    lowest=float('inf');name=None
    for geom in planner.model.gripper_geometries:
        rotation=planner.model.data.geom_xmat[geom].reshape(3,3)
        position=planner.model.data.geom_xpos[geom]
        height=float(np.min(planner.model.mesh_vertices[geom]@rotation[2,:]+position[2]))
        if height<lowest:lowest,name=height,planner.model.model.geom(geom).name
    height=float(np.min(planner.payload_local@tcp[2,:3]+tcp[2,3]))
    if height<lowest:lowest,name=height,'estimated_payload'
    required=guard['required_min_z_m']
    if lowest < required:
        planner.stats['bin_clearance_rejections'] = planner.stats.get('bin_clearance_rejections',0)+1
        raise PickRejected('投放区域内夹爪或载物低于名义框沿和预留高度',details={
            'kind':'experimental_bin_clearance','geometry':name,'lowest_z_m':lowest,
            'required_z_m':required,'distance_xy_m':distance,'radius_m':guard['radius_m'],
            'source_sha256':guard['source_sha256']})
