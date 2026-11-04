"""Freeze collision evidence from the exact rejected state, with no file I/O."""
from dataclasses import asdict
import numpy as np


def frame_metadata(frame, depth_to_color):
    if frame is None:
        return None
    return {'header': asdict(frame.header), 'depth_intrinsics': asdict(frame.depth_intrinsics),
        'color_intrinsics': None if frame.color_intrinsics is None else asdict(frame.color_intrinsics),
        'depth_to_base_cv': np.asarray(frame.camera_to_world_cv).tolist(),
        'depth_to_color_cv': np.asarray(depth_to_color).tolist(),
        'serial_number': frame.serial_number, 'calibration_id': frame.calibration_id,
        'source': frame.source, 'timestamp_metadata': frame.timestamp_metadata}


def capture_collision(planner, q, width, points, ids, geom, *, stamp, contact, payload, released, margin_m=.004):
    """Called only on a rejected point-cloud test; do not modify its verdict."""
    model = planner.model
    target = planner.target
    transform = np.eye(4)
    transform[:3, :3] = model.data.geom_xmat[geom].reshape(3, 3)
    transform[:3, 3] = model.data.geom_xpos[geom]
    equations = model.meshes[geom][0]
    hit_points = points[ids]
    local = (hit_points-transform[:3, 3])@transform[:3, :3]
    plane_distance = np.max(local@equations[:, :3].T+equations[:, 3], axis=1)
    frame = target.detection.get('frame')
    d2c = target.detection.get('depth_to_color_cv', np.eye(4))
    planner.collision_record_count += 1
    key = f'collision_{planner.collision_record_count:04d}'
    context = dict(getattr(planner, 'geometry_context', {}))
    origin = context.get('origin_monotonic_s')
    prediction_stamp = stamp if stamp is not None else context.get('sample_monotonic_s')
    name = model.model.geom(geom).name
    detail = {'collision_id':key, 'kind':'robot_workpiece_point_cloud',
        'geometry':name, 'geometry_id':int(geom),
        'robot_body':model.model.body(int(model.model.geom_bodyid[geom])).name,
        'is_finger':geom in model.fingers, 'collision_margin_m':float(margin_m),
        'target_identity':list(target.identity), 'trajectory_version':planner.version,
        'scene_revision':planner.scene_revision, 'clock_epoch':int(target.epoch),
        'frame_sequence':int(target.sequence), 'capture_monotonic_s':float(target.capture_s),
        'target_velocity_base_m_s':np.asarray(target.velocity).tolist(),
        'environment_collision_scope':planner.config.raw.get('environment_collision_scope','observed_scene'),
        'config_sha256':getattr(planner.config,'source_sha256',None),
        'joint_names':planner.config.raw.get('joint_names'),
        'predicted_monotonic_s':prediction_stamp,
        'trajectory_elapsed_s':context.get('sample_elapsed_s') if prediction_stamp is None or origin is None else prediction_stamp-origin,
        'contact_allowed':bool(contact), 'payload':bool(payload), 'released_estimate':bool(released),
        'coordinate_frame':planner.config.raw.get('base_frame','robot_base'),
        'prediction_not_measured_contact':True, 'geometry_context':context,
        'hit_count':len(ids), 'checked_point_count':len(points),
        'first_hit_base_m':hit_points[0].tolist(),
        'inside_nominal_hull_count':int(np.count_nonzero(plane_distance<=0.)),
        'margin_only_count':int(np.count_nonzero(plane_distance>0.)),
        'maximum_plane_distance_range_m':[float(plane_distance.min()),float(plane_distance.max())],
        'gripper_width_m':float(width), 'joints_rad':np.asarray(q).tolist(),
        'tcp_base':model.tcp().tolist(), 'recorded_monotonic_s':planner.clock()}
    vertices, offsets, names = [], [0], []
    for g in model.meshes:
        world = model.mesh_vertices[g]@model.data.geom_xmat[g].reshape(3,3).T+model.data.geom_xpos[g]
        vertices.append(world); offsets.append(offsets[-1]+len(world)); names.append(model.model.geom(g).name)
    packet = {'metadata':detail, 'frame':frame, 'd2c':np.asarray(d2c).copy(),
        'robot_feedback_at_exposure':target.state,
        'arrays':{
            'object_points_exposure_base_m':target.points.copy(),
            'checked_points_base_m':np.asarray(points).copy(),
            'hit_indices':np.asarray(ids, dtype=np.int64).copy(),
            'hit_points_base_m':hit_points.copy(),
            'hit_maximum_plane_distance_m':plane_distance.copy(),
            'scene_points_base_m':model.scene.copy(),
            'geometry_vertices_local_m':model.mesh_vertices[geom].copy(),
            'geometry_halfspaces_local':equations.copy(), 'geometry_to_base':transform,
            'robot_vertices_base_m':np.concatenate(vertices),
            'robot_geometry_offsets':np.asarray(offsets,dtype=np.int64),
            'robot_geometry_names':np.asarray(names,dtype=str)}}
    if planner.collision_sink is not None:
        planner.collision_sink(packet)
    return detail
