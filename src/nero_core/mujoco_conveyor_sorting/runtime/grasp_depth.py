"""Reconstruct the selected visible target without changing native camera IO.

Only the geometry supplied to GraspGen-X and its target collision model changes.
The captured RGB-D frame, obstacle cloud, support plane and Nero7 input remain
the original device products. A reconstruction failure retains native geometry
and records the reason; it does not create a task rejection.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .artifact_writer import background_artifact

from ..calibration.geometry import inverse
from ..perception import deproject_depth_to_world
from .shape_pick import filter_target_mask
from .depth_diagnostics import measured_stage, nested_events, trace_context


def _json(value):
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _points_sha(points):
    # GraspGen-X's adapter converts its immutable input to this same dtype.
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def save_stereo_capture(stem, frame):
    """Save authentic same-capture IR beside the unchanged native RGB-D files."""
    if frame.left_ir is None or frame.right_ir is None:
        return {'available': False, 'reason': 'same_capture_infrared_unavailable'}
    if frame.left_ir_intrinsics is None or frame.right_ir_intrinsics is None:
        return {'available': False, 'reason': 'stereo_calibration_unavailable'}
    stem = Path(stem)
    files = {'left_ir': stem.name + '_left_ir.npy', 'right_ir': stem.name + '_right_ir.npy'}
    metadata = {'available': False, 'left_ir_intrinsics': asdict(frame.left_ir_intrinsics),
                'right_ir_intrinsics': asdict(frame.right_ir_intrinsics),
                'baseline_m': frame.stereo_baseline_m,
                'calibration': _json(frame.stereo_calibration), 'files': files,
                'synchronization': _json((frame.timestamp_metadata or {}).get('stereo_ir')),
                'capture_header': asdict(frame.header), 'format': 'native_y8',
                'resized': False, 'rectification_applied_by_this_helper': False}
    try:
        for side in ('left', 'right'):
            np.save(stem.parent / files[f'{side}_ir'], getattr(frame, f'{side}_ir'))
        metadata.update(available=True, reason='saved')
    except Exception as error:
        metadata.update(reason='stereo_capture_write_failed', error=f'{type(error).__name__}: {error}')
    return _json(metadata)


def _interval(lower, upper, total, preferred):
    needed = max(int(preferred), int(upper - lower))
    size = min(int(total), int(np.ceil(needed / 32.)) * 32)
    begin = int(np.floor((lower + upper - size) / 2.))
    begin = max(0, min(begin, total - size))
    return begin, begin + size


def target_stereo_crop(mask, *, maximum_disparity=192, preferred_shape=(384, 640)):
    """Choose original pixels around this target, including leftward matches.

    Both images receive the same crop. A left-image pixel at x matches a
    right-image pixel near x-disparity, so the left context includes the model's
    configured disparity range. No image scaling or fixed cell coordinates.
    """
    target = np.asarray(mask, dtype=bool)
    if target.ndim != 2 or not np.any(target):
        raise ValueError('native_target_mask_empty')
    rows, cols = np.nonzero(target)
    height, width = target.shape
    context = 32
    lo_x = max(0, int(cols.min()) - int(maximum_disparity) - context)
    hi_x = min(width, int(cols.max()) + 1 + context)
    lo_y = max(0, int(rows.min()) - context)
    hi_y = min(height, int(rows.max()) + 1 + context)
    x0, x1 = _interval(lo_x, hi_x, width, preferred_shape[1])
    y0, y1 = _interval(lo_y, hi_y, height, preferred_shape[0])
    return x0, y0, x1, y1


def _camera_geometry(frame):
    if frame.left_ir is None or frame.right_ir is None:
        raise ValueError('same_capture_infrared_unavailable')
    if (frame.left_ir_intrinsics is None or frame.right_ir_intrinsics is None
            or frame.stereo_baseline_m is None):
        raise ValueError('stereo_calibration_unavailable')
    if (frame.left_ir.shape != frame.depth_m.shape
            or frame.right_ir.shape != frame.depth_m.shape):
        raise ValueError('infrared_depth_grid_mismatch')
    if frame.left_ir.dtype != np.uint8 or frame.right_ir.dtype != np.uint8:
        raise ValueError('infrared_format_is_not_native_y8')
    for intrinsic in (frame.depth_intrinsics, frame.left_ir_intrinsics,
                      frame.right_ir_intrinsics):
        if np.any(np.abs(intrinsic.distortion) > 1e-10):
            raise ValueError('stereo_not_rectified')
        if not np.allclose(intrinsic.matrix, frame.depth_intrinsics.matrix, atol=1e-6, rtol=0):
            raise ValueError('infrared_depth_intrinsics_mismatch')
    calibration = frame.stereo_calibration or {}
    left_to_depth = np.asarray(calibration.get('left_ir_to_depth_cv'), dtype=float)
    if (left_to_depth.shape != (4, 4)
            or not np.allclose(left_to_depth, np.eye(4), atol=1e-6, rtol=0)):
        raise ValueError('infrared_depth_optical_frames_mismatch')
    left_to_right = np.asarray(calibration.get('left_ir_to_right_ir_cv'), dtype=float)
    if (left_to_right.shape != (4, 4) or not np.isfinite(left_to_right).all()
            or not np.allclose(left_to_right[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(left_to_right[1:3, 3], 0, atol=1e-6, rtol=0)):
        raise ValueError('stereo_rectification_geometry_unavailable')
    if left_to_right[0, 3] >= 0:
        raise ValueError('stereo_baseline_direction_mismatch')
    baseline = float(frame.stereo_baseline_m)
    if (not np.isfinite(baseline) or baseline <= 0
            or not np.isclose(abs(left_to_right[0, 3]), baseline, atol=1e-6, rtol=0)):
        raise ValueError('stereo_baseline_mismatch')
    synchronization = (frame.timestamp_metadata or {}).get('stereo_ir', {}).get('synchronized')
    if synchronization is not True:
        raise ValueError('infrared_capture_unsynchronized' if synchronization is False
                         else 'infrared_capture_synchronization_unknown')
    return frame.left_ir_intrinsics.matrix, baseline


def _plane_transforms(observation):
    transform = observation.get('belt_region', {}).get('plane_to_base')
    if transform is None:
        normal = np.asarray(observation['plane']['normal_base'], dtype=float)
        ex = np.array([1., 0., 0.])
        ex -= normal * np.dot(normal, ex)
        ex /= np.linalg.norm(ex)
        transform = np.eye(4)
        transform[:3, :3] = np.column_stack((ex, np.cross(normal, ex), normal))
        transform[:3, 3] = [0, 0, observation['plane']['abc'][2]]
    transform = np.asarray(transform, dtype=float)
    return transform, inverse(transform)


def _simple_geometry(points, flat_points, depths, frame, plane_to_base):
    """Use the existing simple detector's quantiles, axes and nominal TCP."""
    if len(points) < 70:
        raise ValueError('refined_target_below_original_70_pixel_minimum')
    q = .01
    heights = flat_points[:, 2]
    height = float(np.quantile(heights, .98))
    median_depth = float(np.median(depths))
    padding = median_depth / frame.depth_intrinsics.fx + median_depth / frame.depth_intrinsics.fy
    xy = flat_points[:, :2]
    values, vectors = np.linalg.eigh(np.cov(xy.T))
    projected = xy @ vectors
    lo, hi = np.quantile(projected, [q, 1-q], axis=0)
    raw_extent = np.sort(hi-lo)[::-1]
    extent = raw_extent + padding
    # These are the original component/simple-size conditions. Failure selects
    # the previous native target rather than rejecting the whole task.
    if (raw_extent[0] > .16 or raw_extent[1] < .012
            or not .035 <= height <= .11
            or not .015 <= min(extent) <= max(extent) <= .16):
        raise ValueError('refined_geometry_outside_original_simple_size_range')
    center_xy = (.5 * (lo + hi)) @ vectors.T
    confidence = float((values[-1]-values[0]) / max(values.sum(), 1e-12))
    yaw = float(np.arctan2(vectors[1, -1], vectors[0, -1]))
    local = points
    upper = heights >= .60 * height
    if int(upper.sum()) >= 20:
        local = points[upper]
        upper_xy = xy[upper]
        values, vectors = np.linalg.eigh(np.cov(upper_xy.T))
        projected = upper_xy @ vectors
        lo, hi = np.quantile(projected, [q, 1-q], axis=0)
        center_xy = (.5 * (lo + hi)) @ vectors.T
        confidence = float((values[-1]-values[0]) / max(values.sum(), 1e-12))
        yaw = float(np.arctan2(vectors[1, -1], vectors[0, -1]))
    normal = plane_to_base[:3, 2]
    center = plane_to_base[:3, :3] @ np.r_[center_xy, 0.] + plane_to_base[:3, 3]
    yaw = yaw if confidence >= .2 else 0.
    x = plane_to_base[:3, :3] @ np.array([np.cos(yaw), np.sin(yaw), 0.])
    z = -normal
    rotation = np.column_stack((x, np.cross(z, x), z))
    lower, upper = np.quantile(local @ rotation[:, 1], [q, 1-q])
    width = float(upper-lower+padding)
    if width + .020 > .10:
        raise ValueError('refined_width_exceeds_original_opening_with_side_clearance')
    grasp = np.eye(4)
    grasp[:3, :3] = rotation
    grasp[:3, 3] = center + normal * max(.02, height-.015)
    return {'grasp': grasp, 'center': center, 'height': height, 'width': width,
            'footprint_m': extent.tolist(), 'orientation_confidence': confidence,
            'localization_points': len(local), 'open_clearance_per_side_m': (.10-width)/2,
            'shape_consistency': {'status': 'not_required_in_simple_mode'},
            'cylinder_fit': {'accepted': False, 'reasons': ['not_requested']}}


@background_artifact
def _save_artifacts(directory, frame, native, result_points, result_mask, metadata):
    if directory is None:
        return
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        header = frame.header
        stem = directory / f'grasp_depth_e{header.clock_epoch}_s{header.sequence:06d}'
        ordinal = 1
        while stem.with_suffix('.json').exists():
            stem = directory / f'grasp_depth_e{header.clock_epoch}_s{header.sequence:06d}_{ordinal:02d}'
            ordinal += 1
        np.save(str(stem) + '_native_target.npy', np.asarray(native))
        np.save(str(stem) + '_target.npy', np.asarray(result_points))
        np.save(str(stem) + '_target_mask.npy', np.asarray(result_mask, dtype=bool))
        metadata['artifact_prefix'] = str(stem)
        stem.with_suffix('.json').write_text(json.dumps(_json(metadata), ensure_ascii=False,
                                                       indent=2, allow_nan=False) + '\n')
    except Exception as error:
        # Missing diagnostics never turns an otherwise valid native task into
        # a motion rejection. Interrupts still propagate.
        metadata['artifact_error'] = f'{type(error).__name__}: {error}'
    return {key: metadata[key] for key in ('artifact_prefix', 'artifact_error') if key in metadata}


def refine_grasp_observation(observation, backend, *, output_directory=None, emit=None,
                             robot_model=None):
    """Return a target reconstruction or an explicitly recorded native fallback.

    This function never advances the observation timestamp, installs a scene,
    writes to the camera backend, or starts/reuses a velocity measurement.
    """
    started = time.perf_counter_ns()
    cpu_started = time.thread_time_ns()
    timings = {}
    stage_names = (
        'input_setup', 'native_points_hash', 'request_preparation', 'backend_refine',
        'result_validation', 'hybrid_depth_copy', 'hybrid_depth_update', 'frame_replace',
        'deproject_depth_to_world', 'belt_coordinate_transform', 'visible_depth_mask',
        'mask_bookkeeping', 'filter_target_mask', 'robot_points_copy', 'robot_self_filter',
        'robot_mask_update', 'target_arrays_copy', 'simple_geometry', 'refined_points_hash',
        'result_bookkeeping', 'save_artifacts')
    for name in stage_names:
        # None means this branch was not called; a failed call is timed too.
        timings[name + '_s'] = None
        timings[name + '_thread_cpu_s'] = None

    stage_events = []
    timing_trace = trace_context()
    timing_trace['spans'] = stage_events

    def stage(name):
        return measured_stage(name, timings, events=stage_events)

    with stage('input_setup'):
        frame = observation['frame']
        native = np.asarray(observation.get('native_object_points', observation['object_points']))
        native_mask = np.asarray(observation['mask'], dtype=bool)
    with stage('native_points_hash'):
        native_sha = _points_sha(native)
        network_native_sha = _points_sha(native)
    metadata = {'applied': False, 'requested_source': 'fast_foundationstereo',
                'source': 'native_depth', 'capture_header': asdict(frame.header),
                'timestamp_metadata': frame.timestamp_metadata,
                'native_target_points': len(native), 'native_target_points_sha256': native_sha,
                'network_input_points_sha256': network_native_sha,
                'scene_source': 'native_device_depth', 'velocity_source': 'native_nero7_rgbd',
                'new_capture': False, 'hidden_surface_added': False,
                'timing_trace': timing_trace,
                'input_resized': False, 'native_depth_modified': False,
                'left_right_threshold_applied': False, 'timings_s': timings,
                'timing_inputs': {
                    'depth_shape': list(frame.depth_m.shape),
                    'rgb_shape': list(frame.rgb.shape) if frame.rgb is not None else None,
                    'native_mask_shape': list(native_mask.shape),
                    'native_target_shape': list(native.shape),
                    'depth_grid_points': int(frame.depth_m.size),
                    'native_target_points': len(native)},
                'timing_scope': {
                    'wall_clock': 'perf_counter_ns', 'cpu_clock': 'thread_time_ns',
                    'stage_times': 'exclusive; backend-reported timings may overlap backend_refine',
                    'total': 'entry through _save_artifacts return; excludes final timing JSON publication and emit',
                    'data_copy': 'sum of hybrid_depth_copy, frame_replace, robot_points_copy, target_arrays_copy; hashing and internal callee copies remain in their own stages'}}
    refined = dict(observation)
    # A caller can request another reconstruction for an already refined
    # capture. Its failure must restore the genuine native target, rather than
    # relabel the last learned cloud as native depth.
    native_geometry = observation.get('native_target_geometry')
    if native_geometry is not None:
        refined.update(native_geometry, object_points=native)
        refined.pop('grasp_target_mask', None)
        refined.pop('grasp_target_cloud_filter', None)
    result_points, result_mask = native, native_mask
    try:
        if backend is None:
            raise ValueError('backend_unavailable')
        with stage('request_preparation'):
            matrix, baseline = _camera_geometry(frame)
            if native_mask.shape != frame.depth_m.shape:
                raise ValueError('native_target_mask_grid_mismatch')
            crop = target_stereo_crop(native_mask)
            metadata['crop_xyxy'] = list(crop)
            capture = {'capture_header': asdict(frame.header),
                       'timestamp_metadata': frame.timestamp_metadata,
                       'serial_number': frame.serial_number, 'calibration_id': frame.calibration_id}
        with stage('backend_refine'):
            result = backend.refine(frame.left_ir, frame.right_ir, matrix, baseline,
                                    crop, capture_metadata=capture)
        metadata['timings_s'].update(result.timings_s)
        metadata['worker_provenance'] = result.provenance
        metadata['worker_artifact_prefix'] = result.artifact_prefix
        with stage('result_validation'):
            returned_capture = result.provenance.get('capture_metadata', {}).get('capture_header')
            if returned_capture != asdict(frame.header):
                raise ValueError('fast_result_capture_mismatch')
            if tuple(result.crop_xyxy) != crop:
                raise ValueError('fast_result_crop_mismatch')
            depth = np.asarray(result.reconstructed_depth_m, dtype=np.float32)
            if depth.shape != frame.depth_m.shape:
                raise ValueError('fast_result_grid_mismatch')
        with stage('hybrid_depth_copy'):
            hybrid_depth = frame.depth_m.copy()
        # Stereo reconstructs a larger crop. The old native-depth seed must
        # not permanently cut out the part's low base or depth-hole edges.
        with stage('hybrid_depth_update'):
            hybrid_depth[native_mask] = 0.
            silhouette = observation.get('visible_target_rgb_mask')
            valid = np.isfinite(depth) & (depth > 0.)
            if silhouette is None:
                valid &= native_mask
            hybrid_depth[valid] = depth[valid]
        with stage('frame_replace'):
            # Same calibrated exposure and immutable RGB/IR, with a private
            # learned-depth image. Do not recopy all three native images.
            hybrid_depth.setflags(write=False)
            hybrid = copy.copy(frame)
            object.__setattr__(hybrid, 'depth_m', hybrid_depth)
            object.__setattr__(hybrid, '_perception_products', None)
        plane_to_base, base_to_plane = _plane_transforms(observation)
        products = observation.get('_frame_products')
        local = (products is not None
                 and products.matches(frame, observation['depth_to_color_cv'])
                 and observation.get('points') is products.values.get('native_deprojection')
                 and observation.get('points_plane') is products.values.get(
                     ('belt_coordinates', plane_to_base.tobytes())))
        changed = native_mask | valid
        with stage('deproject_depth_to_world'):
            if local:
                rows, cols = np.nonzero(changed)
                z = hybrid_depth[changed].astype(np.float64)
                k = hybrid.depth_intrinsics
                camera = np.column_stack(((cols-k.cx)*z/k.fx, (rows-k.cy)*z/k.fy, z))
                updated = camera @ hybrid.camera_to_world_cv[:3, :3].T + hybrid.camera_to_world_cv[:3, 3]
                points = observation['points'].copy()
                points[changed] = updated
            else:
                points = deproject_depth_to_world(hybrid.depth_m, hybrid.depth_intrinsics.matrix,
                                                 hybrid.camera_to_world_cv, camera_convention='opencv')
        with stage('belt_coordinate_transform'):
            if local:
                flat = observation['points_plane'].copy()
                flat[changed] = updated @ base_to_plane[:3, :3].T + base_to_plane[:3, 3]
            else:
                flat = points @ base_to_plane[:3, :3].T + base_to_plane[:3, 3]
        metadata['timing_inputs'].update(
            reconstructed_depth_shape=list(depth.shape), world_points_shape=list(points.shape),
            belt_points_shape=list(flat.shape),
            hybrid_valid_depth_points=int(np.count_nonzero(np.isfinite(hybrid.depth_m) & (hybrid.depth_m > 0))),
            local_native_reuse=bool(local), updated_coordinate_points=int(changed.sum()) if local else int(hybrid_depth.size),
            rgb_projection_eligible_points=int(valid.sum()),
            visible_rgb_mask_shape=list(silhouette.shape) if silhouette is not None else None)
        if silhouette is None:
            with stage('mask_bookkeeping'):
                mask = valid & np.isfinite(flat).all(axis=2) & (flat[..., 2] >= .012) & (flat[..., 2] <= .12)
        else:
            with stage('visible_depth_mask'), nested_events(stage_events):
                from .visible_target import visible_depth_mask
                mask, lower_support = visible_depth_mask(
                    hybrid, observation['depth_to_color_cv'], silhouette, flat, observation['plane'],
                    timings_s=timings, diagnostics=metadata.setdefault('visible_projection', {}),
                    pixel_mask=valid)
            with stage('mask_bookkeeping'):
                mask &= valid
            metadata['visible_target_completion'] = {
                **(observation.get('visible_target_completion') or {}),
                'estimated_points_before_support_filter': int(mask.sum()),
                'estimated_points_outside_native_mask': int(np.count_nonzero(mask & ~native_mask)),
                'lower_support_height_m': lower_support,
                'selection': 'same_observed_rgb_outline_on_reconstructed_depth',
                'hidden_surface_added': False}
        metadata['timing_inputs']['filter_target_input_points'] = int(mask.sum())
        with stage('filter_target_mask'):
            mask, support = filter_target_mask(hybrid, observation['depth_to_color_cv'], mask,
                                               flat, base_to_plane @ frame.camera_to_world_cv)
        if robot_model is not None and mask.any():
            with stage('robot_points_copy'):
                robot_points = points[mask]
            metadata['timing_inputs']['robot_filter_input_shape'] = list(robot_points.shape)
            with stage('robot_self_filter'):
                self_check = robot_model.classify_robot_points(robot_points, observation['scene_state'],
                    camera_origin=frame.camera_to_world_cv[:3, 3])
            with stage('robot_mask_update'):
                mask[mask] = ~self_check.removed_mask
                support['estimated_target_self_filter'] = self_check.statistics
        with stage('target_arrays_copy'):
            target = points[mask]
            target_flat, target_depth = flat[mask], hybrid_depth[mask]
        metadata['timing_inputs'].update(target_shape=list(target.shape),
            target_flat_shape=list(target_flat.shape), target_depth_shape=list(target_depth.shape),
            simple_geometry_input_points=len(target))
        with stage('simple_geometry'):
            geometry = _simple_geometry(target, target_flat, target_depth, frame, plane_to_base)
        with stage('refined_points_hash'):
            refined_sha = _points_sha(target)
            network_refined_sha = _points_sha(target)
        with stage('result_bookkeeping'):
            support = dict(support, source='estimated_fast_stereo_rgb_and_metric_neighbor_support',
                           native_depth_modified=False, learned_target_depth=True)
            metadata.update(applied=True, source='estimated_fast_stereo_visible_depth', reason='applied',
                            refined_target_points=len(target), refined_target_pixels=int(mask.sum()),
                            refined_target_points_sha256=refined_sha,
                            network_input_points_sha256=network_refined_sha,
                            native_target_pixels=int(native_mask.sum()), target_cloud_filter=support,
                            geometry_seed_source='original_simple_geometry_rules_on_estimated_visible_depth')
            refined.update(geometry, object_points=target, grasp_target_mask=mask,
                           grasp_target_cloud_filter=support,
                           target_geometry_source='estimated_fast_stereo_visible_depth',
                           native_object_points=native,
                           native_target_geometry=(native_geometry if native_geometry is not None else
                               {key: observation[key] for key in
                                ('height', 'width', 'center', 'grasp', 'footprint_m', 'target_geometry_source',
                                 'shape_consistency', 'cylinder_fit', 'orientation_confidence',
                                 'localization_points', 'open_clearance_per_side_m') if key in observation}))
            result_points, result_mask = target, mask
    except Exception as error:
        metadata.update(applied=False, source='native_depth', reason=str(error),
                        error_type=type(error).__name__)
    # Preserve the original pre-save total for comparison with old logs.
    timings['target_refinement_total_s'] = (time.perf_counter_ns() - started) * 1e-9
    with stage('save_artifacts'):
        artifact_job = _save_artifacts(output_directory, frame, native, result_points, result_mask, metadata)
        if isinstance(artifact_job, dict) and 'job_id' in artifact_job:
            metadata['artifact_write_job_id'] = artifact_job['job_id']
    timings['target_refinement_thread_cpu_s'] = (time.thread_time_ns() - cpu_started) * 1e-9
    timings['target_refinement_wall_total_s'] = (time.perf_counter_ns() - started) * 1e-9
    for suffix in ('_s', '_thread_cpu_s'):
        timings['data_copy' + suffix] = sum(timings[name + suffix] or 0. for name in
            ('hybrid_depth_copy', 'frame_replace', 'robot_points_copy', 'target_arrays_copy'))
    timings['unattributed_wall_s'] = timings['target_refinement_wall_total_s'] - sum(
        timings[name + '_s'] or 0. for name in stage_names)
    timings['unattributed_thread_cpu_s'] = timings['target_refinement_thread_cpu_s'] - sum(
        timings[name + '_thread_cpu_s'] or 0. for name in stage_names)
    # The measured save has already returned. Publish its timing in the same
    # diagnostic JSON without repeating any array writes. Publication cannot
    # include its own duration; it is outside the explicitly stated total.
    if 'artifact_prefix' in metadata and 'artifact_error' not in metadata:
        try:
            Path(metadata['artifact_prefix']).with_suffix('.json').write_text(
                json.dumps(_json(metadata), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        except Exception as error:
            metadata['artifact_error'] = f'{type(error).__name__}: {error}'
    refined['grasp_depth_refinement'] = _json(metadata)
    if emit is not None:
        emit('grasp_depth_refinement', refined['grasp_depth_refinement'])
    return refined
