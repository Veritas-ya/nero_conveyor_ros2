"""Reject native near-depth ghosts supported by measured background and robot occlusion.

Only already isolated, same-depth robot-edge candidates are examined. The left
image must still match measured background variation, and the nominal robot
must occlude that background in the right camera while leaving it visible in
the left. Historical depth is never inserted into the current point cloud.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_dilation

from .grasp_depth import _camera_geometry, _json
from .scene_depth_filter import _native_pixels, DepthEdgeTailConfig
from .scene_stereo_filter import _patches, _census_cost, SceneStereoFilterResult, _save
from .self_filter import _ray_entry
from .types import CameraIntrinsics, RGBDFrame, SampleHeader

DEFAULT_REFERENCE = Path(__file__).resolve().parents[2]/'configs/scene_background_reference.json'


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float64).tobytes()).hexdigest()


def state_digest(state):
    return _sha(np.r_[state['joints'], state['width_m']])


def geometry_digest(bodies):
    digest = hashlib.sha256()
    for body in bodies:
        for array in (body.equations, body.position, body.rotation):
            digest.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BackgroundCapture:
    frame: RGBDFrame
    target_mask: np.ndarray
    bodies: tuple
    source: str
    protected_mask: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        protected = binary_dilation(self.target_mask, structure=np.ones((5, 5), bool))
        protected.setflags(write=False)
        object.__setattr__(self, 'protected_mask', protected)


@dataclass(frozen=True)
class SceneBackgroundReference:
    captures: tuple[BackgroundCapture, ...]
    manifest_sha256: str
    source: str


def load_background_reference(path, bodies_for_state):
    """Read an explicit hash-bound bank once; no device connections or model changes."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if not isinstance(manifest, dict):
        raise ValueError('background reference must be a manifest object')
    entries = manifest.get('captures', [])
    if manifest.get('schema_version') != 1 or not isinstance(entries, list) or len(entries) < 3:
        raise ValueError('background reference requires at least three saved captures')
    captures, identities = [], set()
    for entry in entries:
        files = {}
        for key in ('metadata', 'depth', 'left_ir', 'right_ir', 'target_mask'):
            item = entry['files'][key]
            file = (path.parent/item['path']).resolve()
            if hashlib.sha256(file.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'background reference hash mismatch: {key}')
            files[key] = file
        metadata = json.loads(files['metadata'].read_text())
        stereo = metadata['stereo_capture']
        frame = RGBDFrame(
            header=SampleHeader(**metadata['header']),
            depth_m=np.load(files['depth'], allow_pickle=False),
            depth_intrinsics=CameraIntrinsics(**metadata['depth_intrinsics']),
            camera_to_world_cv=np.asarray(metadata['depth_to_base_cv'], dtype=float),
            serial_number=metadata['serial_number'], calibration_id=metadata['calibration_id'],
            left_ir=np.load(files['left_ir'], allow_pickle=False),
            right_ir=np.load(files['right_ir'], allow_pickle=False),
            left_ir_intrinsics=CameraIntrinsics(**stereo['left_ir_intrinsics']),
            right_ir_intrinsics=CameraIntrinsics(**stereo['right_ir_intrinsics']),
            stereo_baseline_m=stereo['baseline_m'], stereo_calibration=stereo['calibration'],
            timestamp_metadata=metadata['timestamp_metadata'], source='saved_background_reference')
        _camera_geometry(frame)
        identity = (frame.serial_number, frame.header.sample_time, frame.header.source_time,
                    frame.header.sequence, frame.header.clock_epoch)
        if identity in identities:
            raise ValueError('background reference repeats the same capture')
        identities.add(identity)
        target = np.load(files['target_mask'], allow_pickle=False)
        if target.shape != frame.depth_m.shape or target.dtype != bool:
            raise ValueError('background target mask must match original depth')
        for array in (target, frame.depth_m, frame.left_ir, frame.right_ir, frame.camera_to_world_cv):
            array.setflags(write=False)
        captures.append(BackgroundCapture(frame, target,
            tuple(bodies_for_state(metadata['robot_state'])), str(files['metadata'])))
    return SceneBackgroundReference(tuple(captures), hashlib.sha256(raw).hexdigest(), str(path))


def matching_geometry(frame, reference):
    """Background pixels refer to the same calibrated, fixed camera grid."""
    return (bool(frame.serial_number) and bool(frame.calibration_id)
        and frame.serial_number == reference.serial_number
        and frame.calibration_id == reference.calibration_id
        and frame.depth_m.shape == reference.depth_m.shape
        and np.allclose(frame.camera_to_world_cv, reference.camera_to_world_cv, atol=1e-6, rtol=0)
        and np.allclose(frame.depth_intrinsics.matrix, reference.depth_intrinsics.matrix,
                        atol=1e-6, rtol=0)
        and np.allclose(frame.stereo_calibration['left_ir_to_right_ir_cv'],
                        reference.stereo_calibration['left_ir_to_right_ir_cv'], atol=1e-6, rtol=0))


def camera_origins(frame):
    transform = np.asarray(frame.camera_to_world_cv, dtype=float)
    right = transform @ np.linalg.inv(np.asarray(frame.stereo_calibration['left_ir_to_right_ir_cv']))
    return transform[:3, 3], right[:3, 3]


def pixel_points(frame, uv, depth):
    k = frame.depth_intrinsics
    camera = np.column_stack(((uv[:, 0]-k.cx)*depth/k.fx,
                              (uv[:, 1]-k.cy)*depth/k.fy, depth))
    transform = np.asarray(frame.camera_to_world_cv)
    return camera @ transform[:3, :3].T + transform[:3, 3]


def ray_visibility(points, origin, bodies, margin_m):
    """Ray-test nominal convex bodies; self-filter padding does not alter visibility."""
    delta = points-np.asarray(origin)
    ranges = np.linalg.norm(delta, axis=1)
    directions = np.divide(delta, ranges[:, None], out=np.zeros_like(delta), where=ranges[:, None] > 0)
    first = np.full(len(points), np.inf)
    for body in bodies:
        first = np.minimum(first, _ray_entry(directions, origin, body))
    return (first > ranges+margin_m), (first < ranges-margin_m), first, ranges


def background_evidence(frame, uv, native_depth, reference, bodies, *,
                        depth_config=None, ray_margin_m=.003):
    """Per-pixel joint evidence, using three or more actual background observations."""
    config = DepthEdgeTailConfig() if depth_config is None else depth_config
    uv = np.asarray(uv, dtype=int)
    native_depth = np.asarray(native_depth, dtype=float)
    current_patch, current_valid = _patches(frame.left_ir, uv, np.zeros((len(uv), 1)), 2)
    valid_refs, depths, patches = [], [], []
    left_origin, right_origin = camera_origins(frame)
    for capture in reference.captures:
        ref = capture.frame
        if not matching_geometry(frame, ref):
            continue
        z = ref.depth_m[uv[:, 1], uv[:, 0]].astype(float)
        points = pixel_points(ref, uv, np.where(np.isfinite(z), z, 0.))
        left, right = camera_origins(ref)
        visible_left = ray_visibility(points, left, capture.bodies, ray_margin_m)[0]
        visible_right = ray_visibility(points, right, capture.bodies, ray_margin_m)[0]
        patch, patch_valid = _patches(ref.left_ir, uv, np.zeros((len(uv), 1)), 2)
        usable = (np.isfinite(z) & (z > 0) & visible_left & visible_right
                  & patch_valid.all(axis=(1, 2)) & ~capture.protected_mask[uv[:, 1], uv[:, 0]])
        depths.append(z)
        patches.append(patch)
        valid_refs.append(usable)
    if len(depths) < 3:
        return {'removed': np.zeros(len(uv), bool), 'reason': 'fewer_than_three_matching_references'}
    valid = np.stack(valid_refs)
    depth = np.stack(depths)
    count = valid.sum(axis=0)
    # Invalid reference samples do not participate in either depth or appearance statistics.
    low = np.min(np.where(valid, depth, np.inf), axis=0)
    high = np.max(np.where(valid, depth, -np.inf), axis=0)
    center = np.divide(np.where(valid, depth, 0.).sum(axis=0), count,
                       out=np.zeros(len(uv)), where=count > 0)
    stable = (count >= 3) & ((high-low) <= config.robot_depth_tolerance_m)
    points = pixel_points(frame, uv, center)
    left_clear, _, _, _ = ray_visibility(points, left_origin, bodies, ray_margin_m)
    _, right_blocked, right_entry, right_range = ray_visibility(points, right_origin, bodies, ray_margin_m)
    costs = np.stack([_census_cost(current_patch, patch).ravel() for patch in patches])
    current_cost = np.min(np.where(valid, costs, np.inf), axis=0)
    observed_variation = np.zeros(len(uv), dtype=float)
    for i in range(len(patches)):
        for j in range(i):
            cost = _census_cost(patches[i], patches[j]).ravel()
            observed_variation = np.maximum(observed_variation,
                np.where(valid[i] & valid[j], cost, 0))
    # Threshold is the observed inter-reference variation at this pixel,
    # rather than a global intensity cutoff fitted to the pictured clusters.
    appearance = current_valid.all(axis=(1, 2)) & (current_cost <= observed_variation)
    near_jump = (low-native_depth) >= config.minimum_background_jump_m
    removed = stable & left_clear & right_blocked & appearance & near_jump
    return {'removed': removed, 'reason': 'evaluated', 'reference_count': count,
        'background_low_m': low, 'background_high_m': high, 'background_mean_m': center,
        'reference_depth_stable': stable, 'left_background_visible': left_clear,
        'right_background_occluded': right_blocked, 'right_robot_entry_range_m': right_entry,
        'right_background_range_m': right_range, 'near_depth_jump': near_jump,
        'current_background_census_cost_bits': current_cost,
        'observed_background_census_variation_bits': observed_variation,
        'left_background_appearance_matches': appearance}


def filter_scene_background(points, *, frame=None, protected_mask=None, reference=None,
                            bodies=(), depth_edge_result=None, depth_config=None,
                            state=None, ray_margin_m=.003, enabled=True, output_directory=None):
    started = time.perf_counter()
    points = np.asarray(points)
    result = SceneStereoFilterResult(points, np.zeros(len(points), bool), {
        'algorithm': 'measured_background_left_appearance_and_right_robot_occlusion',
        'input_points': len(points), 'removed_points': 0, 'retained_points': len(points),
        'native_depth_modified': False, 'timestamp_modified': False,
        'added_surface_points': 0, 'background_points_inserted': 0,
        'classification': 'invalid_native_measurement_only', 'free_space_inferred': False,
        'visibility_geometry': 'nominal_robot_convex_bodies_without_padding',
        'occlusion_geometry_sha256': geometry_digest(bodies),
        'reference_source': None if reference is None else reference.source,
        'reference_sha256': None if reference is None else reference.manifest_sha256,
        'robot_state_sha256': None if state is None else state_digest(state),
        'parameters': {'minimum_reference_captures': 3, 'census_radius': 2,
                       'ray_margin_m': ray_margin_m}}, [])
    stats = result.statistics
    config = DepthEdgeTailConfig() if depth_config is None else depth_config
    stats['parameters'].update(depth_gap_m=config.minimum_background_jump_m,
                              maximum_reference_depth_spread_m=config.robot_depth_tolerance_m)
    def finish(reason):
        stats.update(reason=reason, removed_points=int(result.removed_mask.sum()),
                     retained_points=int((~result.removed_mask).sum()),
                     processing_s=time.perf_counter()-started)
        _save(result, output_directory, frame, stem_name='scene_background')
        return result
    if not enabled or not config.enabled:
        return finish('disabled')
    if reference is None:
        return finish('background_reference_unavailable')
    if frame is None or depth_edge_result is None or not bodies:
        return finish('missing_native_frame_robot_or_edge_evidence')
    try:
        _camera_geometry(frame)
    except (ValueError, AttributeError, TypeError) as error:
        return finish(str(error))
    protected = np.asarray(protected_mask)
    if protected.shape != frame.depth_m.shape or protected.dtype != bool:
        return finish('invalid_protected_target_mask')
    if not np.array_equal(points, depth_edge_result.retained_points):
        return finish('depth_edge_input_mismatch')
    stats.update(capture_header=asdict(frame.header), serial_number=frame.serial_number,
        calibration_id=frame.calibration_id, camera_to_base_sha256=_sha(frame.camera_to_world_cv),
        protected_target_mask_sha256=hashlib.sha256(protected.tobytes()).hexdigest())
    protected = binary_dilation(protected, structure=np.ones((5, 5), bool))
    index = np.full(len(depth_edge_result.points), -1, dtype=int)
    index[~depth_edge_result.removed_mask] = np.arange(len(points))
    for component in depth_edge_result.components:
        if (component['removed'] or component['reason'] != 'insufficient_deep_background_boundary'
                or component.get('same_depth_robot_boundary_pixels', 0) == 0
                or component.get('unsupported_boundary_fraction', 0.) < config.minimum_unsupported_boundary_fraction):
            continue
        original = np.asarray(component.get('original_point_indices', []), dtype=int)
        if not len(original):
            continue
        selected = index[original]
        if (selected < 0).any():
            continue
        detail = {'point_count': len(selected), 'original_point_indices': selected.tolist(),
                  'removed_points': 0, 'removed': False, 'bbox_base_m': component['bbox_base_m']}
        result.components.append(detail)
        uv, reliable = _native_pixels(points[selected], frame.depth_m, frame.depth_intrinsics,
                                       frame.camera_to_world_cv)
        if not reliable.all() or protected[uv[:, 1], uv[:, 0]].any():
            detail['reason'] = 'protected_target_or_unreliable_native_pixels'
            continue
        try:
            evidence = background_evidence(frame, uv, frame.depth_m[uv[:, 1], uv[:, 0]],
                reference, bodies, depth_config=config, ray_margin_m=ray_margin_m)
            remove = evidence['removed']
            result.removed_mask[selected[remove]] = True
            detail.update(reason=evidence['reason'], native_uv=uv.tolist(),
                          removed=bool(remove.any()), removed_points=int(remove.sum()),
                          per_point_evidence=_json(evidence))
        except (ValueError, AttributeError, TypeError, np.linalg.LinAlgError) as error:
            detail['reason'] = f'background_evidence_unavailable: {error}'
    return finish('evaluated')
