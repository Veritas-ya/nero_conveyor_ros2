"""Target-only learned depth keeps native timing, obstacle and velocity inputs."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import json

import numpy as np
import pytest

from mujoco_conveyor_sorting.perception import deproject_depth_to_world
from mujoco_conveyor_sorting.runtime.grasp_depth import (
    refine_grasp_observation, save_stereo_capture, target_stereo_crop)
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics, RGBDFrame, SampleHeader


def native_observation(*, sequence=4):
    intrinsic = CameraIntrinsics(width=1280, height=800, fx=600., fy=600., cx=640., cy=400.,
                                 distortion=(0., 0., 0., 0., 0.))
    depth = np.ones((800, 1280), dtype=np.float32)
    mask = np.zeros(depth.shape, dtype=bool)
    mask[390:430, 700:730] = True
    depth[mask] = .95
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = 1.
    l2r = np.eye(4)
    l2r[0, 3] = -.095
    ir = np.full(depth.shape, 100, dtype=np.uint8)
    frame = RGBDFrame(
        header=SampleHeader(sample_time=100., received_time=100.02, sequence=sequence,
                            source_time=30. + sequence / 30., clock_epoch=2),
        depth_m=depth, depth_intrinsics=intrinsic,
        rgb=np.full((800, 1280, 3), 128, dtype=np.uint8), color_intrinsics=intrinsic,
        camera_to_world_cv=transform, left_ir=ir, right_ir=ir,
        left_ir_intrinsics=intrinsic, right_ir_intrinsics=intrinsic, stereo_baseline_m=.095,
        stereo_calibration={'left_ir_to_depth_cv': np.eye(4).tolist(),
                            'left_ir_to_right_ir_cv': l2r.tolist()},
        timestamp_metadata={'stereo_ir': {'synchronized': True}, 'exposure_us': 13000})
    points = deproject_depth_to_world(depth, intrinsic.matrix, transform,
                                     camera_convention='opencv')
    grasp = np.diag([1., -1., -1., 1.])
    grasp[:3, 3] = [.12, 0., .035]
    return {'frame': frame, 'mask': mask, 'points': points, 'object_points': points[mask],
            'scene_points': points[~mask], 'depth_to_color_cv': np.eye(4),
            'belt_region': {'plane_to_base': np.eye(4)},
            'plane': {'normal_base': [0., 0., 1.], 'abc': [0., 0., 0.]},
            'height': .05, 'width': .046, 'center': np.array([.12, 0., 0.]),
            'grasp': grasp, 'footprint_m': [.065, .049],
            'target_geometry_source': 'upper_depth_geometry',
            'target_cloud_filter': {'source': 'native'} }


class Reconstruction:
    def __init__(self, observation, *, height=.046):
        self.observation = observation
        self.height = height
        self.calls = []
        self.error = None

    def refine(self, left, right, matrix, baseline, crop, *, capture_metadata=None):
        self.calls.append({'left': left, 'right': right, 'matrix': matrix.copy(),
                           'baseline': baseline, 'crop': crop, 'capture': capture_metadata})
        if self.error is not None:
            raise self.error
        frame = self.observation['frame']
        output = np.full(frame.depth_m.shape, np.nan, dtype=np.float32)
        x0, y0, x1, y1 = crop
        output[y0:y1, x0:x1] = 1.
        output[self.observation['mask']] = 1. - self.height
        matrix_crop = matrix.copy()
        matrix_crop[:2, 2] -= [x0, y0]
        return SimpleNamespace(reconstructed_depth_m=output, crop_xyxy=crop,
                               K_crop=matrix_crop, timings_s={'parent_request_to_result_s': .12},
                               provenance={'capture_metadata': capture_metadata},
                               artifact_prefix='fake/call_0001')


def test_refines_same_target_without_modifying_native_products_or_capture_time(tmp_path):
    original = native_observation()
    before = original['frame'].depth_m.copy()
    original_target = original['object_points'].copy()
    backend = Reconstruction(original)
    events = []
    result = refine_grasp_observation(original, backend, output_directory=tmp_path,
                                      emit=lambda name, data: events.append((name, data)))
    assert result is not original and result['grasp_depth_refinement']['applied']
    for key in ('frame', 'points', 'scene_points', 'mask', 'plane', 'belt_region', 'target_cloud_filter'):
        assert result[key] is original[key]
    np.testing.assert_array_equal(original['frame'].depth_m, before)
    np.testing.assert_array_equal(original['object_points'], original_target)
    assert result['frame'].header is original['frame'].header
    assert result['height'] == pytest.approx(.046, abs=1e-6)
    assert result['grasp'][2, 3] == pytest.approx(.031, abs=1e-6)
    np.testing.assert_allclose(result['object_points'][:, 2], .046, atol=1e-6)
    assert result['target_geometry_source'] == 'estimated_fast_stereo_visible_depth'
    assert result['grasp_depth_refinement']['scene_source'] == 'native_device_depth'
    assert result['grasp_depth_refinement']['velocity_source'] == 'native_nero7_rgbd'
    assert result['grasp_depth_refinement']['hidden_surface_added'] is False
    assert np.all(~result['grasp_target_mask'] | original['mask'])
    assert events[0][0] == 'grasp_depth_refinement'
    metadata_path, = tmp_path.glob('*.json')
    metadata = json.loads(metadata_path.read_text())
    assert metadata['capture_header'] == asdict(original['frame'].header)
    np.testing.assert_array_equal(np.load(str(metadata_path.with_suffix('')) + '_target.npy'),
                                  result['object_points'])


def test_crop_moves_with_native_pixels_preserves_scale_and_leaves_matching_room():
    first = np.zeros((800, 1280), dtype=bool)
    first[400:440, 700:740] = True
    second = np.zeros_like(first)
    second[500:540, 850:890] = True
    a, b = target_stereo_crop(first), target_stereo_crop(second)
    assert b[0]-a[0] == 150 and b[1]-a[1] == 100
    assert (a[2]-a[0], a[3]-a[1]) == (640, 384)
    assert a[0] <= 700-192-32 and a[2] >= 740+32
    assert a[1] <= 400-32 and a[3] >= 440+32


def test_large_and_boundary_crops_contain_original_target_without_resize():
    target = np.zeros((800, 1280), dtype=bool)
    target[0:600, 5:1000] = True
    x0, y0, x1, y1 = target_stereo_crop(target)
    assert 0 <= x0 <= 5 and 1000 <= x1 <= 1280
    assert y0 == 0 and 600 <= y1 <= 800
    assert x1-x0 > 640 and y1-y0 > 384


def test_worker_receives_full_original_intrinsics_and_declared_same_capture():
    observation = native_observation()
    backend = Reconstruction(observation)
    result = refine_grasp_observation(observation, backend)
    assert result['grasp_depth_refinement']['applied']
    request, = backend.calls
    assert request['left'] is observation['frame'].left_ir
    np.testing.assert_array_equal(request['matrix'], observation['frame'].depth_intrinsics.matrix)
    assert request['baseline'] == .095
    assert request['capture']['capture_header'] == asdict(observation['frame'].header)


@pytest.mark.parametrize(('change', 'reason'), [
    ({'left_ir': None}, 'same_capture_infrared_unavailable'),
    ({'timestamp_metadata': {'stereo_ir': {'synchronized': False}}}, 'infrared_capture_unsynchronized'),
    ({'timestamp_metadata': {}}, 'infrared_capture_synchronization_unknown'),
    ({'stereo_calibration': {}}, 'infrared_depth_optical_frames_mismatch'),
    ({'stereo_baseline_m': .12}, 'stereo_baseline_mismatch'),
])
def test_geometry_or_sync_mismatch_retains_native_target_without_calling_worker(change, reason):
    observation = native_observation()
    observation['frame'] = replace(observation['frame'], **change)
    backend = Reconstruction(observation)
    result = refine_grasp_observation(observation, backend)
    assert result['grasp_depth_refinement']['applied'] is False
    assert result['grasp_depth_refinement']['reason'] == reason
    assert result['object_points'] is observation['object_points']
    assert result['height'] == observation['height']
    assert backend.calls == []


def test_intrinsics_mismatch_uses_native_geometry():
    observation = native_observation()
    frame = observation['frame']
    observation['frame'] = replace(frame, left_ir_intrinsics=replace(frame.left_ir_intrinsics, fx=601.))
    backend = Reconstruction(observation)
    result = refine_grasp_observation(observation, backend)
    assert not result['grasp_depth_refinement']['applied']
    assert result['grasp_depth_refinement']['reason'] == 'infrared_depth_intrinsics_mismatch'
    assert not backend.calls


def test_swapped_stereo_baseline_uses_native_geometry():
    observation = native_observation()
    calibration = dict(observation['frame'].stereo_calibration)
    baseline = np.array(calibration['left_ir_to_right_ir_cv'])
    baseline[0, 3] = .095
    calibration['left_ir_to_right_ir_cv'] = baseline.tolist()
    observation['frame'] = replace(observation['frame'], stereo_calibration=calibration)
    backend = Reconstruction(observation)
    result = refine_grasp_observation(observation, backend)
    assert not result['grasp_depth_refinement']['applied']
    assert result['grasp_depth_refinement']['reason'] == 'stereo_baseline_direction_mismatch'
    assert not backend.calls


def test_model_failure_and_original_size_failure_are_explicit_native_fallbacks():
    observation = native_observation()
    backend = Reconstruction(observation)
    backend.error = RuntimeError('model unavailable')
    result = refine_grasp_observation(observation, backend)
    assert result['grasp_depth_refinement']['reason'] == 'model unavailable'
    assert result['object_points'] is observation['object_points']
    small = refine_grasp_observation(observation, Reconstruction(observation, height=.02))
    assert not small['grasp_depth_refinement']['applied']
    assert small['height'] == observation['height']


def test_boundary_reconstructed_at_belt_is_removed_only_from_learned_target():
    observation = native_observation()
    class Boundary(Reconstruction):
        def refine(self, *args, **kwargs):
            result = super().refine(*args, **kwargs)
            result.reconstructed_depth_m[390:394, 700:730] = 1.
            return result
    result = refine_grasp_observation(observation, Boundary(observation))
    assert result['grasp_depth_refinement']['applied']
    assert not np.any(result['grasp_target_mask'][390:394])
    assert np.all(observation['mask'][390:394, 700:730])
    assert result['scene_points'] is observation['scene_points']


def test_each_capture_calls_worker_and_rejects_result_from_earlier_capture():
    first = native_observation(sequence=4)
    second = native_observation(sequence=5)
    backend = Reconstruction(first)
    one = refine_grasp_observation(first, backend)
    two = refine_grasp_observation(second, backend)
    assert one['grasp_depth_refinement']['applied'] and two['grasp_depth_refinement']['applied']
    assert len(backend.calls) == 2
    assert backend.calls[0]['capture']['capture_header'] != backend.calls[1]['capture']['capture_header']
    class OldCapture(Reconstruction):
        def refine(self, *args, **kwargs):
            result = super().refine(*args, **kwargs)
            result.provenance['capture_metadata']['capture_header'] = asdict(first['frame'].header)
            return result
    rejected = refine_grasp_observation(second, OldCapture(second))
    assert not rejected['grasp_depth_refinement']['applied']
    assert rejected['grasp_depth_refinement']['reason'] == 'fast_result_capture_mismatch'


def test_interrupt_propagates_instead_of_falling_back():
    observation = native_observation()
    backend = Reconstruction(observation)
    backend.error = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        refine_grasp_observation(observation, backend)


def test_stereo_capture_saves_authentic_ir_and_json_serializable_geometry(tmp_path):
    observation = native_observation()
    frame = observation['frame']
    metadata = save_stereo_capture(tmp_path / 'frame_000004', frame)
    assert metadata['available']
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata['baseline_m'] == .095
    assert metadata['synchronization']['synchronized'] is True
    assert metadata['capture_header'] == asdict(frame.header)
    for side in ('left', 'right'):
        np.testing.assert_array_equal(np.load(tmp_path / metadata['files'][f'{side}_ir']),
                                      getattr(frame, f'{side}_ir'))
    missing = save_stereo_capture(tmp_path / 'no_ir', replace(frame, left_ir=None))
    assert missing == {'available': False, 'reason': 'same_capture_infrared_unavailable'}
    assert not (tmp_path / 'no_ir_left_ir.npy').exists()


def test_failure_after_success_restores_original_native_geometry_instead_of_old_model_output():
    observation = native_observation()
    observation.update(shape_consistency={'source': 'native'},
                       cylinder_fit={'accepted': False, 'source': 'native'},
                       orientation_confidence=.125, localization_points=12,
                       open_clearance_per_side_m=.027)
    refined = refine_grasp_observation(observation, Reconstruction(observation))
    assert refined['grasp_depth_refinement']['applied']
    restored = refine_grasp_observation(refined, None)
    assert not restored['grasp_depth_refinement']['applied']
    assert restored['grasp_depth_refinement']['reason'] == 'backend_unavailable'
    assert restored['object_points'] is observation['object_points']
    for key in ('height', 'width', 'center', 'grasp', 'footprint_m', 'target_geometry_source',
                'shape_consistency', 'cylinder_fit', 'orientation_confidence',
                'localization_points', 'open_clearance_per_side_m'):
        if isinstance(observation[key], np.ndarray):
            np.testing.assert_array_equal(restored[key], observation[key])
        else:
            assert restored[key] == observation[key]
    assert 'grasp_target_mask' not in restored and 'grasp_target_cloud_filter' not in restored
    # Repeated success must continue retaining the original native geometry.
    twice = refine_grasp_observation(refined, Reconstruction(observation, height=.044))
    again = refine_grasp_observation(twice, None)
    assert again['height'] == observation['height']
    assert again['grasp_depth_refinement']['network_input_points_sha256'] == restored['grasp_depth_refinement']['network_input_points_sha256']
