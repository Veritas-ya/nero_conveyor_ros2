"""Measured belt geometry and its use by the retained Nero7 detector."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.perception.rgbd_vision_pipeline import deproject_depth_to_world
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics, SampleHeader


def make_scene(*, object_width_pixels=18):
    depth = np.ones((80, 100), dtype=float)
    depth[33:43, 45:45 + object_width_pixels] = .95
    k = CameraIntrinsics(width=100, height=80, fx=100., fy=100., cx=50., cy=40.)
    rotation = Rotation.from_euler('zy', [31., 23.], degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.diag([1., -1., -1.])
    origin = np.array([-.3, .1, -.12])
    transform[:3, 3] = origin + rotation @ [0., 0., 1.]
    frame = SimpleNamespace(
        depth_m=depth, depth_intrinsics=k, color_intrinsics=k,
        camera_to_world_cv=transform, rgb=np.zeros((80, 100, 3), np.uint8),
        header=SampleHeader(9000., 9000.1, 1, 10., 0))
    points = deproject_depth_to_world(depth, k.matrix, transform, camera_convention='opencv')
    green_mask = np.zeros(depth.shape, bool)
    green_mask[10:70, 15:90] = True
    green_mask &= depth == 1.
    normal = rotation[:, 2]
    abc = np.array([-normal[0] / normal[2], -normal[1] / normal[2],
                    np.dot(normal, origin) / normal[2]])

    def green(_frame, _d2c):
        return points, green_mask, None

    def plane(_points):
        return abc, {'normal_base': normal.tolist(), 'abc': abc.tolist()}

    return frame, green, plane, normal, origin


@pytest.mark.parametrize('object_width_pixels', [0, 18])
def test_tilted_belt_geometry_is_available_with_or_without_a_graspable_part(object_width_pixels):
    frame, green, plane, normal, origin = make_scene(object_width_pixels=object_width_pixels)
    surface = BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=green)
    np.testing.assert_allclose(surface.normal_base, normal, atol=1e-12)
    rotation = surface.plane_to_base[:3, :3]
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    assert np.dot(surface.plane_to_base[:3, 3] - origin, normal) == pytest.approx(0., abs=1e-12)
    assert np.all(surface.bounds_xy_m[:, 0] < surface.bounds_xy_m[:, 1])

    # The extracted hull must reach the original green image region after
    # base/plane transforms; the part's 50 mm height cannot move that region.
    hull = np.asarray(surface.hull_xy).reshape(-1, 2)
    base = np.column_stack((hull, np.zeros(len(hull)))) @ rotation.T + surface.plane_to_base[:3, 3]
    optical = (base - frame.camera_to_world_cv[:3, 3]) @ frame.camera_to_world_cv[:3, :3]
    uv = optical[:, :2] / optical[:, 2:] * [100., 100.] + [50., 40.]
    np.testing.assert_allclose(uv.min(axis=0), [15., 10.], atol=1e-5)
    np.testing.assert_allclose(uv.max(axis=0), [89., 69.], atol=1e-5)

    # Exercise the real downstream adapter rather than rebuilding its ROI.
    result = Nero7BeltVelocityMeter(surface).update(frame, np.eye(4))
    np.testing.assert_allclose(result['roi_px'], [15, 10, 90, 70], atol=1)
    assert result['speed_m_s'] is None


def test_missing_belt_remains_a_recoverable_fit_error():
    frame, _green, plane, _, _ = make_scene()

    def missing(*_args):
        raise ValueError('no visible belt')

    with pytest.raises(ValueError, match='no visible belt'):
        BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=missing)


def test_off_plane_green_depth_does_not_extend_downstream_roi_onto_table():
    frame, green, plane, _, _ = make_scene()
    reference = BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=green)
    points, mask, _ = green(frame, np.eye(4))
    outlier = reference.plane_to_base[:3, :3] @ [2., 2., .15] + reference.plane_to_base[:3, 3]
    contaminated = np.concatenate((points, np.broadcast_to(outlier, (4, 100, 3))), axis=0)
    contaminated_mask = np.concatenate((mask, np.ones((4, 100), dtype=bool)), axis=0)

    def bad_green(_frame, _d2c):
        return contaminated, contaminated_mask, None

    surface = BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=bad_green)
    np.testing.assert_allclose(surface.bounds_xy_m, reference.bounds_xy_m, atol=1e-12)
    np.testing.assert_array_equal(surface.hull_xy, reference.hull_xy)
    clean = Nero7BeltVelocityMeter(reference).update(frame, np.eye(4))
    noisy = Nero7BeltVelocityMeter(surface).update(frame, np.eye(4))
    assert noisy['roi_px'] == clean['roi_px']


def test_surface_four_position_arguments_remain_compatible_without_reuse_identity():
    surface = BeltSurface(np.eye(4), np.zeros((2, 2)), np.zeros((4, 1, 2)), {})
    frame, _, _, _, _ = make_scene()
    assert surface.point_count == 0
    assert not surface.matches_frame(frame, np.eye(4))


def test_surface_identity_tracks_camera_geometry_and_clock_epoch():
    from fixture_single_pick import synthetic_frame
    frame = synthetic_frame()
    _, green, plane, _, _ = make_scene()
    surface = BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=green)
    assert surface.point_count > 0
    assert surface.matches_frame(frame, np.eye(4))
    assert surface.matches_frame(replace(frame, rgb=np.zeros_like(frame.rgb),
        header=replace(frame.header, sequence=8, sample_time=2.)), np.eye(4))
    changed_transform = frame.camera_to_world_cv.copy()
    changed_transform[0, 3] += .001
    variants = [replace(frame, serial_number='another-camera'),
                replace(frame, calibration_id='another-calibration'),
                replace(frame, header=replace(frame.header, clock_epoch=1)),
                replace(frame, camera_to_world_cv=changed_transform),
                replace(frame, depth_intrinsics=replace(frame.depth_intrinsics, fx=301)),
                replace(frame, color_intrinsics=replace(frame.color_intrinsics, fy=301)),
                replace(frame, color_intrinsics=replace(frame.color_intrinsics,
                    distortion=(.001, 0., 0., 0., 0.)))]
    for changed in variants:
        assert not surface.matches_frame(changed, np.eye(4))
    d2c = np.eye(4)
    d2c[0, 3] = .0001
    assert not surface.matches_frame(frame, d2c)


@pytest.mark.parametrize('shape', ['rgb', 'depth'])
def test_surface_identity_tracks_native_image_shapes(shape):
    frame, green, plane, _, _ = make_scene()
    surface = BeltSurface.from_frame(frame, np.eye(4), fit_plane=plane, green_belt_points=green)
    changed = SimpleNamespace(**vars(frame))
    if shape == 'rgb':
        changed.rgb = frame.rgb[:-1]
    else:
        changed.depth_m = frame.depth_m[:-1]
    assert not surface.matches_frame(changed, np.eye(4))
