"""Same-exposure selection for OBB, without devices or collision-cloud removal."""
import numpy as np
import pytest

from test_local_target_geometry import example
from mujoco_conveyor_sorting.runtime.single_pick import detect_target, TargetNotReady
from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
from nero_core.two_second.depth_separation import grasp_points, validate
from nero_core.two_second.vision import obb_grasps
from mujoco_conveyor_sorting.runtime.depth_separation import separate


def seeded_cloud(body_side, *patches, spacing=.001, step=.001):
    # Adjacent native pixels can be far apart in 3-D at a stereo depth edge.
    yy, xx = np.indices((body_side, body_side))
    body = np.c_[xx.ravel()*step, yy.ravel()*step, np.full(xx.size,.04)]
    points = [body, *[np.c_[((body_side-1)*step + gap) + np.arange(n)*step,
                            np.zeros(n), np.full(n, .04)] for n,gap in patches]]
    points = np.vstack(points)
    iy, ix = np.divmod(np.arange(len(points)), body_side)
    return separate(points, np.c_[iy, ix], np.ones(len(points), bool),
                    spacing=spacing, plane_rms=.002, plane_band=.004)


def test_seeded_3d_island_cannot_stretch_obb_even_when_pixels_touch():
    keep, stats = seeded_cloud(36, (5,.058))
    assert keep[:-5].all() and not keep[-5:].any()
    assert stats['disconnected_completion_removed'] == 0
    assert stats['isolated_seed_island_points_removed'] == 5
    assert stats['isolated_seed_island_count'] == 1
    assert stats['removed'] == 5


@pytest.mark.parametrize('size,patch,spacing', [
    (36, (5,.004), .001),  # actually connected at the metric link radius
    (36, (5,.016), .001),  # disconnected but close to the real edge
    (36, (5,.020), .001),  # gap boundary is preserved
    (36, (5,.024), .002),  # four scale-dependent link radii = 28 mm
    (36, (9,.058), .001),  # substantial individual patch
    (20, (5,.058), .001),  # insufficient dominance for this object size
    (6, (36,.058), .001),  # two equally supported components
])
def test_real_or_ambiguous_seeded_fragments_are_preserved(size, patch, spacing):
    keep, stats = seeded_cloud(size, patch, spacing=spacing)
    assert keep.all() and stats['isolated_seed_island_count'] == 0


def test_two_tiny_remote_clusters_are_preserved_if_close_to_each_other():
    keep, _ = seeded_cloud(36, (5,.05), (4,.066))
    assert keep.all()


def test_disabled_returns_original_cloud_without_requiring_seed_data():
    points = np.ones((10, 3))
    detection = {'object_points': points}
    assert grasp_points(detection, None) is points
    assert list(detection) == ['object_points']


def test_seed_pixels_and_separation_agree_for_local_and_full_detector():
    frame, kwargs = example()
    kwargs['retain_native_seed'] = True
    full = detect_target(frame, np.eye(4), **kwargs)
    local = detect_seed_local(frame, np.eye(4), detector=detect_target, **kwargs)
    assert local['local_processing']['applied']
    np.testing.assert_array_equal(full['object_seed_pixel_indices'], local['object_seed_pixel_indices'])
    raw = {key: local[key].copy() for key in ('object_points', 'scene_points', 'mask')}
    points = grasp_points(local, frame, enabled=True)
    np.testing.assert_array_equal(points, grasp_points(full, frame, enabled=True))
    for key, before in raw.items():
        np.testing.assert_array_equal(local[key], before)
    np.testing.assert_array_equal(points, raw['object_points'][local['grasp_point_keep_mask']])
    assert local['depth_separation']['frame_sequence'] == frame.header.sequence
    assert local['depth_separation']['collision_cloud_preserved']
    poses, box = obb_grasps(points, np.eye(4), np.eye(4))
    assert len(poses) == 2 and np.isfinite(poses).all()


def test_enabled_requires_current_native_seeds_instead_of_guessing():
    frame, kwargs = example()
    detection = detect_target(frame, np.eye(4), **kwargs)
    with pytest.raises(TargetNotReady):
        grasp_points(detection, frame, enabled=True)


def test_configuration_defaults_off_and_exposes_native_only_scope():
    raw = {}
    validate(raw)
    assert raw['depth_separation_enabled'] is False
    validate({'depth_separation_enabled': True})
    with pytest.raises(ValueError):
        validate({'depth_separation_enabled': 'true'})
    with pytest.raises(ValueError, match='Fast'):
        validate({'depth_separation_enabled': True, 'fast_stereo_enabled': True})


def test_archive_saves_exact_obb_selection_and_full_collision_cloud(tmp_path):
    from test_experiment_archive import target, load
    from nero_core.two_second.experiment_archive import ExperimentArchive
    t = target()
    keep = np.arange(len(t.points)) % 2 == 0
    t.detection.update(grasp_point_keep_mask=keep, object_seed_pixel_indices=np.array([10, 11]),
                       depth_separation=dict(enabled=True, collision_cloud_preserved=True))
    archive = ExperimentArchive(tmp_path, lambda *args, **kwargs: None)
    archive.offer_target(t)
    assert archive.close()['complete']
    metadata, arrays = load(next((tmp_path/'experiment').glob('*_target.npz')))
    np.testing.assert_array_equal(arrays['points'], t.points)
    np.testing.assert_array_equal(arrays['grasp_point_keep_mask'], keep)
    assert metadata['detection']['depth_separation']['collision_cloud_preserved']
