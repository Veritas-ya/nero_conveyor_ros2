"""No devices: frame scope, unchanged geometry, indexes, bounded saving."""
from dataclasses import replace
import json
from threading import Event

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime import grasp_depth
from mujoco_conveyor_sorting.runtime.frame_products import frame_products, green_rgb_mask, world_points
from mujoco_conveyor_sorting.runtime.scene_noise_filter import filter_scene_noise, SceneNoiseFilterConfig
from mujoco_conveyor_sorting.runtime.scene_workspace import scene_workspace, spatial_tree
from mujoco_conveyor_sorting.runtime.artifact_writer import artifact_session, background_artifact
from mujoco_conveyor_sorting.runtime.visible_target import project_rgb_labels, selected_rgb_silhouette
from mujoco_conveyor_sorting.perception import DepthGeometryVisionPipeline, GeometryPrototype
from test_grasp_depth import native_observation, Reconstruction


def test_frame_cache_invalidated_by_frame_geometry_calibration_and_clock():
    frame = native_observation()['frame']
    d2c = np.eye(4)
    cache = frame_products(frame, d2c)
    assert frame_products(frame, d2c) is cache
    assert world_points(frame, d2c) is world_points(frame, d2c)
    green_rgb_mask(frame, d2c)
    green_rgb_mask(frame, d2c)
    assert cache.calls['rgb_to_hsv'] == 1
    assert cache.calls['native_deprojection'] == 1
    for changed in (replace(frame), replace(frame, calibration_id='different'),
                    replace(frame, header=replace(frame.header, clock_epoch=99)),
                    replace(frame, depth_intrinsics=replace(frame.depth_intrinsics, fx=601.))):
        assert frame_products(changed, d2c) is not cache
    moved = d2c.copy(); moved[0, 3] = .001
    assert frame_products(frame, moved) is not cache


def test_component_coordinates_are_read_only_and_legacy_call_is_compatible():
    obs = native_observation();frame = obs['frame']
    pipeline = DepthGeometryVisionPipeline(
        [GeometryPrototype(category_id=0, dimensions=(.05, .05, .05), symmetric=True)],
        support_plane_height=0., minimum_height=.012, maximum_height=.12, minimum_mask_pixels=70,
        roi_x=[-1, 1], roi_y=[-1, 1])
    points = world_points(frame, np.eye(4))
    # Here belt and base coincide. Same original values and arithmetic.
    a = pipeline.extract_components(None, frame.depth_m, frame.depth_intrinsics.matrix,
        frame.camera_to_world_cv, camera_convention='opencv')
    b = pipeline.extract_components(None, frame.depth_m, frame.depth_intrinsics.matrix,
        frame.camera_to_world_cv, camera_convention='opencv', points_world=points)
    assert len(a) == len(b)
    for first, second in zip(a, b):
        np.testing.assert_array_equal(first.mask, second.mask)
        np.testing.assert_array_equal(first.position, second.position)
    assert not points.flags.writeable
    with pytest.raises(ValueError, match='HxWx3'):
        pipeline.extract_components(None, frame.depth_m, frame.depth_intrinsics.matrix,
            frame.camera_to_world_cv, points_world=points[::2])


def test_sparse_projection_cache_keeps_label_semantics_and_new_labels():
    frame = native_observation()['frame'];d2c = np.eye(4)
    labels = np.ones(frame.depth_m.shape, np.uint8)
    subset = np.zeros_like(labels, bool);subset[300:410, 690:750] = True
    first = project_rgb_labels(frame, d2c, labels, pixel_mask=subset)
    second = project_rgb_labels(frame, d2c, labels*2, pixel_mask=subset)
    np.testing.assert_array_equal(second, first*2)
    full = project_rgb_labels(frame, d2c, labels)
    np.testing.assert_array_equal(first[subset], full[subset])
    cache = frame_products(frame, d2c).values['native_rgb_pixels']
    assert cache['computed'].all()


def prepare_local(obs):
    f = obs['frame'];d2c = obs['depth_to_color_cv'];p = frame_products(f, d2c)
    obs['points'] = world_points(f, d2c)
    transform = np.asarray(obs['belt_region']['plane_to_base'])
    inverse = grasp_depth.inverse(transform)
    obs['points_plane'] = p.get(('belt_coordinates', transform.tobytes()), lambda:
        obs['points'] @ inverse[:3, :3].T + inverse[:3, 3])
    obs['_frame_products'] = p


@pytest.mark.parametrize('edge', [False, True])
def test_local_depth_equals_full_fallback_with_low_base_and_holes(edge):
    obs = native_observation();frame = obs['frame']
    rgb = np.full_like(frame.rgb, [30, 150, 40]);rgb[388:433, 690:742] = 120
    depth = frame.depth_m.copy();depth[400:405, 696:700] = 0.
    obs['frame'] = replace(frame, rgb=rgb, depth_m=depth)
    silhouette, info = selected_rgb_silhouette(obs['frame'], np.eye(4), obs['mask'])
    obs.update(visible_target_rgb_mask=silhouette, visible_target_completion=info)
    prepare_local(obs)

    class Shoulder(Reconstruction):
        def refine(self, *args, **kwargs):
            result = super().refine(*args, **kwargs)
            result.reconstructed_depth_m[388:433, 690:742] = .994
            result.reconstructed_depth_m[390:430, 700:730] = .954
            if edge:
                result.reconstructed_depth_m[:388] = np.nan
                result.reconstructed_depth_m[433:] = np.nan
            return result

    full_obs = {k: v for k, v in obs.items() if k not in ('_frame_products', 'points_plane')}
    full = grasp_depth.refine_grasp_observation(full_obs, Shoulder(full_obs))
    local = grasp_depth.refine_grasp_observation(obs, Shoulder(obs))
    assert local['grasp_depth_refinement']['applied']
    assert local['grasp_depth_refinement']['timing_inputs']['local_native_reuse']
    np.testing.assert_array_equal(local['grasp_target_mask'], full['grasp_target_mask'])
    np.testing.assert_array_equal(local['object_points'], full['object_points'])
    np.testing.assert_array_equal(local['grasp'], full['grasp'])
    assert np.count_nonzero(local['grasp_target_mask'] & ~obs['mask']) > 0
    assert np.min(local['object_points'][:, 2]) < .007


@pytest.mark.parametrize('clusters', [0, 6, 10])
def test_knn_density_matches_exact_counts_duplicates_boundaries_empty(clusters):
    rng = np.random.default_rng(135)
    points = rng.uniform(-.02, .02, (2000, 3))
    points = np.vstack((points, np.repeat([[.1, .1, 0]], 8, 0),
        [[.5, 0, 0], [.505, 0, 0], [np.nextafter(.505, np.inf), 0, 0]]))
    for cloud in (points, np.empty((0, 3)), np.zeros((1, 3))):
        for minimum in (1, 3, 8, 20):
            config = SceneNoiseFilterConfig(min_neighbors=minimum, min_cluster_points=clusters)
            old = filter_scene_noise(cloud, config)
            new = filter_scene_noise(cloud, config, collect_neighbor_counts=False)
            np.testing.assert_array_equal(old.removed_mask, new.removed_mask)
            assert old.statistics == new.statistics


def test_spatial_indexes_only_reuse_exact_point_set_within_request():
    points = np.arange(300, dtype=float).reshape(-1, 3)
    with scene_workspace() as workspace:
        first = spatial_tree(points)
        assert spatial_tree(points.copy()) is first
        points[0, 0] += .001
        assert spatial_tree(points) is not first
        assert workspace.calls == {'tree_build': 2, 'tree_hits': 1}
    with scene_workspace():
        assert spatial_tree(points) is not first


def test_background_snapshot_is_stable_and_drains_on_exception(tmp_path):
    entered, release = Event(), Event()
    @background_artifact
    def save(path, values):
        entered.set();release.wait(2)
        np.save(path, values)
    values = np.arange(25.)
    with pytest.raises(ValueError, match='decision'):
        with artifact_session(tmp_path, max_jobs=2, max_bytes=1024) as writer:
            save(tmp_path/'values.npy', values)
            assert entered.wait(2)
            values[:] = -1
            release.set()
            raise ValueError('decision')
    np.testing.assert_array_equal(np.load(tmp_path/'values.npy'), np.arange(25.))
    report = json.loads((tmp_path/'artifact_writes.json').read_text())
    assert report['jobs'][0]['status'] == 'completed'
    assert writer.pending_jobs == 0
    assert writer.peak_jobs <= 2 and writer.peak_bytes <= 1024


def test_background_failure_and_oversize_are_explicit(tmp_path):
    @background_artifact
    def fail(values):
        raise OSError('deliberate write failure')
    with artifact_session(tmp_path, max_bytes=128) as writer:
        fail(np.zeros(2))
        fail(np.zeros(50))
    assert len(writer.records) == 2
    assert all(row['status'] == 'failed' for row in writer.records)
    assert writer.records[1]['synchronous_oversize']
    assert writer.peak_bytes <= 128


def test_caught_save_errors_are_recorded_for_queued_and_oversized_jobs(tmp_path):
    @background_artifact
    def save(values):
        return {'artifact_error': 'deliberate caught save error'}
    with artifact_session(tmp_path, max_bytes=128) as writer:
        save(np.zeros(2))
        save(np.zeros(50))
    assert all(row['status'] == 'failed' for row in writer.records)
    assert all('caught save error' in row['error'] for row in writer.records)
    assert all(row['write_thread_cpu_s'] >= 0 for row in writer.records)


def test_writer_manifest_failure_is_visible_without_rejecting_task(tmp_path):
    path = tmp_path/'not_a_directory'
    path.write_text('blocked')
    with pytest.warns(RuntimeWarning, match='diagnostic writer report failed'):
        with artifact_session(path) as writer:
            pass
    assert 'FileExistsError' in writer.report_error


def test_body_cache_invalidates_state_and_model_geometry():
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel
    model = RobotModel()
    state = {'joints': np.zeros(7), 'width_m': .1}
    original = model.self_filter_bodies_for_state(state)
    assert model.self_filter_bodies_for_state(state) is original
    assert all(not body.position.flags.writeable for body in original)
    for changed in (dict(state, joints=np.ones(7)*.001), dict(state, width_m=.099)):
        assert model.self_filter_bodies_for_state(changed) is not original
    model.self_filter_bodies_for_state(state)
    model.model.body_pos[1, 0] += .0001
    assert model.self_filter_bodies_for_state(state) is not original


def test_background_queue_pressure_waits_without_dropping(tmp_path):
    import threading
    entered, release, returned = Event(), Event(), Event()
    written = []
    @background_artifact
    def save(values):
        entered.set();release.wait(2);written.append(values.copy())
    with artifact_session(tmp_path, max_jobs=1, max_bytes=128) as writer:
        save(np.arange(4.))
        assert entered.wait(2)
        producer = threading.Thread(target=lambda: (save(np.arange(4.)+10), returned.set()))
        producer.start()
        assert not returned.wait(.02)
        release.set();producer.join(2)
        assert returned.is_set()
    assert len(written) == 2
    assert writer.peak_jobs == 1
    assert all(row['status'] == 'completed' for row in writer.records)


def test_frame_snapshot_does_not_copy_cache_lock(tmp_path):
    from mujoco_conveyor_sorting.runtime.artifact_writer import _snapshot
    frame = native_observation()['frame']
    frame_products(frame, np.eye(4))
    copied = _snapshot(frame)
    np.testing.assert_array_equal(copied.rgb, frame.rgb)
    assert copied.depth_m is not frame.depth_m
    assert copied._perception_products is None


def test_motion_region_reuses_complete_green_set_before_plane_subset():
    from test_belt_surface import make_scene
    from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
    from mujoco_conveyor_sorting.runtime.nero_moving_source import measured_belt_motion_region
    small, extract, fit, _, _ = make_scene()
    frame = replace(native_observation()['frame'], depth_m=small.depth_m,
        rgb=small.rgb, depth_intrinsics=small.depth_intrinsics,
        color_intrinsics=small.color_intrinsics,
        camera_to_world_cv=small.camera_to_world_cv)
    points, mask, _ = extract(small, np.eye(4))
    points = points.copy()
    # These remain measured green points for PCA/range, but lie outside the
    # plane's original 5 mm support subset. Caching corners/subset would differ.
    points.reshape(-1, 3)[np.flatnonzero(mask)[:60]] += .012*np.asarray(fit(None)[1]['normal_base'])
    calls = []
    def green(current, d2c):
        calls.append(current)
        return points, mask, None
    surface = BeltSurface.from_frame(frame, np.eye(4), fit_plane=fit, green_belt_points=green)
    detection = {'plane': surface.plane, 'belt_region': surface.region_metadata()}
    cached = measured_belt_motion_region(detection, frame, np.eye(4), green)
    assert len(calls) == 1
    assert cached['metadata']['native_point_count'] == np.count_nonzero(mask)
    object.__setattr__(frame, '_perception_products', None)
    uncached = measured_belt_motion_region(detection, frame, np.eye(4), green)
    assert len(calls) == 2
    assert cached == uncached
