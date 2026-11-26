"""Scene preparation owns one exposure; only its requesting planner can publish."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from nero_core.two_second.parallel_scene import ParallelBeltScene, snapshot_target
from nero_core.two_second.planning import MeasuredPlanner
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from test_belt_collision_scope import scene_fixture
from test_two_second_hardware import site


@pytest.mark.parametrize('n', [0, 1, 127])
def test_voxel_coverage_preserves_every_observation_and_native_row_order(n):
    from mujoco_conveyor_sorting.runtime.scene_region_filter import voxel_downsample
    rng = np.random.default_rng(2511)
    points = rng.uniform(-.008, .008, size=(n, 3))
    if n > 5:
        points[:3] = points[4]
    r = voxel_downsample(points, voxel_size_m=.005, max_error_m=.001)
    errors = np.linalg.norm(points-points[r.voxel_primary_indices[r.voxel_inverse]], axis=1)
    coverage = np.where(errors > .001, np.arange(n), r.voxel_primary_indices[r.voxel_inverse])
    indices, counts = np.unique(coverage, return_counts=True)
    np.testing.assert_array_equal(r.coverage_indices, coverage)
    np.testing.assert_array_equal(r.retained_indices, indices)
    np.testing.assert_array_equal(r.representative_counts, counts)
    distances = np.linalg.norm(points-points[coverage], axis=1)
    np.testing.assert_array_equal(r.representative_max_errors_m,
                                  [max(distances[coverage == row]) for row in indices])


@pytest.mark.parametrize('extreme', [-(1<<20), (1<<20)-1, -(1<<20)-1, 1<<20])
def test_voxel_keys_and_native_ties_at_packed_coordinate_limits(extreme):
    from mujoco_conveyor_sorting.runtime.scene_region_filter import voxel_downsample
    size = 1./256.
    cells = np.array([[extreme, -1, 1], [1, extreme, -1], [1, -1, extreme],
                      [0, 1, 2], [1, 0, 2], [-1, -1, -1], [0, 1, 2]])
    # Two equidistant observations per cell. The earliest row must win.
    points = np.vstack(((cells+.25)*size, (cells+.75)*size))[::-1]
    r = voxel_downsample(points, voxel_size_m=size)
    expected, first, inverse, counts = np.unique(np.floor(points/size).astype(np.int64),
        axis=0, return_index=True, return_inverse=True, return_counts=True)
    np.testing.assert_array_equal(r.voxel_keys, expected)
    np.testing.assert_array_equal(r.voxel_inverse, inverse)
    np.testing.assert_array_equal(r.voxel_counts, counts)
    np.testing.assert_array_equal(r.voxel_primary_indices, first)


def fixture():
    model, target = scene_fixture()
    target.points = np.array([[0., 0., .04], [.01, .02, .05]])
    model.check_feedback = lambda state: None
    planner = MeasuredPlanner(SimpleNamespace(raw={
        'environment_collision_scope': 'measured_belt_surface_only'}), model=model)
    return planner, target


def test_snapshot_survives_new_camera_arrays_and_calibration():
    _, target = fixture()
    original = deepcopy(target)
    snapshot = snapshot_target(target)
    target.detection['scene_points'][:] = 99.
    target.detection['scene_pixel_indices'][:] = 99
    target.detection['belt_region']['plane_to_base'][:] = 0.
    target.points[:] = 99.
    for key in ('scene_points', 'scene_pixel_indices'):
        np.testing.assert_array_equal(snapshot.detection[key], original.detection[key])
        assert not snapshot.detection[key].flags.writeable
    np.testing.assert_array_equal(snapshot.detection['belt_region']['plane_to_base'],
                                  original.detection['belt_region']['plane_to_base'])
    np.testing.assert_array_equal(snapshot.points, original.points)


def test_parallel_and_serial_install_have_identical_geometry_and_indices():
    serial, target = fixture()
    parallel, _ = fixture()
    serial.install(target, {})
    prior = parallel.model.scene
    with ParallelBeltScene(parallel, target) as job:
        job.start()
        job.future.result()
        assert parallel.model.scene is prior  # Completion alone cannot publish.
        job.install(target, {})
    np.testing.assert_array_equal(serial.model.scene, parallel.model.scene)
    np.testing.assert_array_equal(serial.target_tree.data, parallel.target_tree.data)
    for key in ('scene_source_indices', 'scene_native_pixel_indices'):
        for name, values in getattr(serial.model, key).items():
            np.testing.assert_array_equal(values, getattr(parallel.model, key)[name])
    assert serial.scene_report == parallel.scene_report
    assert serial.version == parallel.version == 1
    assert serial.scene_revision == parallel.scene_revision == 1
    assert not parallel.model.scene.flags.writeable


@pytest.mark.parametrize('change', ['target', 'version', 'revision', 'cancel'])
def test_late_or_cancelled_result_never_overwrites_current_scene(change):
    planner, target = fixture()
    prior = planner.model.scene
    with ParallelBeltScene(planner, target) as job:
        job.start()
        if change == 'version': planner.version += 1
        if change == 'revision': planner.scene_revision += 1
        with pytest.raises(PickRejected):
            job.install(deepcopy(target) if change == 'target' else target, {},
                        cancelled=lambda: change == 'cancel')
    assert planner.model.scene is prior


def test_version_change_while_waiting_is_rechecked(monkeypatch):
    planner, target = fixture()
    prior = planner.model.scene
    with ParallelBeltScene(planner, target) as job:
        job.start()
        result = job.future.result
        def wait_and_invalidate():
            value = result()
            planner.version += 1
            return value
        monkeypatch.setattr(job.future, 'result', wait_and_invalidate)
        with pytest.raises(PickRejected, match='版本'):
            job.install(target, {})
    assert planner.model.scene is prior


def test_worker_failure_and_caller_exception_join_without_partial_publish():
    planner, target = fixture()
    prior = planner.model.scene
    target.detection['scene_points'][:] = 99.
    with pytest.raises(PickRejected, match='没有有效'):
        with ParallelBeltScene(planner, target) as job:
            job.start()
            job.install(target, {})
    assert job.future.done() and planner.model.scene is prior
    with pytest.raises(RuntimeError, match='cancel request'):
        with ParallelBeltScene(planner, target) as abandoned:
            abandoned.start()
            raise RuntimeError('cancel request')
    assert abandoned.future.done() and planner.model.scene is prior


def test_all_out_of_bounds_candidates_do_not_launch_a_scene_worker(monkeypatch):
    from test_coarse_first_pick import fixture as candidate_fixture, run
    planner, target, *_ = candidate_fixture(monkeypatch)
    planner.config.raw.update(parallel_belt_scene_enabled=True,
        environment_collision_scope='measured_belt_surface_only')
    planner.config.workspace[:] = [[10., 11.]]*3
    monkeypatch.setattr(ParallelBeltScene, 'start', lambda self: pytest.fail('out of bounds scene'))
    with pytest.raises(PickRejected):
        run(planner, target)


@pytest.mark.parametrize('value', [0, 1, 'false', None])
def test_parallel_switch_is_an_explicit_boolean(site, value):
    import yaml
    from nero_core.two_second.config import CellConfig
    path, raw = site
    raw['parallel_belt_scene_enabled'] = value
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='parallel_belt_scene_enabled'):
        CellConfig.load(path)
