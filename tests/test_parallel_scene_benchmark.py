"""Offline benchmark snapshots must not change under a newer camera result."""
from types import SimpleNamespace
import numpy as np
import pytest
from benchmark_parallel_scene import BenchmarkPlanner


def test_worker_snapshot_owns_scene_and_calibration_arrays():
    points=np.arange(12.,dtype=float).reshape(4,3)
    pixels=np.arange(4,dtype=np.int64)
    plane=np.eye(4);bounds=np.array([[-1.,1.],[-2.,2.]])
    target_points=np.ones((21,3));target_points.setflags(write=False)
    target=SimpleNamespace(identity=(7,3,9),epoch=7,sequence=42,capture_s=1.,points=target_points,
        detection={'scene_points':points,'scene_pixel_indices':pixels,
                   'belt_region':{'plane_to_base':plane,'bounds_xy_m':bounds}})
    snapshot=BenchmarkPlanner._snapshot_target(target)
    points[:]=999.;pixels[:]=99;plane[:]=0.;bounds[:]=0.
    assert snapshot.identity==(7,3,9) and snapshot.sequence==42
    np.testing.assert_array_equal(snapshot.detection['scene_points'],np.arange(12.).reshape(4,3))
    np.testing.assert_array_equal(snapshot.detection['scene_pixel_indices'],np.arange(4))
    np.testing.assert_array_equal(snapshot.detection['belt_region']['plane_to_base'],np.eye(4))
    assert not snapshot.detection['scene_points'].flags.writeable
    assert not snapshot.detection['scene_pixel_indices'].flags.writeable
    assert not snapshot.detection['belt_region']['plane_to_base'].flags.writeable


def test_prototype_does_not_install_a_different_target():
    planner=object.__new__(BenchmarkPlanner)
    planner._parallel=True
    planner._benchmark_target=object()
    with pytest.raises(ValueError,match='identity mismatch'):
        planner.install(object(),{})
