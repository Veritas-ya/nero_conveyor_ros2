import numpy as np
from scipy.spatial.transform import Rotation

from experiments.graspgenx_sim.cached_projection import CachedDepthProjector
from mujoco_conveyor_sorting.perception.rgbd_vision_pipeline import deproject_depth_to_world


def test_cached_projection_preserves_pixels_coordinates_and_invalidates_calibration():
    rng = np.random.default_rng(81)
    depth = rng.uniform(.2, 2., (40, 64)).astype(np.float32)
    matrix = np.array([[603.1, 0., 31.4], [0., 604.2, 20.7], [0., 0., 1.]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([.4, -.6, .7]).as_matrix()
    transform[:3, 3] = [.35, -.12, .8]
    projector = CachedDepthProjector()
    for convention in ("mujoco", "opencv"):
        for scale in (1., 1.4):
            expected = deproject_depth_to_world(depth * scale, matrix, transform,
                                                camera_convention=convention)
            actual = projector.project(depth * scale, matrix, transform,
                                       camera_convention=convention)
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
            assert actual.dtype == np.float64
    assert projector.cache_rebuilds == 2
    transform[0, 3] += .1
    matrix[0, 2] += .5
    np.testing.assert_allclose(projector.project(depth, matrix, transform),
                               deproject_depth_to_world(depth, matrix, transform),
                               rtol=1e-14, atol=1e-14)
    assert projector.cache_rebuilds == 3
