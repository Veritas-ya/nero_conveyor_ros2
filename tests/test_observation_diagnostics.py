import numpy as np
from experiments.two_second_grasp.observation_diagnostics import inspect_observation
from experiments.two_second_grasp.speed_observation import percentile_cloud_center


def test_metric_projection_and_detection_are_isolated_by_identity():
    labels = np.array([[1, 1, 2], [1, 1, 2]])
    depth = np.array([[1., 1., 4.], [1., 1., 4.]])
    frame = {'top_instance_labels': labels, 'top_depth': depth,
             'top_camera': {'intrinsics': np.eye(3), 'camera_to_world_opencv': np.eye(4)}}
    result = inspect_observation(frame, 1)
    np.testing.assert_allclose(result['pixel_center_uv'], [.5, .5])
    np.testing.assert_allclose(result['deprojected_center_world_m'], [.5, .5, 1.])
    np.testing.assert_allclose(result['point_cloud_mean_world_m'], [.5, .5, 1.])
    assert result['point_count'] == 4
    assert inspect_observation(frame, 99) is None


def test_speed_position_is_from_same_target_and_matches_diagnostic():
    transform = np.eye(4)
    optical = np.diag([1., -1., -1., 1.])
    frame = {'top_instance_labels': np.array([[1, 1, 2], [1, 1, 2]]),
             'top_depth': np.array([[1., 1., 4.], [1., 1., 4.]]),
             'top_camera': {'intrinsics': np.eye(3), 'camera_to_world': transform,
                            'camera_to_world_opencv': optical}}
    diagnostic = inspect_observation(frame, 1)
    np.testing.assert_allclose(percentile_cloud_center(frame, 1),
                               diagnostic['point_cloud_percentile_box_center_world_m'])
    assert percentile_cloud_center(frame, 99) is None
