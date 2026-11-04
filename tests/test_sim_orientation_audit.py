from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.two_second_grasp.orientation_audit import capture_pose, orientation_error_deg, preferred_obb_axis


def test_overlay_uses_selected_pose_in_frozen_frame():
    selected = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    ref = SimpleNamespace(capture_tcp_world=np.array([.1, .2, .3]),
                          target_quaternion_wxyz=selected)
    pose = capture_pose(ref)
    np.testing.assert_allclose(pose[:3, 3], [.1, .2, .3])
    np.testing.assert_allclose(pose[:3, :3], Rotation.from_euler('z', 90, degrees=True).as_matrix())
    assert orientation_error_deg([1., 0., 0., 0.], selected) == pytest.approx(90.)
    assert orientation_error_deg(-selected, selected) == pytest.approx(0.)


def test_preferred_obb_axis_treats_half_turn_as_equivalent():
    for angle, expected in [(0, 0), (180, 0), (90, 1), (270, 1)]:
        quat = Rotation.from_euler('z', angle, degrees=True).as_quat()[[3, 0, 1, 2]]
        assert preferred_obb_axis(quat, np.eye(4)) == expected
