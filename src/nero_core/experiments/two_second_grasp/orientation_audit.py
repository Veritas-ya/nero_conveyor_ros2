"""Display and compare the selected TCP pose, without controlling the robot."""
import numpy as np
from scipy.spatial.transform import Rotation


def capture_pose(reference):
    """Selected TCP orientation at the frozen exposure's coordinates."""
    pose = np.eye(4)
    pose[:3, 3] = reference.capture_tcp_world
    pose[:3, :3] = Rotation.from_quat(reference.target_quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    return pose


def orientation_error_deg(actual_wxyz, reference_wxyz):
    actual = Rotation.from_quat(np.asarray(actual_wxyz)[[1, 2, 3, 0]])
    reference = Rotation.from_quat(np.asarray(reference_wxyz)[[1, 2, 3, 0]])
    return float(np.degrees((reference.inv() * actual).magnitude()))


def preferred_obb_axis(candidate_wxyz, default_pose):
    """Group the default OBB grasp and its 180-degree flip before the other axis."""
    candidate = Rotation.from_quat(np.asarray(candidate_wxyz)[[1, 2, 3, 0]]).as_matrix()
    alignment = abs(float(candidate[:, 0] @ np.asarray(default_pose)[:3, 0]))
    return 0 if alignment > 2**-.5 else 1
