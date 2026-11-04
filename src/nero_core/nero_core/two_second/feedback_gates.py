"""Shared measured-time gates for hardware execution and physical simulation."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselinePolicyConfig
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected

# Startup recovery keeps its original tolerance; trajectory endpoints allow 1 degree.
JOINT_ARRIVAL_TOLERANCE_RAD = .005
TRAJECTORY_ARRIVAL_TOLERANCE_RAD = float(np.deg2rad(1.))
ARRIVAL_DWELL_S = .1
GRIP_CONFIRMATION_DWELL_S = .2


def closure_feedback(reference, tcp, stamp, previous_tcp, previous_stamp, *,
                     command_elapsed, support=None, gates=None, follow_end_s=None):
    """Use exposure/pose time for alignment, future command time for remaining follow."""
    gates = BaselinePolicyConfig() if gates is None else gates
    normal = np.array([0., 0., 1.]) if support is None else np.asarray(support, float)
    normal = normal/np.linalg.norm(normal)
    xyz = np.asarray(tcp)[:3, 3]
    error = reference.capture_tcp_world+reference.velocity_world*(stamp-reference.capture_s)-xyz
    vertical = float(error@normal)
    horizontal = float(np.linalg.norm(error-normal*vertical))
    rotation = Rotation.from_quat(reference.target_quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
    angle = float(Rotation.from_matrix(rotation.T@np.asarray(tcp)[:3, :3]).magnitude())
    velocity, drift_ok = None, False
    valid_clock = (previous_stamp is not None and previous_tcp is not None and
                   np.isfinite([stamp, previous_stamp]).all() and stamp > previous_stamp)
    if valid_clock:
        velocity = reference.velocity_world-(xyz-np.asarray(previous_tcp)[:3, 3])/(stamp-previous_stamp)
        residual = error+velocity*gates.close_settle_time
        vz = float(residual@normal)
        drift_ok = (abs(vz) <= gates.closure_vertical_tolerance and
                    np.linalg.norm(residual-normal*vz) <= gates.closure_horizontal_tolerance)
    follow_end = (getattr(reference, 'follow_end_s', reference.duration_s+.6)
                  if follow_end_s is None else follow_end_s)
    remaining = float(follow_end-command_elapsed)
    eligible = (stamp-reference.start_s >= reference.duration_s-1e-9 and
        remaining >= GRIP_CONFIRMATION_DWELL_S-1e-9 and
        horizontal <= gates.closure_horizontal_tolerance and
        abs(vertical) <= gates.closure_vertical_tolerance and
        angle <= gates.closure_orientation_tolerance and drift_ok)
    return bool(eligible), {'horizontal_error_m': horizontal, 'vertical_error_m': vertical,
        'orientation_error_rad': angle, 'relative_velocity_m_s': velocity,
        'remaining_follow_s': remaining, 'drift_ok': bool(drift_ok),
        'pose_elapsed_s': float(stamp-reference.start_s), 'command_elapsed_s': float(command_elapsed)}


@dataclass
class GripConfirmation:
    open_width: float
    minimum_force: float
    requested_stamp: float
    since: float | None = None
    previous_stamp: float | None = None
    confirmed: bool = False

    def observe(self, width, force, stamp):
        if not np.isfinite([width, force, stamp]).all():
            raise PickRejected('夹持确认反馈包含非有限数据')
        if self.previous_stamp is not None and stamp < self.previous_stamp:
            raise PickRejected('夹持确认期间反馈时钟倒退')
        if stamp <= self.requested_stamp or stamp == self.previous_stamp:
            return self.confirmed
        self.previous_stamp = stamp
        held = .003 < width < self.open_width-.004 and force >= self.minimum_force
        if not held:
            self.since, self.confirmed = None, False
        elif self.since is None:
            self.since = stamp
        elif stamp-self.since >= GRIP_CONFIRMATION_DWELL_S-1e-9:
            self.confirmed = True
        return self.confirmed


@dataclass
class RestArrival:
    not_before: float
    since: float | None = None
    previous_stamp: float | None = None
    tolerance_rad: float = JOINT_ARRIVAL_TOLERANCE_RAD

    def observe(self, actual, target, stamp):
        actual, target = np.asarray(actual), np.asarray(target)
        if (actual.shape != (7,) or target.shape != (7,) or
                not np.isfinite(actual).all() or not np.isfinite(target).all() or
                stamp is None or not np.isfinite(stamp)):
            raise PickRejected('轨迹到位反馈包含无效或非有限数据')
        if self.previous_stamp is not None and stamp < self.previous_stamp:
            raise PickRejected('轨迹到位期间关节反馈时钟倒退')
        if stamp == self.previous_stamp:
            return False
        self.previous_stamp = stamp
        if stamp < self.not_before or np.max(abs(np.asarray(actual)-target)) > self.tolerance_rad:
            self.since = None
            return False
        if self.since is None:
            self.since = stamp
        return stamp-self.since >= ARRIVAL_DWELL_S-1e-9


@dataclass
class OpenConfirmation:
    open_width: float
    since: float | None = None
    previous_stamp: float | None = None
    not_before: float = float('-inf')

    def observe(self, width, stamp):
        if not np.isfinite([width, stamp]).all():
            raise PickRejected('张爪确认反馈包含非有限数据')
        if self.previous_stamp is not None and stamp < self.previous_stamp:
            raise PickRejected('张爪确认期间反馈时钟倒退')
        if stamp <= self.not_before:
            return False
        if stamp == self.previous_stamp:
            return False
        self.previous_stamp = stamp
        if abs(width-self.open_width) >= .002:
            self.since = None
            return False
        if self.since is None:
            self.since = stamp
        return stamp-self.since >= .15-1e-9  # existing PickCycle.wait default
