"""Measured synchronization and irreversible pre-dispatch branch selection."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp


class SynchronizationGate:
    def __init__(self, config, bundle):
        self.config, self.bundle = config, bundle
        self.spec = config.raw['hover_descent']
        self.samples = deque(maxlen=64)
        self.last = None

    def observe(self, state, now):
        spec, bundle = self.spec, self.bundle
        stamp, joint_stamp = mapped_stamp(state,'pose'), mapped_stamp(state,'joint')
        result = dict(allowed=False, feedback_s=stamp, checked_s=now)
        if (stamp is None or joint_stamp is None or not np.isfinite([stamp,joint_stamp,now]).all()
                or not 0 <= now-stamp <= spec['feedback_max_age_s']
                or not 0 <= now-joint_stamp <= spec['feedback_max_age_s']
                or abs(stamp-joint_stamp) > .02):
            return {**result,'reason':'feedback_age_or_pairing'}
        if self.last is not None and stamp <= self.last:
            return {**result,'reason':'no_new_pose_feedback'}
        self.last = stamp
        origin = bundle.old.reference.start_s
        if stamp < origin+bundle.reference.sync_s:
            self.samples.clear()
            return {**result,'reason':'not_yet_synchronized'}
        actual = pose_xyz_rpy(state['pose'])@self.config.tcp
        expected, quat = bundle.reference.sample(stamp-origin)
        delta = actual[:3,3]-expected
        angle = Rotation.from_matrix(Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().T@actual[:3,:3]).magnitude()
        qerror = np.max(abs(np.asarray(state['joints'])-bundle.prefix.position(joint_stamp-origin)))
        pose_ok = (np.linalg.norm(delta[:2]) <= spec['horizontal_tolerance_m']
            and abs(delta[2]) <= spec['height_tolerance_m'] and angle <= spec['orientation_tolerance_rad']
            and qerror <= spec['joint_tracking_tolerance_rad'])
        result.update(horizontal_error_m=float(np.linalg.norm(delta[:2])), height_error_m=float(delta[2]),
            orientation_error_rad=float(angle), joint_error_rad=float(qerror))
        if not pose_ok:
            self.samples.clear()
            return {**result,'reason':'position_or_orientation_tracking'}
        self.samples.append((stamp,actual[:3,3].copy()))
        while len(self.samples)>1 and stamp-self.samples[1][0] >= spec['observation_span_s']:
            self.samples.popleft()
        dt = stamp-self.samples[0][0]
        if len(self.samples)<3 or dt < spec['observation_span_s']-1e-9:
            return {**result,'reason':'insufficient_distinct_feedback'}
        velocity = (actual[:3,3]-self.samples[0][1])/dt
        error = float(np.linalg.norm(velocity-bundle.old.reference.velocity_world))
        return {**result,'allowed':error<=spec['speed_tolerance_m_s'],
            'reason':'synchronized' if error<=spec['speed_tolerance_m_s'] else 'velocity_tracking',
            'velocity_error_m_s':error,'measured_velocity_m_s':velocity.tolist(),'sample_span_s':dt}


class BranchDecision:
    def __init__(self, bundle, config):
        self.deadline = (bundle.old.reference.start_s+bundle.switch_s-
                         config.raw['command_advance_s']-config.raw['hover_descent']['send_guard_s'])
        self.not_before = self.deadline-2./config.settings.control_hz
        self.grasp = config.raw['hover_descent']['grasp']
        self.branch = None
        self.reason = None

    def update(self, now, gate, *, candidate_ready=False, candidate_failed=False):
        if self.branch is not None:
            return self.branch
        if now >= self.deadline:
            self.branch, self.reason = 'stop', 'switch_deadline'
        elif now < self.not_before:
            return None  # Judge the latest feedback close to dispatch, not early in the plateau.
        elif not self.grasp:
            # Continue collecting the synchronization evidence in dry runs.
            if gate.get('allowed'):
                self.branch, self.reason = 'stop', 'hover_only_test'
        elif candidate_failed:
            self.branch, self.reason = 'stop', 'suffix_rejected'
        elif candidate_ready and gate.get('allowed'):
            self.branch, self.reason = 'descent', 'fresh_feedback_and_certified_suffix'
        return self.branch
