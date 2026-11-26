"""No downward TCP targets during close, braking and the handoff at rest."""
import numpy as np

from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .trajectory import sampling_times

EPS_M = 1e-9  # numerical equality only; no physical deadband


def reference_options(config, target):
    rise = config.raw.get('brake_no_descent_rise_m', 0.)
    if not rise:
        return {}
    # Complete the vertical approach before the physical (advanced) close
    # signal. Allow one coarse IK interval for transitioning to tracking.
    hz = config.raw['planning_knot_hz']
    lead = (np.ceil(config.raw['command_advance_s']*hz)+1)/hz
    if config.raw.get('object_relative_approach'):
        # The relative curve places an exact IK knot at the end of descent.
        # Leave an extra half coarse interval before the existing 100 Hz FK
        # certificate, which still starts before the earliest close command.
        lead = config.raw['command_advance_s'] + 1./config.settings.control_hz + .5/hz
    normal = axes_for(target.detection['plane']['normal_base'])[1]
    return dict(brake_rise_m=rise, height_lead_s=float(lead),height_normal_world=tuple(normal))


def axes_for(normal):
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or n[2] <= 0:
        raise ValueError('缺少有效的本次曝光带面法向')
    return np.array([[0.,0.,1.], n/np.linalg.norm(n)])


class TcpFK:
    """Private kinematics workspace; never mutate the planner during a hold."""
    def __init__(self, model):
        self.robot = model
        self.data = model.mj.MjData(model.model)

    def __call__(self, q):
        m = self.robot
        self.data.qpos[m.indices] = q
        m.mj.mj_kinematics(m.model, self.data)
        return self.data.site_xpos[m.tcp_id].copy()


def certify(config, model, normal, path, ref, cancelled):
    rise = config.raw['brake_no_descent_rise_m']
    advance = config.raw['command_advance_s']
    if getattr(ref, 'brake_rise_m', None) != rise or ref.height_lead_s < advance:
        raise PickRejected('该抓取参考没有提前结束下降，拒绝交付')
    # In periodic sweep mode step() can prepare the close one period early.
    start = ref.duration_s-advance-1./config.settings.control_hz
    if start < 0:
        raise PickRejected('闭爪保护窗口无效')
    axes = axes_for(normal)
    hz = config.settings.control_hz
    offsets = sampling_times(path.duration-start, 1./hz, ref.duration_s-start, ref.follow_end_s-start)
    fk = TcpFK(model)
    peak = None
    for dt in offsets:
        if cancelled():
            raise PickRejected('抓取高度认证已取消或超时')
        h = axes @ fk(path.position(start+dt))
        if not np.isfinite(h).all():
            raise PickRejected('抓取高度认证遇到非有限坐标')
        if peak is not None and np.any(h < peak-EPS_M):
            raise PickRejected('闭爪至减速终点关节曲线出现下移', details={
                'reason':'planned_tcp_descent', 'elapsed_s':float(start+dt),
                'drop_base_z_belt_m':(peak-h).tolist()})
        peak = h if peak is None else np.maximum(peak,h)
    return dict(start_s=start, end_s=path.duration, rise_m=rise, samples=len(offsets),
                check_hz=hz, normal=axes[1].tolist())


class HeightViolation(RuntimeError):
    """Terminal fault: the executor stops the device; no automatic recovery route."""
    def __init__(self, reason, **details):
        super().__init__('闭爪高度保护停止运动：'+reason)
        self.details = dict(reason='tcp_no_descent', cause=reason, **details)


class HeightGuard:
    def __init__(self, normal, tcp, fk, on_fault=None):
        self.axes = axes_for(normal)
        self.tcp, self.fk = tcp, fk
        self.on_fault = on_fault
        self.fault = None
        self.armed = False
        self.lift_start = None
        self.stamp = self.measured = self.commanded = None
        self.max_feedback_drop = np.zeros(2)
        self.feedback_samples = self.invalid_samples = 0
        self.by_stage = {}

    def observe(self, state, stage=None):
        # Diagnostic only. Never use measured height to reject a command,
        # release a handoff, or modify the commanded rest position.
        try:
            stamp = mapped_stamp(state,'pose')
            if stamp is None or not np.isfinite(stamp) or (self.stamp is not None and stamp < self.stamp):
                self.invalid_samples += 1
                return
            if stamp == self.stamp:
                return
            h = self.axes @ (pose_xyz_rpy(state['pose']) @ self.tcp)[:3,3]
            if not np.isfinite(h).all():
                self.invalid_samples += 1
                return
        except (KeyError, TypeError, ValueError):
            self.invalid_samples += 1
            return
        self.measured = h if self.measured is None else np.maximum(self.measured,h)
        self.max_feedback_drop = np.maximum(self.max_feedback_drop,self.measured-h)
        self.feedback_samples += 1
        self.stamp = stamp
        if stage is not None:
            entry = self.by_stage.get(stage)
            if entry is None:
                entry = self.by_stage[stage] = dict(peak=h.copy(),drop=np.zeros(2),samples=0)
            entry['peak'] = np.maximum(entry['peak'],h)
            entry['drop'] = np.maximum(entry['drop'],entry['peak']-h)
            entry['samples'] += 1

    def feedback_report(self):
        return dict(mode='observe_only', distinct_samples=self.feedback_samples,
                    invalid_samples=self.invalid_samples, last_pose_monotonic_s=self.stamp,
                    maximum_base_z_drawdown_mm=float(self.max_feedback_drop[0]*1000),
                    maximum_belt_normal_drawdown_mm=float(self.max_feedback_drop[1]*1000),
                    stages={k:dict(samples=v['samples'],max_drop_base_z_belt_mm=(1000*v['drop']).tolist())
                            for k,v in self.by_stage.items()})

    def trip(self, reason, **details):
        if self.fault is None:
            self.fault = HeightViolation(reason, **details)
            if self.on_fault is not None:
                self.on_fault(self.fault)
        raise self.fault

    def _check(self, h, peak, reason, stamp):
        if not np.isfinite(h).all():
            self.trip('高度值无效')
        if peak is not None and np.any(h < peak-EPS_M):
            self.trip(reason, pose_monotonic_s=stamp,
                height_base_z_belt_m=h.tolist(), peak_base_z_belt_m=peak.tolist(),
                drop_base_z_belt_mm=((peak-h)*1000).tolist())

    def check_command(self, q):
        h = self.axes @ self.fk(q)
        self._check(h,self.commanded,'待发送的关节目标会使 TCP 下移',self.stamp)
        return h

    def record_sent(self, q):
        h = self.axes @ self.fk(q)
        self.commanded = h if self.commanded is None else np.maximum(h,self.commanded)

    def lift_started(self):
        if self.lift_start is None:
            self.lift_start = self.commanded.copy()
            return False
        return np.all(self.commanded > self.lift_start+EPS_M)
