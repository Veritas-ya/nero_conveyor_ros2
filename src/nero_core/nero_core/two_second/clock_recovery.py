"""Read-only recovery handshake after a proven host wall-clock adjustment.

The interrupted trajectory stays cancelled. Never remap cached packets or
change the CAN adapter's epoch; replace only the executor's checked baseline.
"""
import math
import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .streaming import fresh_feedback


PACKETS = ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz',
           'joint_12', 'joint_34', 'joint_56', 'joint_7', 'gripper')
HOST_DOMAIN = 'host_monotonic_from_can_unix'
TIMEOUT_S = .5
STABLE_S = .1
STABLE_SPREAD_RAD = .001
STEP_THRESHOLD_S = .005


def _reject(reason):
    raise PickRejected('系统校时后的反馈不能建立恢复基准：'+reason,
        details={'reason':'host_clock_recovery_invalid','cause':reason})


def _evidence(evidence):
    if not isinstance(evidence, dict) or any(key not in evidence for key in PACKETS):
        _reject('缺少完整的CAN接收时间证据')
    result = {}
    for key in PACKETS:
        item = evidence[key]
        if (not isinstance(item, dict) or item.get('mapping_quality') != 'mapped'
                or item.get('clock_domain') != HOST_DOMAIN
                or item.get('encoder_sample_time') is not None):
            _reject('反馈不是有效的主机接收时间映射')
        fields = ('raw_unix_timestamp', 'mapped_monotonic', 'host_monotonic_received',
                  'reference_wall', 'reference_monotonic', 'reference_uncertainty_s')
        if any(not isinstance(item.get(k), (int, float)) or not math.isfinite(item[k])
               for k in fields):
            _reject('时间映射包含无效数值')
        epoch = item.get('clock_epoch')
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            _reject('时钟代次无效')
        uncertainty = item['reference_uncertainty_s']
        tolerance = uncertainty+2*math.ulp(item['reference_wall'])+1e-6
        mapped = item['mapped_monotonic']
        predicted = item['raw_unix_timestamp']+item['reference_monotonic']-item['reference_wall']
        if (not 0 <= uncertainty <= .005 or item['raw_unix_timestamp'] <= 0
                or abs(mapped-predicted) > tolerance
                or mapped > item['host_monotonic_received']+tolerance
                or item['host_monotonic_received'] > item['reference_monotonic']+uncertainty+1e-6):
            _reject('CAN接收与映射参考不一致')
        result[key] = dict(item)
    return result


def rebase_after_host_clock_step(e, *, epoch):
    """Require one coherent host step and 100 ms of fresh, stable measured pose.

    Caller has joined the sender and invalidated the failed trajectory. No
    command, gripper action, or timestamp-baseline write occurs before success.
    """
    if e.sender is not None:
        _reject('旧发送器尚未退出')
    baseline = dict(e.feedback_stamps)
    old = _evidence(baseline.get('clock_mapping_evidence'))
    old_epochs = {v['clock_epoch'] for v in old.values()}
    if len(old_epochs) != 1:
        _reject('旧反馈代次不一致')
    old_epoch = old_epochs.pop()
    new_epoch = old_epoch+1
    anchor = np.asarray(e.command_history.q, dtype=float)
    if anchor.shape != (7,) or not np.isfinite(anchor).all():
        _reject('缺少最后已发送的关节目标')
    anchor = anchor.copy()
    began = e.clock()
    identity, version = e.owner, e.version
    last_packets = {key: old[key]['mapped_monotonic'] for key in PACKETS}
    local_stamps = {key: baseline[key] for key in ('pose', 'joint', 'gripper')}
    window = []
    previous_counted = None
    observed_step = None
    checks = 0
    last_error = None
    e._emit('recovery_clock_wait', {'previous_epoch':old_epoch,
        'timeout_s':TIMEOUT_S,'required_stable_s':STABLE_S,'recovery_targets_sent':0})
    while True:
        e._active(identity, version, epoch)
        now = e.clock()
        if now-began > TIMEOUT_S:
            raise PickRejected('系统校时后等待连续稳定反馈超时', details={
                'reason':'host_clock_recovery_timeout','waited_s':now-began,
                'fk_checks':checks,'last_check':last_error})
        state = e.device.snapshot_nowait()  # retains CAN/enable/mode/driver checks
        if state is not None:
            fresh_feedback(state, e.clock())
            current = _evidence(state.get('feedback_time'))
            now = e.clock()
            if any(not 0 <= now-v['mapped_monotonic'] <= .2
                   or not 0 <= now-v['host_monotonic_received'] <= .2
                   or not 0 <= now-v['reference_monotonic'] <= .2
                   for v in current.values()):
                _reject('接收时间证据过期或晚于当前时间')
            for kind, keys in (('pose', PACKETS[:3]), ('joint', PACKETS[3:7]),
                               ('gripper', PACKETS[7:])):
                if abs(mapped_stamp(state, kind)-min(current[k]['mapped_monotonic'] for k in keys)) > 1e-6:
                    _reject('反馈时间与接收证据不一致')
            stamps = tuple(current[key]['mapped_monotonic'] for key in PACKETS)
            for key, stamp in zip(PACKETS, stamps):
                if stamp < last_packets[key]:
                    _reject('单调反馈时间倒退')
                last_packets[key] = stamp
            epochs = {v['clock_epoch'] for v in current.values()}
            if not epochs <= {old_epoch, new_epoch}:
                _reject('反馈代次倒退或发生第二次校时')
            # A cached jaw packet can straddle the change. It does not count
            # toward confirmation; all eight packets must reach the new epoch.
            if epochs == {new_epoch}:
                if any(current[k]['mapped_monotonic'] <= old[k]['mapped_monotonic'] for k in PACKETS):
                    _reject('新代次反馈没有更新')
                shifts = np.array([
                    (current[k]['reference_wall']-current[k]['reference_monotonic'])-
                    (old[k]['reference_wall']-old[k]['reference_monotonic']) for k in PACKETS])
                step = float(np.median(shifts))
                if (abs(step) <= STEP_THRESHOLD_S or np.ptp(shifts) > STEP_THRESHOLD_S
                        or (observed_step is not None and abs(step-observed_step) > STEP_THRESHOLD_S)):
                    _reject('各路反馈不对应同一次主机校时')
                observed_step = step
                # Every constituent packet must be newly received. Repeated
                # snapshots never extend the stable-observation window.
                fresh_feedback(state, e.clock(), local_stamps)
                if previous_counted is None or all(a > b for a, b in zip(stamps, previous_counted)):
                    if previous_counted is not None and any(a-b > .2 for a,b in zip(stamps,previous_counted)):
                        window.clear()  # reacquired feedback starts a new stable window
                    previous_counted = stamps
                    q = np.asarray(state['joints'], dtype=float)
                    if q.shape != (7,) or not np.isfinite(q).all():
                        _reject('关节位置无效')
                    checks += 1
                    try:
                        e.planner.model.check_feedback(state)
                    except PickRejected as error:
                        from .recovery import _fk_mismatch
                        if not _fk_mismatch(error):
                            raise
                        last_error = {'fk_error':getattr(error, 'details', {})}
                        window.clear()
                    else:
                        anchor_error = float(np.max(abs(q-anchor)))
                        last_error = {'last_target_error_rad':anchor_error}
                        if anchor_error > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
                            window.clear()
                        else:
                            window.append((min(stamps), q.copy()))
                            while len(window) > 1 and window[-1][0]-window[1][0] >= STABLE_S:
                                window.pop(0)
                            spread = float(np.max(np.ptp([row[1] for row in window], axis=0)))
                            if (len(window) >= 3 and window[-1][0]-window[0][0] >= STABLE_S-1e-9
                                    and spread <= STABLE_SPREAD_RAD):
                                e._active(identity, version, epoch)
                                fresh_feedback(state, e.clock(), local_stamps)
                                if e.clock()-began > TIMEOUT_S:
                                    continue
                                e.feedback_stamps = dict(local_stamps)
                                e._emit('recovery_clock_rebased', {'previous_epoch':old_epoch,
                                    'current_epoch':new_epoch,'host_wall_step_s':observed_step,
                                    'waited_s':e.clock()-began,'stable_samples':len(window),
                                    'stable_span_s':window[-1][0]-window[0][0],
                                    'joint_spread_rad':spread,'last_target_error_rad':anchor_error,
                                    'fk_checks':checks,'recovery_targets_sent':0})
                                return state
            elif previous_counted is not None:
                _reject('确认期间重新出现旧代次反馈')
        remaining = TIMEOUT_S-(e.clock()-began)
        e.sleep(min(.005, max(0., remaining)+1e-9))
