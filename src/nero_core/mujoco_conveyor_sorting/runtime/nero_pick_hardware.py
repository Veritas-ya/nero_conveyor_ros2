"""NERO SDK adapter for the standalone pick test. Imports SDK only on construction."""
import sys
from pathlib import Path
from threading import RLock, local
import time
import numpy as np
from .feedback_clock import FeedbackClock
from .motion_lease import MotionLease
from .feedback_pairing import JointFeedbackHistory
from .coherent_flange import CoherentFlangeCache, POSE_CAN_IDS
ROOT = Path(__file__).resolve().parents[2]
FEEDBACK_MAX_AGE_S = .2
POSE_JOINT_PACKETS = ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz',
                      'joint_12', 'joint_34', 'joint_56', 'joint_7')
JOINT_CAN_AXES={0x2A5:(0,1),0x2A6:(2,3),0x2A7:(4,5),0x2A9:(6,)}

class FeedbackError(RuntimeError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


class FeedbackPending(FeedbackError):
    """A receive update can resolve this; controller faults must not be retried."""
    def __init__(self, missing=(), stale=None):
        stale = {} if stale is None else stale
        parts = []
        if missing:
            parts.append('未收到反馈包：'+', '.join(missing))
        if stale:
            parts.append('反馈未更新：'+', '.join(
                f'{label}（距上次接收{age:.3f}s）' for label, age in stale.items()))
        super().__init__('；'.join(parts), details={
            'missing_packets': list(missing), 'stale_packets_age_s': dict(stale)})


def fresh(message, label, now, *, timing=None):
    if message is None:
        raise FeedbackPending(missing=[label])
    if timing is not None:
        if timing.mapping_quality != 'mapped' or timing.mapped_monotonic is None:
            raise FeedbackPending(missing=[label+':clock_mapping_'+timing.mapping_quality])
        stamp = timing.mapped_monotonic
    else:
        stamp = float(message.timestamp)
    age = now-stamp
    if not np.isfinite(age) or age < 0:
        raise FeedbackError(f'{label}反馈时间戳无效或晚于本机时间', details={
            'packet': label, 'age_s': age if np.isfinite(age) else None})
    if age > FEEDBACK_MAX_AGE_S:
        raise FeedbackPending(stale={label: age})
    return message.msg


class Hardware:
    """Explicit SDK adapter. Construction/connect never enables or moves arm."""
    def __init__(self, channel, firmware, execute, joint_receive_offset_s=0.):
        if not np.isfinite(joint_receive_offset_s):
            raise ValueError('关节与法兰接收时间偏移必须为有限数')
        self.joint_receive_offset_s = float(joint_receive_offset_s)
        sys.path.insert(0, str(ROOT/'vendor/pyAgxArm'))
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        config=create_agx_arm_config(robot='nero', firmeware_version=firmware,
                                    interface='socketcan', channel=channel,
                                    receive_own_messages=False)
        self.robot=AgxArmFactory.create_arm(config)
        self.gripper=self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.execute=execute
        self.channel=channel
        self._motion_lease=MotionLease(channel, owner='nero-sdk') if execute else None
        self._joint_command_mode='j'
        self.lock=RLock()
        self._hold_joint_target=None
        self._hold_joint_stamp=None
        self._feedback_clock = FeedbackClock()
        self._pose_group_cache = CoherentFlangeCache(max_age_s=FEEDBACK_MAX_AGE_S)
        self._pose_group_capture_error = None
        self._joint_feedback_history=JointFeedbackHistory(FEEDBACK_MAX_AGE_S)
        self._joint_history_capture_error=None
        # Copy feedback under the same lock as in-place CAN decoding.
        for parser in (self.robot._parser,self.gripper._parser):
            original=parser.parse_packet
            def guarded(*args, _original=original, **kwargs):
                callback_entered = time.monotonic()
                with self.lock:
                    frame = args[0] if args else kwargs['rx_can_frame']
                    timing = self._feedback_clock.receive((_original.__self__, frame.arbitration_id),
                        frame.timestamp, callback_entered=callback_entered)
                    decoded = _original(*args,**kwargs)
                    if decoded is not None:
                        decoded.feedback_time = timing
                    self._pose_group_cache.observe_epoch(timing.clock_epoch)
                    if (_original.__self__ is self.robot._parser and frame.arbitration_id in JOINT_CAN_AXES
                            and decoded is not None and timing.mapping_quality=='mapped'):
                        try:
                            axes=JOINT_CAN_AXES[frame.arbitration_id]
                            values=[getattr(decoded.msg,f'joint_{axis+1}') for axis in axes]
                            self._joint_feedback_history.observe_packet(axes,values,timing.mapped_monotonic,
                                epoch=timing.clock_epoch)
                            self._joint_history_capture_error=None
                        except Exception as error:
                            self._joint_history_capture_error=f'{type(error).__name__}: {error}'
                    if _original.__self__ is self.robot._parser and frame.arbitration_id in POSE_CAN_IDS:
                        try:
                            self._pose_group_cache.capture(self.robot._parser,
                                self.robot.get_flange_pose, time.monotonic())
                        except Exception as error:
                            # Do not break CAN decoding; a missing coherent
                            # group is reported by snapshot and never invented.
                            self._pose_group_capture_error = f'{type(error).__name__}: {error}'
                    return decoded
            parser.parse_packet=guarded
        comm=self.robot.create_comm()
        # Reuse the SDK receive timeout, rather than treating one aged sample
        # as a failed connection. Startup retains its existing 3 s allowance.
        self.feedback_wait_timeout_s=float(comm._timeout)
        if execute:
            original_send = comm.send
            owned_send = self._motion_lease.guard(original_send)
            self._fault_stop_scope = local()
            def leased_send(*args, **kwargs):
                # Only Hardware.stop's unchanged SDK emergency-stop call can
                # use this thread-local scope. Losing ownership must never
                # prevent the original fault stop; it grants no motion lease.
                if getattr(self._fault_stop_scope, 'active', False):
                    return original_send(*args, **kwargs)
                return owned_send(*args, **kwargs)
            comm.send = leased_send
        if not execute:
            def forbidden(*args,**kwargs):
                raise RuntimeError('只检查模式禁止任何CAN发送')
            comm.send=forbidden

    def start(self):
        lease = getattr(self, '_motion_lease', None)
        if lease is not None:lease.acquire()
        try:
            self.robot.connect()
        except BaseException:
            if lease is not None:lease.close()
            raise

    def close(self):
        try:
            self.robot.disconnect()
        finally:
            lease = getattr(self, '_motion_lease', None)
            if lease is not None:lease.close()

    def configure_joint_command_mode(self, mode):
        """Select the SDK joint API locally; do not send or switch modes yet.

        The SDK applies J/JS mode when the first corresponding movement or
        position-hold target is sent. No fallback changes the requested API.
        """
        if mode not in ('j', 'js'):
            raise ValueError('关节指令模式须为j或js')
        if not callable(getattr(self.robot, 'move_'+mode, None)):
            raise RuntimeError('当前SDK不支持move_'+mode)
        self._joint_command_mode=mode

    def _send_joint_target(self, joints):
        mode=getattr(self,'_joint_command_mode','j')
        getattr(self.robot,'move_'+mode)(list(joints))

    def snapshot(self, *, wait_timeout_s=None):
        wait_s = self.feedback_wait_timeout_s if wait_timeout_s is None else wait_timeout_s
        started = time.monotonic()
        retries = 0
        while True:
            try:
                state = self._snapshot_once()
            except FeedbackPending as error:
                elapsed = time.monotonic()-started
                if elapsed >= wait_s:
                    details = {**error.details, 'channel': self.channel,
                               'waited_s': elapsed, 'receive_wait_s': wait_s,
                               'retry_count': retries}
                    if set(POSE_JOINT_PACKETS).issubset(details['missing_packets']):
                        message = '未收到任何机械臂关节/法兰反馈，请检查机械臂端CAN通信'
                    else:
                        message = str(error)
                    raise FeedbackError(
                        f'{message}（接口{self.channel}，已等待{elapsed:.2f}s）',
                        details=details) from error
                # _snapshot_once released the parser lock. The receive thread
                # must be able to replace the missing/old packets during this wait.
                retry_s = .002 if 'coherent_flange_pose' in error.details['missing_packets'] else .02
                time.sleep(min(retry_s, wait_s-elapsed))
                retries += 1
            else:
                state['feedback_timing'].update(
                    read_wait_s=time.monotonic()-started, read_retry_count=retries)
                return state

    def snapshot_nowait(self):
        """Copy available SDK feedback without waiting for a receive retry.

        A busy parser or pending packet yields no new sample. The next normal
        cycle retains the existing SDK receive wait and fault handling. Real
        controller/driver faults still propagate immediately here.
        """
        if not self.lock.acquire(blocking=False):
            return None
        started = time.monotonic()
        try:
            try:
                state = self._snapshot_once()
            except FeedbackPending:
                return None
            state['feedback_timing'].update(
                read_wait_s=time.monotonic()-started, read_retry_count=0,
                nonblocking_cache_read=True)
            return state
        finally:
            self.lock.release()

    def snapshot_teaching_joints(self):
        """Passive joint-only feedback; permits the controller's teach mode.

        This deliberately does not establish robot readiness for motion.
        It is only available on the adapter whose CAN sender is disabled.
        """
        if self.execute:
            raise RuntimeError('示教录制仅允许只读CAN适配器')
        started = time.monotonic()
        while True:
            try:
                with self.lock:
                    if self.robot.has_comm_error():
                        raise FeedbackError(f'CAN通信异常：{self.robot.get_comm_error()}')
                    now = time.monotonic()
                    parts = [getattr(self.robot._parser, name, None) for name in POSE_JOINT_PACKETS[3:]]
                    timings = []
                    for name, part in zip(POSE_JOINT_PACKETS[3:], parts):
                        timing = getattr(part, 'feedback_time', None)
                        if timing is None:
                            raise FeedbackPending(missing=[name+':clock_mapping_missing'])
                        fresh(part, name, now, timing=timing)
                        timings.append(timing)
                    if len({t.clock_epoch for t in timings}) != 1:
                        raise FeedbackPending(missing=['joint_clock_epoch'])
                    message = self.robot.get_joint_angles()
                    if message is None:
                        raise FeedbackPending(missing=['关节'])
                    joints = list(message.msg)
                    if len(joints) != 7 or not np.isfinite(joints).all():
                        raise FeedbackError('示教关节反馈无效')
                    mapped = [t.mapped_monotonic for t in timings]
                    return {'joints': joints, 'joint_stamps_monotonic': mapped,
                        'feedback_time': {name: t.to_dict() for name, t in zip(POSE_JOINT_PACKETS[3:], timings)},
                        'clock_epoch': timings[0].clock_epoch,
                        'joint_packet_span_s': max(mapped)-min(mapped)}
            except FeedbackPending:
                if time.monotonic()-started >= self.feedback_wait_timeout_s:
                    raise
                time.sleep(.002)

    def _snapshot_once(self):
        with self.lock:
            if self.robot.has_comm_error():
                raise FeedbackError(f'CAN通信异常：{self.robot.get_comm_error()}')
            live_clock = hasattr(self, '_feedback_clock')
            now=time.monotonic() if live_clock else time.time()
            parser=self.robot._parser
            missing, stale = [], {}

            def read(message, label, *, aggregate=False):
                try:
                    if live_clock:
                        if aggregate:
                            # All real constituent packets were checked below.
                            if message is None:
                                raise FeedbackPending(missing=[label])
                            return message.msg
                        timing = getattr(message, 'feedback_time', None)
                        if timing is None:
                            raise FeedbackPending(missing=[label+':clock_mapping_missing'])
                        return fresh(message, label, now, timing=timing)
                    return fresh(message, label, now)
                except FeedbackPending as error:
                    missing.extend(error.details['missing_packets'])
                    stale.update(error.details['stale_packets_age_s'])

            parts = [getattr(parser, name, None) for name in POSE_JOINT_PACKETS]
            for name, part in zip(POSE_JOINT_PACKETS, parts):
                read(part, name)
            pose=read(self.robot.get_flange_pose(),'法兰',aggregate=True)
            joints=read(self.robot.get_joint_angles(),'关节',aggregate=True)
            status=read(self.robot.get_arm_status(),'机械臂状态')
            # Check available fault feedback even while another stream is
            # missing; an emergency stop must not spend the receive retry budget.
            if status is not None:
                if int(status.arm_status)!=0 or int(status.err_code)!=0:
                    raise FeedbackError(f'机械臂异常：{status.arm_status}, err={status.err_code}')
                if int(status.ctrl_mode) not in (0,1,3):
                    raise FeedbackError(f'当前控制模式 {status.ctrl_mode}；请退出示教并停止其他控制程序')
                if int(status.teach_status) not in (0,2,6):
                    raise FeedbackError(f'示教/轨迹仍活动：{status.teach_status}')
            for index in range(1,8):
                driver=read(self.robot.get_driver_states(index),f'关节{index}驱动')
                if driver is None:
                    continue
                if not driver.foc_status.driver_enable_status:
                    raise FeedbackError(f'关节{index}未使能，请先在现场使能')
                for key in ('voltage_too_low','motor_overheating','driver_overcurrent',
                            'driver_overheating','collision_status','stall_status','driver_error_status'):
                    if getattr(driver.foc_status,key,False):
                        raise FeedbackError(f'关节{index}驱动异常：{key}')
            message=self.gripper.get_gripper_status()
            grip=read(message,'夹爪')
            if grip is not None:
                if grip.mode!='width' or not grip.foc_status.driver_enable_status:
                    raise FeedbackError('需要已使能、行程(width)模式的AGX夹爪')
                for key in ('voltage_too_low','motor_overheating','driver_overcurrent',
                            'driver_overheating','sensor_status','driver_error_status'):
                    if getattr(grip.foc_status,key,False):
                        raise FeedbackError(f'夹爪异常：{key}')
            if missing or stale:
                raise FeedbackPending(missing, stale)
            if live_clock:
                group = self._pose_group_cache.select(now)
                if group is None:
                    raise FeedbackPending(missing=['coherent_flange_pose'])
                if any(part.feedback_time.clock_epoch != group.epoch for part in parts[3:]):
                    raise FeedbackPending(missing=['coherent_flange_pose'])
                pose = group.pose
                parts[:3] = group.parts
            pose, joints = list(pose), list(joints)
            stamps = [float(part.timestamp) for part in parts]
            # Joint and flange feedback are independent CAN streams (observed
            # at about 200 Hz and 50 Hz). Their receipt-time spread is not a
            # controller fault or a hardware synchronization guarantee. Keep
            # it as evidence, not a second age limit. Every constituent packet
            # must still be fresh above; RobotModel checks measured FK/pose
            # consistency before planning and on each policy observation.
            mapped = [part.feedback_time.mapped_monotonic for part in parts] if live_clock else stamps
            feedback_timing = {
                'packet_span_s': max(mapped)-min(mapped),
                'pose_packet_span_s': max(mapped[:3])-min(mapped[:3]),
                'joint_packet_span_s': max(mapped[3:])-min(mapped[3:]),
                'oldest_packet_age_s': now-min(mapped),
            }
            if live_clock:
                feedback_timing['pose_grouping'] = {
                    'method': 'cached_host_receive_time_group', 'max_span_s': self._pose_group_cache.max_span_s,
                    'span_s': group.span_s, 'age_s': now-min(mapped[:3]),
                    'clock_epoch': group.epoch,
                    'limitation': 'host receipt proximity does not prove simultaneous device sampling'}
            if not np.isfinite(pose+joints+[grip.value,grip.force]).all():
                raise FeedbackError('非有限反馈数据')
            if len(pose)!=6 or len(joints)!=7 or not 0<=float(grip.value)<=.11:
                raise FeedbackError('反馈尺寸/夹爪开口不合理')
            # Preserve the SDK's zeroing bit for diagnosis; it is not an
            # independent check that the physical jaw gap matches this value.
            gripper_status=getattr(grip,'status_code',None)
            gripper_homing=getattr(grip.foc_status,'homing_status',None)
            # CAN 0x2A8 carries signed motor force; this gripper reports
            # negative force while closing. PickCycle consumes force magnitude
            # for contact, holding and the initial loaded-gripper guard.
            # Keep the signed SDK value for diagnosis and never alter the SDK.
            signed_force=float(grip.force)
            state = {'pose':pose,'joints':joints,'width_m':float(grip.value),
                    'force_n':abs(signed_force),'force_signed_n':signed_force,
                    'pose_stamp':min(stamps[:3]),'pose_stamps':stamps[:3],
                    'joint_stamp':min(stamps[3:]),'joint_stamps':stamps[3:],
                    'feedback_timing':feedback_timing,
                    'feedback_time': {name: (part.feedback_time.to_dict()
                        if getattr(part, 'feedback_time', None) is not None else None)
                        for name, part in zip(POSE_JOINT_PACKETS, parts)},
                    'feedback_read_monotonic': time.monotonic(),
                    'gripper_stamp':float(message.timestamp),'ctrl_mode':int(status.ctrl_mode),
                    'gripper_feedback':{'mode':str(grip.mode),
                        'status_code':None if gripper_status is None else int(gripper_status),
                        'homing_status':None if gripper_homing is None else bool(gripper_homing)}}
            if live_clock:
                state.update(pose_stamp_monotonic=min(mapped[:3]),pose_stamps_monotonic=mapped[:3],
                    joint_stamp_monotonic=min(mapped[3:]),joint_stamps_monotonic=mapped[3:],
                    gripper_stamp_monotonic=message.feedback_time.mapped_monotonic)
                state['feedback_time']['gripper'] = message.feedback_time.to_dict()
            if not hasattr(self, '_feedback_clock'):
                state.pop('feedback_time')
            history = getattr(self, '_joint_feedback_history', None)
            if history is None:
                history = self._joint_feedback_history = JointFeedbackHistory(FEEDBACK_MAX_AGE_S)
            epoch = group.epoch if live_clock else None
            offset = getattr(self, 'joint_receive_offset_s', 0.) if live_clock else 0.
            paired, pairing = history.pair(joints, mapped[3:], min(mapped[:3]), epoch=epoch,
                                           joint_receive_offset_s=offset)
            # Retain dense CAN-history alternatives for offline delay analysis.
            # They are never substituted into the FK acceptance check: receipt
            # timestamps alone cannot establish controller sampling latency.
            if live_clock and paired is not None:
                alternatives = []
                for offset in (-.006, -.004, -.002, 0., .002):
                    candidate, evidence = history.at(min(mapped[:3])+offset)
                    alternatives.append({'offset_s': offset, 'joints_rad': candidate,
                                         'pairing': evidence})
                pairing['diagnostic_receive_time_offsets'] = alternatives
            state['feedback_timing']['pose_joint_pairing'] = pairing
            state['feedback_timing']['joint_history_capture_error']=getattr(self,'_joint_history_capture_error',None)
            if live_clock and paired is None:
                raise FeedbackPending(missing=['coherent_flange_pose'])
            if paired is not None:
                state['joints_at_pose_stamp'] = paired
                state['joints_at_pose_stamp_time'] = min(mapped[:3])
                state['joints_at_pose_joint_receive_time'] = pairing['joint_interpolation_stamp']
            return state

    def prepare_motion(self,speed, *, query_limits=False, requested_velocity_limits=None,
                       requested_acceleration_limits=None):
        if not self.execute:
            raise RuntimeError('未指定--execute')
        if not isinstance(speed, int) or not 1 <= speed <= 100:
            raise ValueError('速度百分比须为1–100整数')
        from .single_pick import PickSettings
        cfg = PickSettings(speed_percent=speed)
        requested_velocity = np.asarray(
            (cfg.joint_velocity_limit,) * 7 if requested_velocity_limits is None
            else requested_velocity_limits, dtype=float)
        if (requested_velocity.shape != (7,) or not np.isfinite(requested_velocity).all()
                or np.any(requested_velocity <= 0)):
            raise ValueError('请求的关节速度须为7个有限正数，单位rad/s')
        requested_acceleration = np.asarray(
            (cfg.joint_acceleration_limit,) * 7 if requested_acceleration_limits is None
            else requested_acceleration_limits, dtype=float)
        if (requested_acceleration.shape != (7,) or not np.isfinite(requested_acceleration).all()
                or np.any(requested_acceleration <= 0)):
            raise ValueError('请求的关节加速度须为7个有限正数，单位rad/s²')
        # Request/response configuration reads are optional diagnostics. Some
        # controllers stream normal feedback but do not answer these requests;
        # that does not establish a motion or communication fault. The shared
        # trajectory/controller still enforce the host profile, and firmware
        # settings are never increased or disabled here.
        limits = []
        unavailable = []
        if query_limits:
            for joint in range(1, 8):
                velocity = self.robot.get_joint_angle_vel_limits(joint, timeout=.5, min_interval=0.)
                acceleration = self.robot.get_joint_acc_limits(joint, timeout=.5, min_interval=0.)
                if velocity is None or acceleration is None:
                    unavailable.append({'joint': joint,
                                        'velocity_reply': velocity is not None,
                                        'acceleration_reply': acceleration is not None})
                v = None if velocity is None else float(velocity.msg.max_joint_spd)
                a = None if acceleration is None else float(acceleration.msg.max_joint_acc)
                if ((v is not None and (not np.isfinite(v) or v < requested_velocity[joint-1]))
                        or (a is not None and (not np.isfinite(a) or a < requested_acceleration[joint-1]))):
                    raise FeedbackError(f'关节{joint}固件速度/加速度上限低于请求轨迹或返回值无效，未设置运动速度')
                limits.append({'joint': joint, 'velocity_rad_s': v, 'acceleration_rad_s2': a})
                if unavailable:
                    # One unsupported request should not add seven seconds of
                    # repeated timeouts to a ten-second task cycle.
                    break
        self.robot.set_speed_percent(speed)
        return {'speed_percent': speed, 'queried_joint_limits': limits,
                'joint_command_mode':getattr(self,'_joint_command_mode','j'),
                'joint_command_api':'move_'+getattr(self,'_joint_command_mode','j'),
                'limit_query_requested': bool(query_limits),
                'unavailable_joint_limits': unavailable,
                'firmware_limits_verified': bool(query_limits and len(limits) == 7 and not unavailable),
                'trajectory_limits_source': 'host_profile',
                'requested_joint_velocity_limits_rad_s': requested_velocity.tolist(),
                'requested_joint_acceleration_limits_rad_s2': requested_acceleration.tolist(),
                'firmware_limits_modified': False}

    def grip(self,width,force):
        if not self.execute:
            raise RuntimeError('未指定--execute')
        self.gripper.move_gripper_m(value=float(width),force=float(force))

    def move_joints(self,joints):
        if not self.execute:
            raise RuntimeError('未指定--execute')
        self._send_joint_target(joints)
        self._hold_joint_target=None
        self._hold_joint_stamp=None

    def hold_current_position(self, state=None):
        """Latch measured joints once, then replace the previous motion target.

        ``state`` may be the snapshot just obtained by the control loop. When
        omitted, read normal hardware feedback here, including fault/enable
        status. Continue the selected joint-command API; do not issue
        an emergency stop, reset, enable, speed change, or gripper command.
        The SDK has no separate position-hold primitive. The frozen target is
        repeated through ``move_j`` or ``move_js`` by ``maintain_hold_position``;
        actual braking and hold behavior must be measured on the real arm.
        """
        if not self.execute:
            raise RuntimeError('未指定--execute')
        if state is None:
            state=self.snapshot()
        target=np.asarray(state['joints'],dtype=float)
        if target.shape!=(7,) or not np.isfinite(target).all():
            raise FeedbackError('保持位置需要7个有限的实测关节角')
        # Copy the measured state. Re-latching every cycle would follow drift
        # instead of holding the position captured when pause was requested.
        self._hold_joint_target=target.copy()
        self._hold_joint_stamp=state.get('joint_stamp')
        return self.maintain_hold_position(state)

    def maintain_hold_position(self, state=None):
        """Repeat the frozen measured target while checking normal feedback."""
        if not self.execute:
            raise RuntimeError('未指定--execute')
        if getattr(self,'_hold_joint_target',None) is None:
            raise RuntimeError('尚未捕获保持位置')
        if state is None:
            state=self.snapshot()
        target=self._hold_joint_target.tolist()
        # No RobotModel/path-limit projection here: the arm is holding its
        # already measured pose, including a start outside the nominal model.
        # SDK joint-limit clamping is disabled by its constructor by default;
        # no controller/firmware limit is changed by this method.
        self._send_joint_target(target)
        return {'mode':'joint_position_hold', 'target_joints_rad':target,
                'joint_command_mode':getattr(self,'_joint_command_mode','j'),
                'joint_command_api':'move_'+getattr(self,'_joint_command_mode','j'),
                'target_joint_stamp':self._hold_joint_stamp,
                'observed_joints_rad':list(state['joints']),
                'observed_joint_stamp':state.get('joint_stamp'),
                'gripper_command_changed':False,
                'enable_command_sent':False,
                'firmware_limits_modified':False}

    def stop(self):
        if self.execute:
            scope = getattr(self, '_fault_stop_scope', None)
            if scope is not None:scope.active = True
            try:
                self.robot.electronic_emergency_stop()
            finally:
                if scope is not None:scope.active = False
        self._hold_joint_target=None
        self._hold_joint_stamp=None
