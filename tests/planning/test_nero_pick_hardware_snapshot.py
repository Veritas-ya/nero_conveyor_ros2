"""Snapshot diagnostics with fake feedback only; never construct the SDK adapter."""
import json
from threading import RLock
from types import SimpleNamespace

import pytest

from mujoco_conveyor_sorting.runtime import nero_pick_hardware
from mujoco_conveyor_sorting.runtime.single_pick import PickCycle, PickRejected, PickSettings


def fake_snapshot_adapter(monkeypatch, gripper, *, clock=lambda: 100.):
    monkeypatch.setattr(nero_pick_hardware.time, 'time', clock)

    def message(value):
        return SimpleNamespace(timestamp=nero_pick_hardware.time.time(), msg=value)

    parser = SimpleNamespace(**{
        name: message(None) for name in
        ('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz',
         'joint_12', 'joint_34', 'joint_56', 'joint_7')
    })
    driver = SimpleNamespace(foc_status=SimpleNamespace(driver_enable_status=True))
    status = SimpleNamespace(arm_status=0, err_code=0, ctrl_mode=3, teach_status=0)
    def receive():
        for part in vars(parser).values():
            part.timestamp = clock()
        return False
    # No connect, send, motion, or calibration methods exist on these fakes.
    robot = SimpleNamespace(
        _parser=parser, has_comm_error=receive,
        get_flange_pose=lambda: message([0., 0., .3, 0., 0., 0.]),
        get_joint_angles=lambda: message([0.]*7),
        get_arm_status=lambda: message(status), get_driver_states=lambda _: message(driver))
    adapter = nero_pick_hardware.Hardware.__new__(nero_pick_hardware.Hardware)
    adapter.robot = robot
    adapter.gripper = SimpleNamespace(get_gripper_status=lambda: message(gripper))
    adapter.lock = RLock()
    adapter.channel = 'fake0'
    adapter.feedback_wait_timeout_s = 1.
    return adapter


@pytest.mark.parametrize('homing', [False, True])
def test_snapshot_preserves_gripper_status_without_homing_gate(monkeypatch, homing):
    status_code = 64 | (128 if homing else 0)
    grip = SimpleNamespace(mode='width', value=.0, force=.063,
                           status_code=status_code,
                           foc_status=SimpleNamespace(driver_enable_status=True,
                                                      homing_status=homing))
    snapshot = fake_snapshot_adapter(monkeypatch, grip).snapshot()
    assert snapshot['width_m'] == .0
    assert snapshot['force_n'] == .063
    assert snapshot['gripper_feedback'] == {
        'mode': 'width', 'status_code': status_code, 'homing_status': homing}
    assert json.loads(json.dumps(snapshot))['gripper_feedback'] == snapshot['gripper_feedback']


def test_snapshot_missing_optional_gripper_status_is_unknown(monkeypatch):
    grip = SimpleNamespace(mode='width', value=.04, force=.0,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    snapshot = fake_snapshot_adapter(monkeypatch, grip).snapshot()
    assert snapshot['width_m'] == .04
    assert snapshot['gripper_feedback'] == {
        'mode': 'width', 'status_code': None, 'homing_status': None}


def test_nonblocking_snapshot_copies_current_feedback_without_receive_retry(monkeypatch):
    grip = SimpleNamespace(mode='width', value=.04, force=0.,
        foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip)
    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', lambda _: pytest.fail('must not wait'))
    state = adapter.snapshot_nowait()
    assert state['width_m'] == .04
    assert state['feedback_timing']['read_retry_count'] == 0
    assert state['feedback_timing']['nonblocking_cache_read']


def test_nonblocking_pending_packet_returns_no_sample_without_new_hardstop(monkeypatch):
    grip = SimpleNamespace(mode='width', value=.04, force=0.,
        foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip)
    adapter._snapshot_once = lambda: (_ for _ in ()).throw(nero_pick_hardware.FeedbackPending(['joint_7']))
    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', lambda _: pytest.fail('must not wait'))
    assert adapter.snapshot_nowait() is None


def test_nonblocking_parser_lock_is_never_waited_for(monkeypatch):
    adapter = nero_pick_hardware.Hardware.__new__(nero_pick_hardware.Hardware)
    called = []
    adapter.lock = SimpleNamespace(acquire=lambda **kw: called.append(kw) or False)
    assert adapter.snapshot_nowait() is None
    assert called == [{'blocking': False}]


def test_nonblocking_snapshot_preserves_real_controller_fault(monkeypatch):
    adapter = nero_pick_hardware.Hardware.__new__(nero_pick_hardware.Hardware)
    adapter.lock = RLock()
    adapter._snapshot_once = lambda: (_ for _ in ()).throw(nero_pick_hardware.FeedbackError('真实驱动异常'))
    with pytest.raises(nero_pick_hardware.FeedbackError, match='真实驱动异常'):
        adapter.snapshot_nowait()


@pytest.mark.parametrize('force', [-1.997, 1.997])
def test_signed_sdk_force_supports_contact_and_holding_but_still_detects_loss(monkeypatch, force):
    now = [100.]
    def sleep(dt): now[0] += dt
    grip = SimpleNamespace(mode='width', value=.042364, force=force,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip, clock=lambda: now[0])
    cycle = PickCycle(adapter, None, PickSettings(), lambda *args: None,
                      clock=lambda: now[0], sleep=sleep)
    before = adapter.snapshot()
    held = cycle.wait(lambda s: s['gripper_stamp'] > before['gripper_stamp'] and
                      .003 < s['width_m'] < cycle.cfg.open_width_m-.004 and
                      s['force_n'] >= cycle.cfg.min_force_n,
                      stable=.4, anchor=before['pose'])
    assert held['force_n'] == pytest.approx(1.997)
    assert held['force_signed_n'] == force
    assert now[0] >= 100.4
    cycle.held_width = held['width_m']
    cycle.wait(lambda s: True, stable=.3, holding=True, anchor=held['pose'])
    grip.force = 0.
    with pytest.raises(PickRejected, match='夹持力持续丢失'):
        cycle.wait(lambda s: True, stable=.4, holding=True, anchor=held['pose'])


def test_negative_loaded_gripper_is_rejected_before_initial_open(monkeypatch):
    grip = SimpleNamespace(mode='width', value=.042364, force=-1.997,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip)
    cycle = PickCycle(adapter, None, PickSettings(), lambda *args: None)
    # The fake has no send, grip, motion or preparation methods. The signed
    # force must trigger the loaded guard before any of them could be called.
    with pytest.raises(PickRejected, match='夹爪可能已经夹持物体'):
        cycle.run({'start_joints_rad': [0.]*7, 'stages': []}, execute=True)


@pytest.mark.parametrize('force', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_signed_force_is_rejected(monkeypatch, force):
    grip = SimpleNamespace(mode='width', value=.042364, force=force,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    with pytest.raises(nero_pick_hardware.FeedbackError, match='非有限反馈'):
        fake_snapshot_adapter(monkeypatch, grip).snapshot()


def asynchronous_adapter(monkeypatch):
    grip = SimpleNamespace(mode='width', value=.04, force=0.,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip)
    # Preserve each packet's actual timestamp instead of advancing all streams
    # together, as the older test helper does for grasp-force tests.
    adapter.robot.has_comm_error = lambda: False
    return adapter


@pytest.mark.parametrize('lag', [.045, .080, .150])
def test_fresh_asynchronous_streams_are_accepted_with_original_timestamps(monkeypatch, lag):
    adapter = asynchronous_adapter(monkeypatch)
    parser = adapter.robot._parser
    pose_stamps = [100.-lag, 100.-lag+.001, 100.-lag+.002]
    for name, stamp in zip(('end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz'), pose_stamps):
        getattr(parser, name).timestamp = stamp
    parser.joint_12.timestamp = 99.999

    state = adapter.snapshot()
    assert state['pose_stamps'] == pose_stamps
    assert state['pose_stamp'] == min(pose_stamps)
    assert state['joint_stamp'] == 99.999
    assert state['feedback_timing']['packet_span_s'] == pytest.approx(lag)
    assert state['feedback_timing']['oldest_packet_age_s'] == pytest.approx(lag)
    assert state['feedback_timing']['joint_packet_span_s'] == pytest.approx(.001)
    assert state['feedback_timing']['pose_packet_span_s'] == pytest.approx(.002)
    assert json.loads(json.dumps(state)) == state


@pytest.mark.parametrize('part', ['end_pose_xy', 'end_pose_zrx', 'end_pose_ryrz',
                                  'joint_12', 'joint_34', 'joint_56', 'joint_7'])
@pytest.mark.parametrize('stamp', [99.799, 100.001, float('nan')])
def test_each_constituent_packet_must_still_be_fresh(monkeypatch, part, stamp):
    adapter = asynchronous_adapter(monkeypatch)
    getattr(adapter.robot._parser, part).timestamp = stamp
    # Aggregated SDK messages report "now" in this fake. They must not hide
    # an old/malformed constituent, including joints absent from their stamp.
    with pytest.raises(nero_pick_hardware.FeedbackError, match=part):
        adapter.snapshot(wait_timeout_s=0.)


def test_missing_packet_is_not_replaced_by_sdk_aggregate(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    del adapter.robot._parser.joint_34
    with pytest.raises(nero_pick_hardware.FeedbackError, match='joint_34'):
        adapter.snapshot(wait_timeout_s=0.)


def test_asynchronous_feedback_does_not_mask_emergency_stop(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    adapter.robot._parser.end_pose_xy.timestamp = 99.94
    adapter.robot.get_arm_status().msg.arm_status = 1
    with pytest.raises(nero_pick_hardware.FeedbackError, match='机械臂异常'):
        adapter.snapshot()


def test_asynchronous_feedback_still_requires_consistent_measured_pose(monkeypatch):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel

    adapter = asynchronous_adapter(monkeypatch)
    adapter.robot._parser.end_pose_xy.timestamp = 99.94
    adapter.robot.get_joint_angles = lambda: SimpleNamespace(
        timestamp=100., msg=[.1, .2, -.1, 1., .2, .3, .1])
    geometry = RobotModel()
    state = adapter.snapshot()
    geometry.set_state(state['joints'], state['width_m'])
    flange = geometry.flange()
    pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()
    adapter.robot.get_flange_pose = lambda: SimpleNamespace(timestamp=100., msg=pose)
    geometry.check_feedback(adapter.snapshot())
    pose[0] += .02
    with pytest.raises(PickRejected, match='关节FK与法兰反馈不符'):
        geometry.check_feedback(adapter.snapshot())


def receive_clock(monkeypatch):
    now = [100.]
    monkeypatch.setattr(nero_pick_hardware.time, 'time', lambda: now[0])
    monkeypatch.setattr(nero_pick_hardware.time, 'monotonic', lambda: now[0])
    return now


@pytest.mark.parametrize('stream', ['end_pose_xy', 'joint_7', 'driver', 'gripper'])
def test_delayed_feedback_recovers_without_reusing_old_data(monkeypatch, stream):
    adapter = asynchronous_adapter(monkeypatch)
    now = receive_clock(monkeypatch)
    parser = adapter.robot._parser
    if stream in ('driver', 'gripper'):
        owner = adapter.robot if stream == 'driver' else adapter.gripper
        name = 'get_driver_states' if stream == 'driver' else 'get_gripper_status'
        original = getattr(owner, name)
        delayed = original(1) if stream == 'driver' else original()
        delayed.timestamp = 99.75
        setattr(owner, name, lambda *_args: delayed)
    else:
        delayed = getattr(parser, stream)
        delayed.timestamp = 99.75

    def receive(dt):
        # Acquire from a different thread: a retry that holds the parser lock
        # would prevent the real receive thread from delivering the next packet.
        from concurrent.futures import ThreadPoolExecutor
        def update():
            assert adapter.lock.acquire(timeout=.2), 'receive retry held the parser lock'
            try:
                now[0] += dt
                for part in vars(parser).values():
                    if part is not delayed:
                        part.timestamp = now[0]
                if now[0] >= 100.3:
                    delayed.timestamp = now[0]
            finally:
                adapter.lock.release()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(update).result(timeout=1.)

    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', receive)
    state = adapter.snapshot()
    assert state['feedback_timing']['read_wait_s'] >= .3
    assert state['feedback_timing']['read_retry_count'] > 0
    assert state['feedback_timing']['oldest_packet_age_s'] == 0.
    if stream == 'end_pose_xy':
        assert state['pose_stamps'][0] == now[0]
    if stream == 'joint_7':
        assert state['joint_stamps'][-1] == now[0]


def test_missing_packet_arrives_during_startup(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    now = receive_clock(monkeypatch)
    adapter.robot._parser.joint_7 = None

    def receive(dt):
        now[0] += dt
        for name in nero_pick_hardware.POSE_JOINT_PACKETS:
            if name != 'joint_7' or now[0] >= 102.:
                setattr(adapter.robot._parser, name, SimpleNamespace(timestamp=now[0], msg=None))

    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', receive)
    state = adapter.snapshot(wait_timeout_s=3.)
    assert state['joint_stamps'][-1] >= 102.
    assert state['feedback_timing']['read_wait_s'] >= 2.


def test_silent_can_reports_all_missing_packets_and_actual_wait(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    now = receive_clock(monkeypatch)
    adapter.robot._parser = SimpleNamespace()
    for name in ('get_flange_pose', 'get_joint_angles', 'get_arm_status', 'get_driver_states'):
        setattr(adapter.robot, name, lambda *_args: None)
    adapter.gripper.get_gripper_status = lambda: None
    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', lambda dt: now.__setitem__(0, now[0]+dt))
    with pytest.raises(nero_pick_hardware.FeedbackError, match='未收到任何机械臂关节/法兰反馈') as caught:
        adapter.snapshot(wait_timeout_s=3.)
    details = caught.value.details
    assert details['waited_s'] == pytest.approx(3.)
    assert details['channel'] == 'fake0'
    assert set(nero_pick_hardware.POSE_JOINT_PACKETS).issubset(details['missing_packets'])
    assert details['stale_packets_age_s'] == {}
    assert '200 ms' not in str(caught.value)
    json.dumps(details, allow_nan=False)


def test_one_dead_stream_does_not_hide_behind_live_aggregates(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    now = receive_clock(monkeypatch)
    parser = adapter.robot._parser
    parser.joint_7.timestamp = 99.75

    def receive(dt):
        now[0] += dt
        for name, part in vars(parser).items():
            if name != 'joint_7':
                part.timestamp = now[0]

    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', receive)
    with pytest.raises(nero_pick_hardware.FeedbackError, match='反馈未更新：joint_7') as caught:
        adapter.snapshot()
    assert caught.value.details['waited_s'] == pytest.approx(1.)
    assert caught.value.details['stale_packets_age_s'] == {'joint_7': pytest.approx(1.25)}
    assert caught.value.details['missing_packets'] == []


@pytest.mark.parametrize('fault', ['emergency', 'driver', 'gripper', 'communication'])
def test_actual_fault_during_missing_pose_is_not_retried(monkeypatch, fault):
    adapter = asynchronous_adapter(monkeypatch)
    adapter.robot._parser.end_pose_xy = None
    if fault == 'emergency':
        adapter.robot.get_arm_status().msg.arm_status = 1
    elif fault == 'driver':
        adapter.robot.get_driver_states(1).msg.foc_status.driver_error_status = True
    elif fault == 'gripper':
        adapter.gripper.get_gripper_status().msg.foc_status.driver_overcurrent = True
    else:
        adapter.robot.has_comm_error = lambda: True
        adapter.robot.get_comm_error = lambda: 'CAN read failed'
    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', lambda _: pytest.fail('fault must not wait'))
    with pytest.raises(nero_pick_hardware.FeedbackError, match='异常'):
        adapter.snapshot()


def test_emergency_received_during_retry_is_reported_immediately(monkeypatch):
    adapter = asynchronous_adapter(monkeypatch)
    now = receive_clock(monkeypatch)
    adapter.robot._parser.end_pose_xy = None

    def receive(dt):
        now[0] += dt
        adapter.robot.get_arm_status().msg.arm_status = 1

    monkeypatch.setattr(nero_pick_hardware.time, 'sleep', receive)
    with pytest.raises(nero_pick_hardware.FeedbackError, match='机械臂异常'):
        adapter.snapshot()
    assert now[0] == pytest.approx(100.02)


@pytest.mark.parametrize('offset', [0., -.002])
def test_live_snapshot_requires_coherent_group_and_bracketed_joint_feedback(monkeypatch, offset):
    from mujoco_conveyor_sorting.runtime.coherent_flange import CoherentFlangeCache
    grip = SimpleNamespace(mode='width', value=.04, force=0.,
                           foc_status=SimpleNamespace(driver_enable_status=True))
    adapter = fake_snapshot_adapter(monkeypatch, grip)
    adapter.joint_receive_offset_s = offset
    adapter.robot.has_comm_error = lambda: False
    monkeypatch.setattr(nero_pick_hardware.time, 'monotonic', lambda: 100.)
    def timing(stamp, epoch=0):
        d = {'mapped_monotonic': stamp, 'clock_epoch': epoch, 'mapping_quality': 'mapped'}
        return SimpleNamespace(**d, to_dict=lambda: d.copy())
    parser = adapter.robot._parser
    for i, name in enumerate(nero_pick_hardware.POSE_JOINT_PACKETS):
        part = getattr(parser, name)
        stamp = 99.950+i*.001 if i < 3 else 99.940
        part.timestamp = stamp; part.feedback_time = timing(stamp)
    for obj, name in ((adapter.robot, 'get_arm_status'), (adapter.robot, 'get_driver_states'),
                      (adapter.gripper, 'get_gripper_status')):
        original = getattr(obj, name)
        def wrapped(*args, _original=original):
            message = _original(*args); message.feedback_time = timing(99.99); return message
        setattr(obj, name, wrapped)
    adapter._feedback_clock = object()
    adapter._pose_group_cache = CoherentFlangeCache()
    adapter._pose_group_cache.observe_epoch(0)
    # An absent complete group must not fall back to the SDK aggregate.
    with pytest.raises(nero_pick_hardware.FeedbackPending, match='coherent_flange_pose'):
        adapter._snapshot_once()
    assert adapter._pose_group_cache.capture(parser, adapter.robot.get_flange_pose, 100.)
    # The first snapshot adds real joint samples but does not extrapolate.
    with pytest.raises(nero_pick_hardware.FeedbackPending, match='coherent_flange_pose'):
        adapter._snapshot_once()
    for name in nero_pick_hardware.POSE_JOINT_PACKETS[3:]:
        part = getattr(parser, name); part.timestamp = 99.970; part.feedback_time = timing(99.970)
    state = adapter._snapshot_once()
    assert state['feedback_timing']['pose_grouping']['span_s'] == pytest.approx(.002)
    assert state['feedback_timing']['pose_joint_pairing']['available']
    assert state['joints_at_pose_stamp_time'] == pytest.approx(99.95)
    assert state['joints_at_pose_joint_receive_time'] == pytest.approx(99.95+offset)
    assert state['feedback_timing']['pose_joint_pairing']['pose_stamp'] == pytest.approx(99.95)
    assert state['feedback_timing']['pose_joint_pairing']['joint_receive_offset_s'] == offset
    # A new clock epoch cannot reuse a previously valid pose group.
    adapter._pose_group_cache.observe_epoch(1)
    with pytest.raises(nero_pick_hardware.FeedbackPending, match='coherent_flange_pose'):
        adapter._snapshot_once()
    # A real device fault must not wait for the coherent pose cache.
    adapter.robot.get_arm_status().msg.arm_status = 1
    with pytest.raises(nero_pick_hardware.FeedbackError, match='机械臂异常'):
        adapter._snapshot_once()
