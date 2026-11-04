"""Fake SDK/ROS processes share a kernel lease; no CAN interface is accessed."""
import multiprocessing as mp
import os
import uuid
import pytest
from mujoco_conveyor_sorting.runtime.motion_lease import MotionLease, MotionLeaseUnavailable


def competitor(channel, queue, *, crash=False):
    try:
        with MotionLease(channel,owner='ros'):
            queue.put('acquired')
            if crash:
                queue.close();queue.join_thread();os._exit(9)
    except MotionLeaseUnavailable:queue.put('denied')


def launch(channel, *, crash=False):
    ctx=mp.get_context('spawn');queue=ctx.Queue()
    child=ctx.Process(target=competitor,args=(channel,queue),kwargs={'crash':crash});child.start()
    value=queue.get(timeout=5.);child.join(5.)
    assert not child.is_alive()
    queue.close();queue.join_thread()
    return value,child.exitcode


def test_sdk_and_ros_processes_are_mutually_exclusive_and_file_unlink_cannot_steal(tmp_path):
    channel='fake-'+uuid.uuid4().hex
    with MotionLease(channel,owner='sdk'):
        lock=tmp_path/'can.lock';lock.write_text('owner');lock.unlink()
        assert launch(channel)==('denied',0)
        assert launch(channel)==('denied',0)
    assert launch(channel)==('acquired',0)


def test_exception_and_process_crash_release_kernel_ownership():
    channel='fake-'+uuid.uuid4().hex
    with pytest.raises(RuntimeError):
        with MotionLease(channel):raise RuntimeError('original fault')
    assert launch(channel,crash=True)==('acquired',9)
    with MotionLease(channel):pass


def test_read_only_client_does_not_acquire_and_guard_never_sends_without_lease():
    channel='fake-'+uuid.uuid4().hex;lease=MotionLease(channel);sent=[]
    send=lease.guard(lambda packet:sent.append(packet))
    with pytest.raises(MotionLeaseUnavailable):send('not-owned')
    assert sent==[]
    with lease:
        send('owned');assert sent==['owned']
        assert launch(channel)==('denied',0)
    with pytest.raises(MotionLeaseUnavailable):send('released')
    assert sent==['owned']


def test_two_channels_are_independent():
    with MotionLease('fake-'+uuid.uuid4().hex):
        assert launch('fake-'+uuid.uuid4().hex)==('acquired',0)


def test_fork_child_cannot_inherit_send_authority():
    channel='fake-'+uuid.uuid4().hex
    lease=MotionLease(channel).acquire()
    ctx=mp.get_context('fork');queue=ctx.Queue()
    def inherited():
        try:lease.assert_owned();queue.put('inherited')
        except MotionLeaseUnavailable:queue.put('denied')
    child=ctx.Process(target=inherited);child.start()
    assert queue.get(timeout=5.)=='denied';child.join(5.)
    assert child.exitcode==0
    lease.assert_owned();lease.close()
    queue.close();queue.join_thread()


def test_hardware_start_failure_and_disconnect_fault_release_lease():
    from types import SimpleNamespace
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    channel='fake-'+uuid.uuid4().hex
    hw=Hardware.__new__(Hardware);hw._motion_lease=MotionLease(channel)
    def fail():raise RuntimeError('original device fault')
    hw.robot=SimpleNamespace(connect=fail,disconnect=fail)
    with pytest.raises(RuntimeError,match='original device fault'):hw.start()
    with MotionLease(channel):pass
    hw._motion_lease.acquire()
    with pytest.raises(RuntimeError,match='original device fault'):hw.close()
    with MotionLease(channel):pass


def test_real_adapter_wraps_fake_sdk_transport_and_preserves_fault_stop(monkeypatch):
    import sys
    from types import SimpleNamespace
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    sent=[]
    class Parser:
        def parse_packet(self, frame):
            return SimpleNamespace(timestamp=frame.timestamp,msg=None)
    comm=SimpleNamespace(_timeout=.1,send=lambda packet:sent.append(packet))
    gripper=SimpleNamespace(_parser=Parser())
    robot=SimpleNamespace(_parser=Parser(),
        OPTIONS=SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER=1)),
        init_effector=lambda _:gripper,create_comm=lambda:comm,
        connect=lambda:None,disconnect=lambda:None,
        electronic_emergency_stop=lambda:comm.send('original-emergency-stop'))
    monkeypatch.setitem(sys.modules,'pyAgxArm',SimpleNamespace(
        AgxArmFactory=SimpleNamespace(create_arm=lambda _:robot),create_agx_arm_config=lambda **kw:kw))
    hw=Hardware('fake-'+uuid.uuid4().hex,'fake',True)
    with pytest.raises(MotionLeaseUnavailable):comm.send('unowned-motion')
    hw.start();comm.send('owned-motion');hw.close()
    with pytest.raises(MotionLeaseUnavailable):comm.send('released-motion')
    hw.stop()
    assert sent==['owned-motion','original-emergency-stop']
    with pytest.raises(MotionLeaseUnavailable):comm.send('still-unowned')
    decoded=robot._parser.parse_packet(SimpleNamespace(timestamp=1.,arbitration_id=1))
    assert decoded.feedback_time.raw_unix_timestamp==1.
    assert decoded.feedback_time.encoder_sample_time is None
