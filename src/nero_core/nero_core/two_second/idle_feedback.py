"""Drain an idle GC pause using real post-pause feedback, without any send."""
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .streaming import fresh_feedback


def refresh_after_collection(executor, memory, *, epoch):
    # A full collection cannot yield the GIL. Keep it outside all senders,
    # then let receive processing catch up before returning the cycle handoff.
    # Short collections need no extra wait. This is not a feedback age waiver.
    if memory.get('idle_collection_wall_s', 0.) < .02:
        return
    e = executor
    if e.sender is not None:
        raise RuntimeError('回收后刷新反馈时发送器必须已经退出')
    began = e.clock()
    deadline = began+.2
    e.stage = 'idle_feedback_refresh'
    newest = {}
    while e.clock() < deadline:
        e._active(e.owner, e.version, epoch)
        sample = e.device.snapshot_nowait()
        if sample is not None:
            newest = {kind: mapped_stamp(sample, kind) for kind in ('pose', 'joint', 'gripper')}
            if all(stamp is not None and stamp >= began for stamp in newest.values()):
                now = e.clock()
                if now >= deadline:
                    break
                fresh_feedback(sample, now, e.feedback_stamps)
                detail = dict(waited_s=now-began, requires_packets_after_monotonic_s=began,
                              packet_stamps_monotonic_s=newest,
                              maximum_feedback_age_s=max(now-stamp for stamp in newest.values()),
                              robot_commands=0)
                memory['post_collection_feedback'] = detail
                e._emit('idle_feedback_refreshed', detail)
                return
        e.sleep(min(.002, max(0., deadline-e.clock())))
    raise PickRejected('垃圾回收后未及时获得新反馈，未开始下一件运动', details={
        'reason': 'post_collection_feedback_timeout', 'waited_s': e.clock()-began,
        'requires_packets_after_monotonic_s': began, 'packet_stamps_monotonic_s': newest,
        'robot_commands': 0})
