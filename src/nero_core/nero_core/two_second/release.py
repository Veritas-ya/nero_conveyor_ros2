"""Open at the certified rest endpoint while retaining its one joint sender."""
from contextlib import nullcontext
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from .feedback_gates import OpenConfirmation
from .stationary_hold import StationaryHold


def release_at_rest(executor, state, *, epoch, sender_factory):
    e = executor
    # Opening is checked against the still-attached payload before the jaw is
    # commanded. Its release estimate is not a measurement of settled bin contents.
    e.planner.validate_release_opening(state)
    e.planner.release_measured(state)
    # Do not treat intended jaw opening as lost grip. Keep the loaded motion
    # profile until opening is confirmed; only the subsequent return is empty.
    hold_context = (StationaryHold(e, sender_factory, epoch=epoch, carrying=False, payload_motion=True)
                    if e.config.raw.get('trajectory_profile') == 'smooth' else nullcontext())
    with hold_context as hold:
        if hold is not None:
            hold.check()
        with e.command_lock:
            e._active(e.owner, e.version, epoch)
            started = e.clock()
            e.device.grip(e.config.settings.open_width_m, 1.)
            completed = e.clock()
        opening = OpenConfirmation(e.config.settings.open_width_m, not_before=completed)
        e._emit('release_open_requested', {
            'command_started_monotonic_s':started, 'command_completed_monotonic_s':completed,
            'gripper_feedback_before_command_monotonic_s':mapped_stamp(state,'gripper'),
            'stationary_sender_active':hold is not None})
        result = e.holding.wait(lambda x: opening.observe(x['width_m'],mapped_stamp(x,'gripper')),
            anchor=state['pose'], stable=0., heartbeat=None if hold is None else hold.check)
        if hold is not None:
            hold.check()
    e.holding.held_width = None
    return result
