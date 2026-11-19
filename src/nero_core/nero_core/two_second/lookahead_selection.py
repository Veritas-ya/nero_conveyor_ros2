"""Order frozen observations by their last feasible dispatch, without IK or IO."""
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def latest_dispatch(target, config):
    speed = float(target.velocity @ config.direction)
    if speed <= 0:
        return float('inf')
    raw = config.raw
    duration = min(raw.get('synchronized_join_candidates_s',
                           [raw.get('prealign_orientation_s', 0.)]))
    overhead = (duration + raw.get('rotate_first_s', 0.) +
        raw.get('synchronized_duration_s', 0.) +
        getattr(config.settings, 'full_plan_close_window_s', 0.) + raw['command_advance_s'])
    positions = [float(g[:3, 3] @ config.direction) for g in target.grasps]
    # An optimistic last dispatch across allowed grasps, used for ordering
    # only. Production scheduling, IK and certification decide feasibility.
    return target.capture_s+(config.follow_bounds[1]-min(positions))/speed-overhead


def pending_error(vision, target, config, now):
    status = getattr(vision, 'pending_status', None)
    if status is not None:
        reason = status(target, now)
    elif target.epoch != vision.clock_epoch or now < target.capture_s:
        reason = 'clock_invalidated'
    elif float((target.entry_position+target.velocity*(now-target.capture_s)) @ config.direction) > config.follow_bounds[1]:
        reason = 'past_follow_region'
    else:
        reason = None if vision.poll_pending() is target else 'exposure_replaced'
    if reason is None:
        return None
    code, message = {
        'clock_invalidated': ('lookahead_clock_invalidated', '预取曝光的相机时钟已重置'),
        'past_follow_region': ('lookahead_target_past_region', '预取工件已越过抓取区，等待后续工件'),
        'consumed_or_cancelled': ('lookahead_target_cancelled', '预取目标已消费或取消'),
        'exposure_replaced': ('lookahead_exposure_replaced', '预取曝光已被替换或作废'),
    }[reason]
    return PickRejected(message, details={'reason': code,
        'identity': target.identity, 'capture_monotonic_s': target.capture_s,
        'observation_age_s': now-target.capture_s,
        'coordinate_m': float((target.entry_position+target.velocity*(now-target.capture_s)) @ config.direction)})
