"""Candidate retry support for the command-free measured-vision diagnostic.

This does not own an executor and must not be used to switch an executing arm.
"""
from nero_core.measured_motion import MeasuredMotionSession

RETRYABLE_GEOMETRY_REJECTIONS = frozenset({
    'full pick approach failed complete geometry/TCP checks',
    'full pick closing failed complete geometry/TCP checks',
    'full pick lift failed complete geometry/TCP checks',
    # The worker exhausted its candidate/IK search without producing a
    # certificate. Cancellation is reported separately by the planner.
    'no completely certified full pick',
})


def evaluated_variants(accepted, rejected):
    """Return only evaluated pose/variant pairs, never budget-skipped pairs."""
    return {(int(row['source_index']), row['variant']) for row in [*accepted, *rejected]
            if 'source_index' in row and 'variant' in row and
            row.get('evaluated') is not False and row.get('reason') not in
            ('not_evaluated_budget', 'not_evaluated_after_selection')}


def rebuild_after_rejection(context, session, identity, captured, recent, now, reason, *, allow_late_delivery=False,
                            allow_budget_retry=False):
    """Close the old shadow planner and replay actual observations into a new session.

    A new session has no bound grasp and cannot accept results from the closed
    planner. The caller must rerun result admission against the latest target,
    bind the next candidate, and perform full scene/trajectory certification.
    """
    if reason not in RETRYABLE_GEOMETRY_REJECTIONS and not (
            allow_late_delivery and reason == 'full_pick_delivery_missed_start') and not (
            allow_budget_retry and reason == 'planning_candidate_budget_exceeded'):
        raise ValueError('rejection is not eligible for candidate retry')
    latest = session.snapshot(now)
    if (latest is None or not latest.valid or latest.target_identity != identity or
            latest.velocity_source != 'nero7_xy_kalman'):
        raise ValueError('target is cancelled, changed, or measurement expired')
    context.close()
    session.cancel()
    replacement = MeasuredMotionSession(identity, **session.tracker_options)
    def apply(item):
        return replacement.update(item.detection, identity=identity, timestamp=item.capture_s,
            sequence=item.sequence, clock_epoch=item.clock_epoch, source_timestamp=item.capture_s,
            velocity_measurement=item.velocity_measurement)
    if not apply(captured):
        raise ValueError('network capture no longer accepted')
    replacement.pin_network_capture(identity=identity,timestamp=captured.capture_s,sequence=captured.sequence)
    for item in recent:
        if item.sequence > captured.sequence and not apply(item):
            raise ValueError('replayed observation rejected')
    return replacement


def conveyor_world_x_axis_in_base(world_to_base):
    """Express world-X conveyor coordinates in base without changing bounds."""
    import numpy as np
    transform=np.asarray(world_to_base,dtype=float)
    if (transform.shape != (4,4) or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0,0,0,1])
        or not np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-6)):
        raise ValueError('world-to-base must be a rigid transform')
    return transform[:3,0].copy(), transform[:3,3].copy()


def remaining_window_check(observation, target_id, policy, *, budget_override=None):
    """Reject only if even the existing minimum schedule cannot fit.

    Times in the observation, budget, and policy are in the same policy clock.
    This necessary-condition check does not prove a path can be delivered.
    """
    import math
    predictions=observation.get('predictions',{})
    prediction=predictions.get(target_id, predictions.get(str(target_id),{}))
    window=prediction.get('grasp_window',{})
    now=float(observation['timestamp'])
    end=float(window.get('end_time',float('nan')))
    config=policy.config
    budget=float(config.full_plan_compute_budget_s if budget_override is None else budget_override)
    dt=float(policy._execution_dt(observation))
    required=budget+float(config.trajectory_minimum_duration)+dt+float(config.descent_lead_time)+float(config.intercept_deadline_margin)
    if not all(math.isfinite(value) for value in (now,end,budget,dt,required)) or budget<=0:
        return {'available':False,'reason':'invalid_or_missing_window_timing'}
    remaining=end-now
    return {'available':bool(window.get('feasible',False) and remaining+1e-9>=required),
        'reason':('ok' if window.get('feasible',False) and remaining+1e-9>=required
                  else 'insufficient_time_for_minimum_certified_schedule'),
        'now_s':now,'window_end_s':end,'remaining_s':remaining,'minimum_required_s':required,
        'compute_reserve_s':budget,'duration_minimum_s':float(config.descent_lead_time),
        'delivery_lead_s':float(config.trajectory_minimum_duration)+dt,
        'deadline_margin_s':float(config.intercept_deadline_margin)}
