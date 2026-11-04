"""Recompute moving-target plans using observed planning time, never shift a stale path."""
from dataclasses import replace
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def plan_with_measured_timing(planner, target, state, *, cancelled, planning_fn):
    clock = planner.clock
    started = clock()
    deadline = started + planner.config.raw['planning_budget_s']
    previous_planning = float(getattr(planner, 'measured_planning_lead_s', 0.))
    post_planning = float(getattr(planner, 'measured_post_planning_s', 0.))
    lead = previous_planning + post_planning
    rounds = []
    observed_candidates = []
    for _ in range(4):
        if cancelled() or clock() >= deadline:
            raise PickRejected('测时重规划预算耗尽或任务取消', details={'rounds':rounds})
        begin = clock()
        try:
            plan = planning_fn(planner, target, state, cancelled=cancelled,
                reservation_s=lead, measure_late=True, global_deadline=deadline)
        except PickRejected as error:
            failures = (error.details or {}).get('failures', [])
            missed = [f for f in failures
                      if f.get('details', {}).get('reason') == 'measured_deadline_missed']
            rounds.append({'reservation_s':lead, 'wall_s':clock()-begin,
                           'result':'missed' if missed else 'rejected', 'failures':failures})
            if not missed or cancelled() or clock() >= deadline:
                raise PickRejected(str(error), details={**(error.details or {}), 'measured_rounds':rounds}) from error
            values = [float(f['details']['measured_candidate_wall_s']) for f in missed]
            observed_candidates.extend(values)
            # Use actual completed candidates. An ulp avoids an exactly zero
            # lead on clocks with finite resolution; it is not a time budget.
            lead = max(lead, max(values)+post_planning) + float(np.spacing(max(abs(clock()), 1.)))*4
            continue
        elapsed = clock()-begin
        rounds.append({'reservation_s':lead, 'wall_s':elapsed, 'result':'delivered'})
        planner.measured_planning_lead_s = max(previous_planning, elapsed, *observed_candidates)
        total = clock()-started
        details = {'mode':'measured', 'rounds':rounds,
                   'estimate_for_next_target_s':planner.measured_planning_lead_s,
                   'measured_post_planning_s':post_planning,
                   'total_wall_s':total, 'completed_monotonic_s':clock()}
        return replace(plan, path=replace(plan.path, planning_s=total,
                       diagnostics={**plan.path.diagnostics, 'measured_planning':details}),
                       timings_s={**plan.timings_s, 'measured_planning':details})
    raise PickRejected('多次测时重规划仍未在发送时限内交付', details={'measured_rounds':rounds})
