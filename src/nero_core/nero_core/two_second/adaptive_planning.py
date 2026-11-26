"""Recompute moving-target plans using observed planning time, never shift a stale path."""
from dataclasses import replace
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .planning_time import history_for


def plan_with_measured_timing(planner, target, state, *, cancelled, planning_fn, global_deadline=None):
    clock = planner.clock
    started = clock()
    deadline = started + planner.config.raw['planning_budget_s']
    if global_deadline is not None:
        deadline = min(deadline, global_deadline)
    history = history_for(planner)
    previous_planning = (history.estimate_s if history is not None else
        max(float(getattr(planner, 'measured_planning_lead_s', 0.)),
                            float(planner.config.raw.get('measured_planning_initial_s', 0.)))
        )
    planner.measured_planning_lead_s = previous_planning
    post_planning = float(getattr(planner, 'measured_post_planning_s', 0.))
    lead = previous_planning + post_planning
    rounds = []
    # Only this invocation owns the cache: never reuse an earlier exposure or
    # a scene from execution/recovery. No trajectories or IK results are cached.
    scene_cache = {}

    def observe(elapsed, reused_scene_s, *, outcome, completed=False, components=None):
        # Only completed rounds train the rolling estimator. Legacy maximum
        # mode retains failed work. Add back scene work skipped by this cache.
        cold = elapsed + reused_scene_s
        eligible = not cancelled()
        if history is not None:
            eligible = history.observe(cold, outcome=outcome, completed=completed,
                cancelled=not eligible, components=components)
            planner.measured_planning_lead_s = history.estimate_s
        elif eligible:
            planner.measured_planning_lead_s = max(planner.measured_planning_lead_s, cold)
        return cold, eligible

    def diagnostics():
        return {'mode':'measured', 'rounds':rounds,
                'initial_estimate_s':previous_planning,
                'initial_estimate_source':planner.config.raw.get('measured_planning_initial_source'),
                'estimate_for_next_target_s':planner.measured_planning_lead_s,
                'measured_post_planning_s':post_planning,
                'estimator':history.diagnostics() if history is not None else {'policy':'maximum'},
                'total_wall_s':clock()-started, 'completed_monotonic_s':clock()}

    for _ in range(4):
        if cancelled() or clock() >= deadline:
            stopped = cancelled()
            raise PickRejected('测时重规划已取消' if stopped else '测时重规划总预算耗尽',
                details={'reason':'planning_cancelled' if stopped else 'planning_budget_exhausted',
                         'measured_rounds':rounds, 'measured_planning':diagnostics()})
        begin = clock()
        scene_cache['reused_wall_s'] = 0.
        try:
            plan = planning_fn(planner, target, state, cancelled=cancelled,
                reservation_s=lead, measure_late=True, global_deadline=deadline,
                scene_cache=scene_cache)
        except PickRejected as error:
            elapsed = clock()-begin
            reused = scene_cache['reused_wall_s']
            failures = (error.details or {}).get('failures', [])
            missed = [f for f in failures
                      if f.get('details', {}).get('reason') == 'measured_deadline_missed']
            components = {}
            for failure in failures:
                for key, value in failure.get('phase_timings_s', {}).items():
                    components[key] = components.get(key, 0.)+value
            cold, eligible = observe(elapsed, reused, outcome='missed' if missed else 'rejected',
                completed=bool(missed), components=components)
            rounds.append({'reservation_s':lead, 'wall_s':elapsed,
                           'scene_reused_s':reused, 'cold_equivalent_wall_s':cold,
                           'estimate_eligible':eligible,
                           'result':'missed' if missed else 'rejected', 'failures':failures})
            if not missed or cancelled() or clock() >= deadline:
                raise PickRejected(str(error), details={**(error.details or {}),
                    'measured_rounds':rounds, 'measured_planning':diagnostics()}) from error
            # The next round includes selection/ordering overhead, but need not
            # budget a scene preparation which this invocation can reuse.
            saved = scene_cache.get('prepare_wall_s', 0.)
            current_key = (id(target), id(state), id(planner.config), id(getattr(planner, 'model', None)),
                           getattr(planner, 'version', None), getattr(planner, 'scene_revision', None))
            if scene_cache.get('key') != current_key:
                saved = 0.
            # This same target has just demonstrated its cost. A population
            # percentile must not under-reserve its immediate retry.
            retry_cost = max(cold, planner.measured_planning_lead_s)
            lead = max(0., retry_cost-saved)+post_planning
            lead += float(np.spacing(max(abs(clock()), 1.)))*4
            continue
        elapsed = clock()-begin
        reused = scene_cache['reused_wall_s']
        outcome = ('expired_candidate_for_repair' if
            plan.path.diagnostics.get('expired_delivery_candidate') else 'delivered')
        cold, eligible = observe(elapsed, reused, outcome=outcome, completed=True,
            components={k:v for k,v in plan.timings_s.items() if isinstance(v, (int, float))
                        and k != 'full_planning_delivery'})
        rounds.append({'reservation_s':lead, 'wall_s':elapsed, 'result':outcome,
                       'scene_reused_s':reused, 'cold_equivalent_wall_s':cold,
                       'estimate_eligible':eligible})
        total = clock()-started
        details = diagnostics()
        return replace(plan, path=replace(plan.path, planning_s=total,
                       diagnostics={**plan.path.diagnostics, 'measured_planning':details}),
                       timings_s={**plan.timings_s, 'measured_planning':details})
    raise PickRejected('多次测时重规划仍未在发送时限内交付',
        details={'measured_rounds':rounds, 'measured_planning':diagnostics()})
