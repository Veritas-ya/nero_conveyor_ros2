"""Recompute moving-target plans using observed planning time, never shift a stale path."""
from dataclasses import replace
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def plan_with_measured_timing(planner, target, state, *, cancelled, planning_fn):
    clock = planner.clock
    started = clock()
    deadline = started + planner.config.raw['planning_budget_s']
    previous_planning = max(float(getattr(planner, 'measured_planning_lead_s', 0.)),
                            float(planner.config.raw.get('measured_planning_initial_s', 0.)))
    planner.measured_planning_lead_s = previous_planning
    post_planning = float(getattr(planner, 'measured_post_planning_s', 0.))
    lead = previous_planning + post_planning
    rounds = []
    # Only this invocation owns the cache: never reuse an earlier exposure or
    # a scene from execution/recovery. No trajectories or IK results are cached.
    scene_cache = {}

    def observe(elapsed, reused_scene_s):
        # Retain failed work too. A cheap bounds rejection must not erase a
        # previous full planning measurement. Cached rounds estimate cold work
        # by adding back the scene preparation they actually skipped.
        cold = elapsed + reused_scene_s
        if not cancelled():
            planner.measured_planning_lead_s = max(planner.measured_planning_lead_s, cold)
        return cold

    def diagnostics():
        return {'mode':'measured', 'rounds':rounds,
                'initial_estimate_s':previous_planning,
                'initial_estimate_source':planner.config.raw.get('measured_planning_initial_source'),
                'estimate_for_next_target_s':planner.measured_planning_lead_s,
                'measured_post_planning_s':post_planning,
                'total_wall_s':clock()-started, 'completed_monotonic_s':clock()}

    for _ in range(4):
        if cancelled() or clock() >= deadline:
            raise PickRejected('测时重规划预算耗尽或任务取消',
                details={'measured_rounds':rounds, 'measured_planning':diagnostics()})
        begin = clock()
        scene_cache['reused_wall_s'] = 0.
        try:
            plan = planning_fn(planner, target, state, cancelled=cancelled,
                reservation_s=lead, measure_late=True, global_deadline=deadline,
                scene_cache=scene_cache)
        except PickRejected as error:
            elapsed = clock()-begin
            reused = scene_cache['reused_wall_s']
            cold = observe(elapsed, reused)
            failures = (error.details or {}).get('failures', [])
            missed = [f for f in failures
                      if f.get('details', {}).get('reason') == 'measured_deadline_missed']
            rounds.append({'reservation_s':lead, 'wall_s':elapsed,
                           'scene_reused_s':reused, 'cold_equivalent_wall_s':cold,
                           'estimate_eligible':not cancelled(),
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
            lead = max(0., planner.measured_planning_lead_s-saved)+post_planning
            lead += float(np.spacing(max(abs(clock()), 1.)))*4
            continue
        elapsed = clock()-begin
        reused = scene_cache['reused_wall_s']
        cold = observe(elapsed, reused)
        rounds.append({'reservation_s':lead, 'wall_s':elapsed, 'result':'delivered',
                       'scene_reused_s':reused, 'cold_equivalent_wall_s':cold,
                       'estimate_eligible':not cancelled()})
        total = clock()-started
        details = diagnostics()
        return replace(plan, path=replace(plan.path, planning_s=total,
                       diagnostics={**plan.path.diagnostics, 'measured_planning':details}),
                       timings_s={**plan.timings_s, 'measured_planning':details})
    raise PickRejected('多次测时重规划仍未在发送时限内交付',
        details={'measured_rounds':rounds, 'measured_planning':diagnostics()})
