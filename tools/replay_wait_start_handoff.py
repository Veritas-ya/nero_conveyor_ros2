#!/usr/bin/env python3
"""Audit recorded rejection causes and replay the archived exposure without devices.

Only the last preview preserves a complete frozen target. Other cycles are
log audits, not reconstructed collision certificates. Fresh timestamps in the
handoff exercise are simulated and never represented as new robot feedback.
"""
import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import Event
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/nero_core'))
from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.paper_replay import restore_input, clone_input
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.return_lookahead import ReturnLookahead, PreparedIntercept
from nero_core.two_second.phase_timing import verify_plan
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.trajectory import extrema
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.suffix == '.jsonl' else json.loads(path.read_text())


def failure_summary(details):
    rows = details.get('phase_timing_search', {}).get('rows', [])
    candidates = []
    for f in details.get('failures', []):
        d = f.get('details') or {}
        candidates.append({k: f.get(k) for k in ('reason', 'failed_phase', 'ik_knot_hz')}
                          | {'details': d})
    return {'reason': details.get('reason'), 'candidates': candidates,
        'shortest_brake_bounds': [r for r in rows if r.get('brake_s') == .4],
        'measured_planning': details.get('measured_planning')}


def audit(run, cfg):
    events = read_events(run/'hardware/events.jsonl')
    cycles = []
    for path in sorted((run/'hardware').glob('cycle_*/events.json')):
        records = read_events(path)
        submitted = next(r['data'] for r in records if r['event'] == 'planning_submitted')
        cancelled = next((r['data'] for r in records if r['event'] == 'attempt_cancelled'), {})
        delta = np.asarray(submitted['start_joints_rad'])-cfg.standby
        cycles.append({'cycle': path.parent.name, 'identity': submitted['identity'],
            'submission': submitted,
            'nominal_start_error_deg': float(np.rad2deg(np.max(abs(delta)))),
            'nominal_start_gate_passed': bool(np.max(abs(delta)) <= .005),
            'formally_delivered': any(r['event'] == 'planning_delivery' for r in records),
            'start_mismatch_replan': any(r['event'] == 'lookahead_start_replan' for r in records),
            'error': cancelled.get('error'), 'failed_stage': cancelled.get('failed_stage'),
            'failure': failure_summary(cancelled.get('details') or {})})
    rejections = [{'identity': e['data']['identity'], 'reason': e['data']['reason'],
        'failure': failure_summary(e['data'].get('failure_details') or {})}
        for e in events if e['event'] == 'lookahead_rejected']
    return events, cycles, rejections


class ReplayVision:
    def __init__(self, target): self.target = target; self.clock_epoch = target.epoch
    def poll_pending(self): return self.target
    def discard_pending(self, identity):
        if self.target is not None and self.target.identity == identity: self.target = None


def replay(cfg, target, state, *, start_s, lead_s, post_s, use_actual, via_worker):
    cfg = deepcopy(cfg)
    cfg.raw['velocity_source'] = target.speed_report['source']
    cfg.raw['fixed_belt_speed_m_s'] = target.speed_report['configured_speed_m_s']
    model = calibrated_model(cfg)
    previous = SimpleNamespace(intercept_not_before_s=None)
    holder = [None]
    clock = lambda: start_s if holder[0] is None else start_s+time.monotonic()-holder[0]
    planner = MeasuredPlanner(cfg, model=model, clock=clock)
    planner.measured_planning_lead_s = lead_s
    planner.measured_post_planning_s = post_s
    target = clone_input(target)
    actual = {**deepcopy(state), 'force_n': 0.}
    chosen = actual if use_actual else nominal_state(model, cfg.standby, cfg.settings.open_width_m)
    def snapshot():
        now = clock()
        return {**deepcopy(actual), 'joint_stamp': now, 'pose_stamp': now, 'gripper_stamp': now}
    emitted = []
    worker = ReturnLookahead(cfg, ReplayVision(target), planner, cancelled=Event(),
        clock=clock, emit=lambda name, **data: emitted.append({'event':name, 'data':data}))
    row = {'start_s': start_s, 'lead_s': lead_s, 'post_s':post_s, 'use_actual': use_actual,
        'via_actual_wait_worker': via_worker, 'feedback_refresh_is_simulated': True}
    worker.begin_cycle((target.epoch, -1, -1))
    worker.confirm_return(snapshot())
    holder[0] = time.monotonic()
    try:
        if via_worker:
            worker._step()
            item = worker.ready
            if item is None:
                rejection = next((e['data'] for e in reversed(emitted)
                    if e['event'] in ('lookahead_rejected', 'lookahead_planning_deferred')), {})
                row.update(status='planning_rejected', failure=rejection, events=emitted)
                return row
            plan = item.plan
        else:
            planner.intercept_not_before_s = start_s+cfg.raw['lookahead_return_margin_s']
            plan = planner.intercept(target, chosen)
            item = PreparedIntercept(target, planner, plan, chosen, worker.owner, clock(), start_s)
        phase = verify_plan(cfg, plan.path, plan.reference)
        row.update(status='full_plan_ready', planning_wall_s=clock()-start_s,
            remaining_send_lead_s=plan.reference.start_s-cfg.raw['command_advance_s']-clock(),
            start_error_rad=float(np.max(abs(np.asarray(actual['joints'])-plan.path.q0))),
            phase_timing=phase, timings_s=plan.timings_s,
            peak_velocity_rad_s=np.max(abs(np.asarray(extrema(plan.path.curve, 1))),axis=0),
            peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(plan.path.curve, 2))),axis=0))
        row['executor_start_gate_passed'] = row['start_error_rad'] <= .005
        if via_worker:
            # Emulate continuing fresh, stationary feedback while the planner runs.
            # This exercises the real poll/claim logic, not a hardware sender.
            worker.update_wait_state(snapshot())
            ready = worker.poll_ready()
            row['worker_poll_ready_passed'] = ready is item
            if ready is not item:
                raise PickRejected('offline worker did not expose ready plan')
            claimed = worker.claim(item, previous)
            row['worker_claim_passed'] = claimed is item
        item.validate(cfg, target.epoch, clock())
        row['identity_version_deadline_passed'] = True
        row['events'] = emitted
    except PickRejected as error:
        row.update(status='rejected', error=str(error), details=error.details)
    finally:
        row['elapsed_s'] = clock()-start_s
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.repeats < 1: p.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    run = args.run.resolve(strict=True)
    cfg = CellConfig.load(run/'effective.yaml')
    cfg.raw['taught_segments'] = {kind: TaughtSegment.load(cfg.raw['taught_'+kind+'_file'], cfg, kind)
        for kind in ('transfer','return')}
    events, cycles, rejected = audit(run, cfg)
    target, state, info = restore_input(run/'hardware', cfg)
    match = next(c for c in cycles if c['identity'] == list(target.identity))
    started = next(e for e in events if e['event'] == 'lookahead_planning_started'
        and e['data']['identity'] == list(target.identity))
    # The most recent failed attempt on this spare planner records its retained
    # time estimate; leave conservative history intact in the counterfactual.
    preceding = [e for e in events if e['event'] == 'lookahead_rejected'
        and e['monotonic_s'] < started['monotonic_s']]
    history = preceding[-1]['data']['failure_details']['measured_planning']
    lead = history['estimate_for_next_target_s']
    post = history['measured_post_planning_s']
    begin = started['monotonic_s']
    late_lead = match['failure']['measured_planning']['initial_estimate_s']
    late_post = match['failure']['measured_planning']['measured_post_planning_s']
    report = {'run': str(run), 'hardware_connections':0, 'commands_sent':0,
        'physical_success':None, 'input':info, 'historical_cycles':cycles,
        'historical_background_rejections':rejected, 'full_planner_replays':[],
        'historical_background_ready_count':sum(e['event']=='lookahead_plan_ready' for e in events),
        'handoff_tolerance_rad':.005, 'velocity_limits':cfg.velocity_limits,
        'acceleration_limits':cfg.acceleration_limits,
        'limitations':[
            'Full frozen target survives only in latest preview; other cycles are historical log audits.',
            'Belt plane is refitted from archived RGB-D, not the original cached online plane.',
            'Actual starting joints are recorded at foreground submission, after background submission; earlier stationary state is assumed.',
            'Flange is reconstructed from those joints; jaw is assumed open and unloaded.',
            'Continuous fresh stationary feedback during replay is simulated; no physical sender, jitter, sweep service, grip, carry or return is tested.',
            'Replay starts at historical planning time and consumes present host wall time; camera/online CPU contention is excluded.',
            'Nominal and actual groups retain historical conservative planning estimates and all configured motion limits.',
        ], 'source_sha256': {str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'src/nero_core/nero_core/two_second').glob('*.py')}}
    groups = [('old_nominal_early',False,False,begin,lead,post),
              ('new_actual_early',True,True,begin,lead,post),
              ('actual_late_foreground',True,False,info['submission_s'],late_lead,late_post)]
    for repeat in range(args.repeats):
        for name, actual, worker, stamp, reserve, post_reserve in (groups if repeat%2 == 0 else groups[::-1]):
            row = replay(cfg,target,state,start_s=stamp,lead_s=reserve,post_s=post_reserve,use_actual=actual,via_worker=worker)
            row.update(group=name, repeat=repeat+1)
            report['full_planner_replays'].append(row)
            (args.output/'report.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(json_value({k:row[k] for k in ('group','repeat','status','elapsed_s',
                'remaining_send_lead_s','executor_start_gate_passed','worker_claim_passed') if k in row}),ensure_ascii=False),flush=True)


if __name__ == '__main__': main()
