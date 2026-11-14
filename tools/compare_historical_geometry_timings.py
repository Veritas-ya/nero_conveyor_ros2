#!/usr/bin/env python3
"""Deduplicate cumulative geometry snapshots; never invent missing samples."""
import argparse
import json
from pathlib import Path
import numpy as np


def stats(values):
    a = np.asarray(values, float)*1000
    if not len(a):
        return None
    return dict(n=len(a), mean_ms=float(a.mean()), p50_ms=float(np.percentile(a,50)),
        p95_ms=float(np.percentile(a,95)), max_ms=float(a.max()), min_ms=float(a.min()))


def extract(directory):
    summary = json.loads((directory/'summary.json').read_text())
    vision = summary.get('vision_diagnostics') or {}
    final = (vision.get('grasp_geometry') or {}).get('operation_timings_s',{}).get('single_exposure_geometry')
    if not final:
        return None
    entries = {}
    events_path = directory/'events.jsonl'
    with events_path.open() as stream:
        for line_number, line in enumerate(stream,1):
            event = json.loads(line)
            geometry = ((event.get('data',{}).get('vision') or {}).get('grasp_geometry') or {})
            t = geometry.get('operation_timings_s',{}).get('single_exposure_geometry')
            if not t:
                continue
            count = t['count']
            item = entries.setdefault(count, dict(operation_index=count, wall_s=t['last_s'],
                cumulative_s=t['total_s'], outcome='unknown', source_line=line_number))
            assert abs(item['wall_s']-t['last_s']) < 1e-9, 'inconsistent duplicate'
            # A later frame may already be detecting while last_s is unchanged.
            # Only use settled snapshots with reconciled ready/blocker counts.
            ready = geometry.get('target_ready_evaluations',0)
            blocked = sum(geometry.get('blocker_counts',{}).values())
            if ready+blocked == count and geometry.get('stage') == 'target_ready':
                item.update(outcome='ready',frame=geometry.get('frame'),source_line=line_number)
            elif ready+blocked == count and geometry.get('stage') == 'waiting_target_geometry':
                item.update(outcome='rejected',frame=geometry.get('frame'),source_line=line_number)
    ordered = [entries[k] for k in sorted(entries)]
    # Cumulative increments recover a sample only if exactly one operation elapsed.
    previous_count, previous_total = 0, 0.
    gaps = []
    for row in ordered:
        delta = row['operation_index']-previous_count
        residual = row['cumulative_s']-previous_total-row['wall_s']
        if delta == 1:
            assert abs(residual) < 1e-8, 'cumulative increment mismatch'
        elif delta > 1:
            gaps.append(dict(indices=list(range(previous_count+1,row['operation_index'])),
                combined_wall_s=residual, individual_durations_available=False))
        previous_count, previous_total = row['operation_index'],row['cumulative_s']
    assert previous_count == final['count']
    assert abs(previous_total-final['total_s']) < 1e-8
    success = [r for r in ordered if r['outcome']=='ready']
    ready_count = vision['grasp_geometry']['target_ready_evaluations']
    return dict(run=directory.parent.name, speed_cm_s=vision.get('velocity_policy',{}).get('fixed_belt_speed_cm_s'),
        final_accumulator=final, all_outcomes_exact_mean_ms=final['total_s']/final['count']*1000,
        ready_count=ready_count, blockers=vision['grasp_geometry'].get('blocker_counts'),
        ready_samples_complete=len(success)==ready_count,
        ready_distribution=stats([r['wall_s'] for r in success]),samples=ordered,gaps=gaps,
        source_summary=str(directory/'summary.json'),source_events=str(events_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,default=Path('runs/phase_timing_test'))
    parser.add_argument('--new',type=Path,default=Path('outputs/camera_timing_trial_01/summary.json'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    runs = [extract(p.parent) for p in sorted(args.runs.glob('20261113_*/hardware/summary.json'))]
    runs = [r for r in runs if r]
    assert all(r['ready_samples_complete'] for r in runs), 'missing successful samples'
    new = json.loads(args.new.read_text())
    report = dict(old_runs=runs,old_ready_pooled=stats([s['wall_s'] for r in runs for s in r['samples'] if s['outcome']=='ready']),
        new_source=str(args.new),new_ready=new['distributions']['geometry_ready']['stages']['geometry_total']['wall_s'],
        limitations=[
            'Historical metric is frozen_geometry wall time; new metric is detection plus scene/OBB/seed checking, without speed/admission steps.',
            'Historical robot execution/feedback/recording load differs from camera-only process with assumed standby/open state.',
            'Old samples are per triggered exposure, new samples include repeated observations; belt speed of new run was not recorded.',
            'Do not treat pooled historical speeds/size policies as a controlled before-after experiment.',
            'Two rejection durations in the 53.5cm/s run are missing individually. Only their sum is recoverable.',
            'Quantiles use linear interpolation; four old successful samples cannot establish a stable tail distribution.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for r in runs:
        print(r['run'],r['speed_cm_s'],r['ready_distribution'],'gaps',r['gaps'])
    print('pooled',report['old_ready_pooled'])
    print('new',report['new_ready'])


if __name__ == '__main__':
    main()
