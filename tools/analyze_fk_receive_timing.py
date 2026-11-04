#!/usr/bin/env python3
"""Compare receive-time shifts offline; sparse trace rows are not raw CAN samples."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model


def evaluate(path, model):
    trace = np.load(path, allow_pickle=False)
    q = trace['feedback_joints_at_pose_rad']
    pose = trace['feedback_flange_pose_m_rad']
    stamps = np.min(trace['feedback_pose_packet_monotonic_s'], axis=1)
    stage = trace['feedback_sample_stage']
    finite = np.isfinite(q).all(axis=1) & np.isfinite(pose).all(axis=1) & np.isfinite(stamps)
    q, pose, stamps, stage = q[finite], pose[finite], stamps[finite], stage[finite]
    unique = np.r_[True, np.diff(stamps) > 1.e-8]
    q, pose, stamps, stage = q[unique], pose[unique], stamps[unique], stage[unique]
    if len(stamps) < 4:
        return None
    delta = np.diff(stamps)
    speed = np.r_[0., np.max(abs(np.diff(q, axis=0)), axis=1)/np.maximum(delta, 1.e-9)]
    offsets = np.arange(-.008, .0081, .002)
    rows = []
    for i in np.flatnonzero(speed > .5):
        brackets = []
        for offset in offsets:
            t = stamps[i]+offset
            j = np.searchsorted(stamps, t)
            if (j == 0 or j == len(stamps) or stamps[j]-stamps[j-1] > .05
                    or stage[j] != stage[i] or stage[j-1] != stage[i]):
                break
            brackets.append(j)
        else:
            rows.append((i, brackets))
    if len(rows) < 10:
        return None
    results = []
    for k, offset in enumerate(offsets):
        distances, rotations = [], []
        for i, brackets in rows:
            j = brackets[k]
            weight = (stamps[i]+offset-stamps[j-1])/(stamps[j]-stamps[j-1])
            model.set_state(q[j-1]*(1-weight)+q[j]*weight, .1)
            flange = model.flange()
            distances.append(1000*np.linalg.norm(flange[:3, 3]-pose[i, :3]))
            rotations.append(np.rad2deg((Rotation.from_matrix(flange[:3, :3]).inv()
                * Rotation.from_euler('xyz', pose[i, 3:])).magnitude()))
        results.append({'offset_ms': round(float(offset)*1000, 3),
            'position_median_mm': float(np.median(distances)),
            'position_p95_mm': float(np.percentile(distances, 95)),
            'angle_median_deg': float(np.median(rotations))})
    return {'trace': str(path), 'moving_samples_same_for_all_offsets': len(rows),
            'best_median': min(results, key=lambda x: x['position_median_mm']),
            'zero_offset': results[4], 'offsets': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = calibrated_model(CellConfig.load(args.config))
    result = {'method': 'offline interpolation of saved pose-paired rows, velocity > 0.5 rad/s',
        'limitations': ['host receive time is not controller sample time',
            'saved feedback rows are sparse; interpolated rows do not reconstruct raw CAN packets',
            'same motion rows are used at every offset; this is not a validated hardware clock calibration'],
        'traces': [item for path in args.traces if (item := evaluate(path, model)) is not None]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
