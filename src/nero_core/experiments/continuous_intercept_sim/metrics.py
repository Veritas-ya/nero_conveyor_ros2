"""Evaluation-only contact and retention metrics from actual simulator state."""
from __future__ import annotations

import numpy as np


def longest_interval(timestamps, mask):
    times = np.asarray(timestamps, float)
    mask = np.asarray(mask, bool)
    longest = 0.
    start = None
    for t, valid in zip(times, mask):
        if valid and start is None:
            start = t
        elif not valid:
            start = None
        if start is not None:
            longest = max(longest, float(t-start))
    return longest


def contact_summary(rows, *, lift_threshold=.020, hold_seconds=.5):
    """A close command and an actual lifted, retained object are different facts.

    Contact is read from MuJoCo force evaluation. No policy lifecycle bit is
    used as grasp evidence. The height reference is the settled support height.
    """
    rows=[r for r in rows if r.get('object_active',True)]
    if not rows:
        return {'closure_command_sent': False, 'physical_lift_and_hold': False,
                'bilateral_contact': False, 'lift_threshold_m': lift_threshold,
                'required_hold_s': hold_seconds}
    times = np.array([r['sim_s'] for r in rows])
    positions = np.array([r['object_pose'][:3] for r in rows])
    baseline = float(np.median(positions[times <= min(times[-1], times[0]+.3), 2]))
    contact = np.array([r['left_normal_force_n'] > 0 and r['right_normal_force_n'] > 0 for r in rows])
    raised = positions[:, 2]-baseline >= lift_threshold
    unsupported = np.array([not r['object_belt_contact'] for r in rows])
    retained = contact & raised & unsupported
    duration = longest_interval(times, retained)
    return {'closure_command_sent': any(r['gripper_command'] < .9 for r in rows),
            'bilateral_contact': bool(contact.any()),
            'max_lift_m': float(np.max(positions[:, 2]-baseline)),
            'support_reference_z_m': baseline,
            'lifted_bilateral_contact_duration_s': duration,
            'physical_lift_and_hold': duration >= hold_seconds,
            'lift_threshold_m': lift_threshold, 'required_hold_s': hold_seconds,
            'scope': 'MuJoCo contact and gravity; no physical hardware trial'}
