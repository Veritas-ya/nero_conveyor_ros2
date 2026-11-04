"""Shared reference operations, no scheduler, controller or device ownership."""
import numpy as np


def prefix_segment(plan, timestamp):
    while plan.prefix is not None and float(timestamp) < plan.start:
        plan = plan.prefix
    return plan


def splice_errors(parent, candidate, timestamp, *, sampler='sample_full'):
    return tuple(float(np.max(np.abs(a-b))) for a,b in
        zip(getattr(parent,sampler)(timestamp),getattr(candidate,sampler)(timestamp)))
