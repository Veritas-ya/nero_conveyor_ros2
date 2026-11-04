"""Limit the operator's lightweight-workpiece contact policy to one recovery."""
from contextlib import contextmanager

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def recovery_contact_allowed(planner):
    target = getattr(planner, '_recovery_workpiece_contact_target', None)
    return target is not None and target is planner.target


@contextmanager
def recovery_workpiece_contact(planner, target, *, enabled):
    """Permit robot contact only with this cancelled task's point cloud.

    Native self/environment checks and payload-to-environment checks still run.
    The executor enters only after joining the failed attempt's sender/workers;
    it leaves only after joining the recovery sender/workers, including faults.
    """
    if not enabled:
        yield False
        return
    if planner.target is not target:
        raise PickRejected('恢复接触策略与当前工件不一致')
    previous = getattr(planner, '_recovery_workpiece_contact_target', None)
    planner._recovery_workpiece_contact_target = target
    try:
        yield True
    finally:
        planner._recovery_workpiece_contact_target = previous
