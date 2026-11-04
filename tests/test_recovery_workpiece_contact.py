"""Exercise the contact scope with real cloud predicates; no hardware devices."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from test_collision_visualization import fixture as collision_fixture
from nero_core.two_second.workpiece_contact import (
    recovery_contact_allowed, recovery_workpiece_contact)
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


@pytest.mark.parametrize('mode', ['moving', 'payload', 'released'])
def test_only_recovery_accepts_robot_contact_with_cancelled_workpiece(mode):
    p, _ = collision_fixture(finger=False)
    points = p.target_points_at(2.)
    kwargs = {'stamp':2.}
    if mode == 'payload':
        p.payload_local = points-p.model.tcp()[:3, 3]
        p.payload_tree = cKDTree(p.payload_local)
        kwargs = {'payload':True}
    elif mode == 'released':
        p.released_points = points
        p.released_tree = cKDTree(points)
        kwargs = {}
    with pytest.raises(PickRejected, match='工件点云碰撞'):
        p.valid(np.zeros(7), .1, **kwargs)
    with recovery_workpiece_contact(p, p.target, enabled=True):
        assert recovery_contact_allowed(p)
        p.valid(np.zeros(7), .1, **kwargs)
    assert not recovery_contact_allowed(p)
    with pytest.raises(PickRejected, match='工件点云碰撞'):
        p.valid(np.zeros(7), .1, **kwargs)


def test_disabled_policy_and_another_task_never_inherit_contact_permission():
    p, _ = collision_fixture()
    with recovery_workpiece_contact(p, p.target, enabled=False):
        with pytest.raises(PickRejected, match='工件点云碰撞'):
            p.valid(np.zeros(7), .1, stamp=2.)
    with pytest.raises(PickRejected, match='当前工件不一致'):
        with recovery_workpiece_contact(p, SimpleNamespace(), enabled=True):
            pytest.fail('a different task must not enter the scope')
    original = p.target
    with recovery_workpiece_contact(p, original, enabled=True):
        p.target = SimpleNamespace(**vars(original))  # even the same identity
        assert not recovery_contact_allowed(p)
        with pytest.raises(PickRejected, match='工件点云碰撞'):
            p.valid(np.zeros(7), .1, stamp=2.)


@pytest.mark.parametrize('fault', ['native_self', 'native_scene', 'workspace', 'belt', 'payload_scene'])
def test_non_workpiece_constraints_remain_active_during_recovery(fault):
    p, _ = collision_fixture(finger=False)
    message = fault
    kwargs = {'stamp':2.}
    if fault.startswith('native_'):
        def reject(*args, **kw): raise PickRejected(message)
        p.model.check_state = reject
    elif fault == 'workspace':
        p.config.workspace[2] = [-2., .5]
        message = '超出现场工作区'
    elif fault == 'belt':
        p.support_origin = np.array([0., 0., 1.1])
        p.plane[:3, 3] = p.support_origin
        p.model.gripper_geometries = [0]
        message = '低于实测带面'
    else:
        points = p.target_points_at(2.)
        p.payload_local = points-p.model.tcp()[:3, 3]
        p.payload_tree = cKDTree(p.payload_local)
        p.capture_tcp = p.model.tcp()
        p.model.scene = points.copy()
        p.model.tree = cKDTree(points)
        kwargs = {'payload':True}
        message = '载物点云与实测场景碰撞'
    with recovery_workpiece_contact(p, p.target, enabled=True):
        with pytest.raises(PickRejected, match=message):
            p.valid(np.zeros(7), .1, **kwargs)


def test_fault_unwinds_contact_permission_before_next_attempt():
    p, _ = collision_fixture()
    with pytest.raises(RuntimeError, match='injected failure'):
        with recovery_workpiece_contact(p, p.target, enabled=True):
            raise RuntimeError('injected failure')
    assert not recovery_contact_allowed(p)
