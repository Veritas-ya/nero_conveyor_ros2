"""Offline checks for the narrowly scoped operator-selected grasp policy."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_collision_visualization import fixture
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def test_moving_workpiece_contact_is_skipped_only_when_configured():
    p, _ = fixture(finger=False)
    with pytest.raises(PickRejected, match='工件点云碰撞'):
        p.valid(np.zeros(7), .1, stamp=2.)
    p.config.raw['grasp_workpiece_collision_check_enabled'] = False
    p.valid(np.zeros(7), .1, stamp=2.)
    p.valid(np.zeros(7), .1, stamp=2., contact=True)
    assert p.stats['moving_workpiece_checks_skipped'] == 2
    p.config.raw['grasp_workpiece_collision_check_enabled'] = True
    with pytest.raises(PickRejected, match='工件点云碰撞'):
        p.valid(np.zeros(7), .1, stamp=2.)


@pytest.mark.parametrize('mode', ['payload', 'released'])
def test_payload_and_released_robot_collision_still_checked(mode):
    p, _ = fixture(finger=False)
    p.config.raw['grasp_workpiece_collision_check_enabled'] = False
    points = p.target_points_at(2.)
    if mode == 'payload':
        p.payload_local = points-p.model.tcp()[:3, 3]
        p.payload_tree = cKDTree(p.payload_local)
        kwargs = {'payload': True}
    else:
        p.released_points = points
        p.released_tree = cKDTree(points)
        kwargs = {}
    with pytest.raises(PickRejected, match='工件点云碰撞'):
        p.valid(np.zeros(7), .1, **kwargs)


@pytest.mark.parametrize('fault', ['robot_scene_or_self', 'workspace', 'belt'])
def test_grasp_policy_keeps_other_checks(fault):
    p, _ = fixture(finger=False)
    p.config.raw['grasp_workpiece_collision_check_enabled'] = False
    if fault == 'robot_scene_or_self':
        def reject(*args, **kwargs):
            raise PickRejected('native_robot_check')
        p.model.check_state = reject
        message = 'native_robot_check'
    elif fault == 'workspace':
        p.config.workspace[2] = [-2., .5]
        message = '超出现场工作区'
    else:
        p.support_origin = np.array([0., 0., 1.1])
        p.plane[:3, 3] = p.support_origin
        p.model.gripper_geometries = [0]
        message = '低于实测带面'
    with pytest.raises(PickRejected, match=message):
        p.valid(np.zeros(7), .1, stamp=2.)
