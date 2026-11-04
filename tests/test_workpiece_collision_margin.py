"""The real-cell workpiece margin changes independently of the belt margin."""
import numpy as np
import pytest
from scipy.spatial import cKDTree

from test_collision_visualization import fixture
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def configure(planner, mode, distance):
    # A cube ends at x=.010. The moving cloud shifts +.1 m at stamp=2.
    planner.target.points = np.array([[.010 + distance - .1, 0., 1.]])
    planner.target_tree = cKDTree(planner.target.points)
    points = planner.target_points_at(2.)
    if mode == 'moving':
        return {'stamp': 2.}
    if mode == 'payload':
        planner.payload_local = points - planner.model.tcp()[:3, 3]
        planner.payload_tree = cKDTree(planner.payload_local)
        return {'payload': True}
    planner.released_points = points
    planner.released_tree = cKDTree(points)
    return {}


@pytest.mark.parametrize('mode', ['moving', 'payload', 'released'])
def test_2mm_workpiece_margin_and_recorded_value(mode):
    planner, _ = fixture(finger=False)
    planner.config.raw['workpiece_collision_margin_m'] = .002
    kwargs = configure(planner, mode, .003)
    planner.valid(np.zeros(7), .1, **kwargs)  # 3 mm clearance passes
    kwargs = configure(planner, mode, .001)
    with pytest.raises(PickRejected, match='工件点云碰撞') as failure:
        planner.valid(np.zeros(7), .1, **kwargs)
    assert failure.value.details['collision_margin_m'] == .002
    assert failure.value.details['margin_only_count'] == 1


def test_old_configuration_keeps_4mm_default():
    planner, _ = fixture(finger=False)
    kwargs = configure(planner, 'moving', .003)
    with pytest.raises(PickRejected) as failure:
        planner.valid(np.zeros(7), .1, **kwargs)
    assert failure.value.details['collision_margin_m'] == .004
