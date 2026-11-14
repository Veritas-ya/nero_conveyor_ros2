"""Unchanged collision predicates with scalar-reference broad phases; no devices."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from test_payload_belt_clearance import payload_case


class ScalarBalls:
    """Reference: perform the original independent query for every geometry."""
    def __init__(self, tree): self.tree = tree
    def query_ball_point(self, centers, radii, **kwargs):
        return [self.tree.query_ball_point(c, r) for c, r in zip(centers, radii)]


def outcome(call):
    try:
        call()
    except PickRejected as error:
        return str(error), error.details
    return None


def test_batched_robot_scene_query_preserves_first_collision_and_passes():
    config = CellConfig.load(Path(__file__).parents[1]/
        'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    model = calibrated_model(config)
    rng = np.random.default_rng(182)
    q = config.standby
    scenes = [np.empty((0, 3)), model.data.geom_xpos[list(model.meshes)].copy(),
        rng.uniform(-1., 1., (4000, 3))]
    passed = rejected = 0
    for scene in scenes:
        tree = cKDTree(scene)
        model.scene = scene
        for delta in rng.uniform(-.08, .08, (20, 7)):
            model.tree = tree
            actual = outcome(lambda: model.check_state(q+delta, .1, allow_finger_contact=True))
            model.tree = ScalarBalls(tree)
            reference = outcome(lambda: model.check_state(q+delta, .1, allow_finger_contact=True))
            assert actual == reference
            passed += actual is None
            rejected += actual is not None
    assert passed and rejected


class ExactPayloadTree:
    """Keep exact neighbors but hide bounds to force the full-cloud fallback."""
    def __init__(self, tree): self.tree = tree
    def query_ball_point(self, *args, **kwargs): return self.tree.query_ball_point(*args, **kwargs)


def test_payload_height_bound_matches_exact_cloud_on_rotated_planes_and_boundaries():
    rng = np.random.default_rng(190)
    passed = rejected = 0
    for height in np.r_[[-.05, -.010000000001, -.01, -.009999999999, 0.,
                          .009999999999, .01, .010000000001, .05], rng.uniform(-.03, .03, 50)]:
        planner = payload_case(.004)
        rotation = Rotation.random(random_state=rng).as_matrix()
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.random(random_state=rng).as_matrix()
        tcp[:3, 3] = [0., 0., 1.]
        planner.model.tcp = lambda: tcp.copy()
        planner.support = rotation[:, 2]
        planner.support_origin = tcp[:3, 3]-planner.support*height
        planner.plane[:3, :3] = rotation
        planner.plane[:3, 3] = planner.support_origin
        local_scene = rng.uniform([-.015, -.015, -.005], [.015, .015, .005], (500, 3))
        planner.model.scene = local_scene@rotation.T+planner.support_origin
        planner.model.tree = cKDTree(planner.model.scene)
        planner.payload_local = rng.uniform(-.01, .01, (100, 3))
        tree = cKDTree(planner.payload_local)
        planner.payload_tree = tree
        actual = outcome(lambda: planner.valid(np.zeros(7), .1, payload=True))
        planner.payload_tree = ExactPayloadTree(tree)
        reference = outcome(lambda: planner.valid(np.zeros(7), .1, payload=True))
        assert actual == reference
        passed += actual is None
        rejected += actual is not None
    assert passed and rejected


def test_separated_payload_skips_scene_query_but_near_belt_retains_it():
    planner = payload_case(.05)
    def forbidden(*args): pytest.fail('a fully separated cloud cannot have a belt neighbor')
    planner.model.tree = SimpleNamespace(query=forbidden)
    planner.valid(np.zeros(7), .1, payload=True)
    with pytest.raises(PickRejected, match='载物点云与实测场景碰撞'):
        payload_case(.004).valid(np.zeros(7), .1, payload=True)
