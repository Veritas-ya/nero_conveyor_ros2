"""Check measured payload/belt separation without hardware."""
import numpy as np
import pytest
from scipy.spatial import cKDTree
from test_collision_visualization import fixture
from nero_core.two_second.belt_collision_scene import BELT_SURFACE_ONLY
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def payload_case(distance, scope=BELT_SURFACE_ONLY):
    p, _ = fixture(finger=False)
    p.config.raw['payload_belt_clearance_m'] = .005
    p.model.meshes = {}
    point = np.array([[0., 0., 1.]])
    p.payload_local = point - p.model.tcp()[:3, 3]
    p.payload_tree = cKDTree(p.payload_local)
    p.model.scene = np.array([[0., 0., 1. - distance]])
    p.model.tree = cKDTree(p.model.scene)
    p.support_origin = p.model.scene[0].copy()
    p.support = np.array([0., 0., 1.])
    p.plane[:3, 3] = p.support_origin
    p.capture_tcp = np.eye(4)  # well away from the support-contact exception
    p.scene_report = {'scope': scope}
    return p


def test_belt_5mm_accepts_6mm_and_rejects_4mm_with_recorded_clearance():
    payload_case(.006).valid(np.zeros(7), .1, payload=True)
    p = payload_case(.004)
    with pytest.raises(PickRejected, match='载物点云与实测场景碰撞') as error:
        p.valid(np.zeros(7), .1, payload=True)
    assert error.value.details['margin_m'] == .005
    assert error.value.details['minimum_distance_m'] == pytest.approx(.004)


def test_general_scene_keeps_its_original_8mm_clearance():
    p = payload_case(.006, scope='observed_scene')
    with pytest.raises(PickRejected) as error:
        p.valid(np.zeros(7), .1, payload=True)
    assert error.value.details['margin_m'] == .008
