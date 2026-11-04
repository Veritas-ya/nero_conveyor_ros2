"""Admission uses the same moving-endpoint ETA as a direct full request."""
from types import SimpleNamespace
import numpy as np
import pytest
from mujoco_conveyor_sorting.controllers.dynamic_intercept import intercept_lead
from experiments.continuous_intercept_sim.full_preplanned import FullPickMirrorMixin

def test_upstream_early_prediction_covers_compute_and_direct_motion():
    mirror=FullPickMirrorMixin()
    mirror.control_dt=.02
    mirror.policy=SimpleNamespace(config=SimpleNamespace(planning_mode='full_preplanned',
        moving_approach_mode='continuous',full_plan_compute_budget_s=3.,
        trajectory_minimum_duration=.02,cartesian_linear_speed=.28,descent_lead_time=1.))
    p=np.array([-.78,0.,.5]);v=np.array([.2,0.,0.]);ee=np.array([.18,0.,.66])
    lead=mirror._dynamic_intercept_prediction_lead(p,v,ee,grasp_height=.035,
        approach_clearance=.1,cartesian_speed=.22,minimum_motion_time=.65,latency=.08,maximum_horizon=7.5)
    assert 4.04<=lead<4.041
    assert .02<=p[0]+v[0]*lead<.03
    travel=np.linalg.norm(p+v*lead+np.array([0,0,.035])-ee)/.28
    assert travel<=lead-3.04

def test_future_endpoint_not_current_upstream_distance_sets_motion_estimate():
    p=np.array([-.78,0.,.535]);v=np.array([.2,0.,0.]);ee=np.array([.18,0.,.66])
    lead=intercept_lead(p,v,ee,grasp_height=0.,approach_clearance=0.,
        cartesian_speed=.28,minimum_motion_time=1.,latency=3.04,maximum_horizon=7.5)
    assert abs(lead-4.04)<1e-7
    old_motion=np.linalg.norm(p-ee)/.28
    assert old_motion>3.4
    assert lead-3.04<1.001

def test_low_station_wait_is_separate_from_motion_duration():
    mirror=FullPickMirrorMixin();mirror.control_dt=.02;mirror.contact_station_x_m=.175
    mirror.policy=SimpleNamespace(config=SimpleNamespace(planning_mode='full_preplanned',
        moving_approach_mode='continuous',full_plan_compute_budget_s=3.,
        trajectory_minimum_duration=.02,cartesian_linear_speed=.28,descent_lead_time=2.))
    p=np.array([-1.2,0.,.5]);v=np.array([.2,0.,0.]);ee=np.array([.18,0.,.66])
    lead=mirror._dynamic_intercept_prediction_lead(p,v,ee,grasp_height=.035,
        approach_clearance=.1,cartesian_speed=.22,minimum_motion_time=.65,latency=.08,maximum_horizon=7.5)
    assert lead==pytest.approx(6.875)
    assert p[0]+v[0]*lead==pytest.approx(.175)
