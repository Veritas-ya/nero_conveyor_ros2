"""Simulation and deployed planner use the same moving-cloud predicate."""
from types import SimpleNamespace
import numpy as np
from test_collision_visualization import fixture
from experiments.two_second_grasp.moving_target_check import SimulationMovingTargetCheck
from nero_core.two_second.moving_target_collision import first_moving_target_hit
from experiments.two_second_grasp.smooth_execution import _rejected_state_details


def test_simulation_uses_observed_cloud_velocity_and_same_contact_exception():
    planner,_=fixture()
    model=planner.model
    model.set_state=lambda q,width:None
    points=planner.target.points.copy()
    checker=SimulationMovingTargetCheck(model,points,np.eye(4),.1)
    reference=SimpleNamespace(start_s=1.,capture_s=1.,contact_s=3.,velocity_world=np.array([.1,0,0]))
    hit=checker.check(np.zeros(7),reference,1.)
    expected=first_moving_target_hit(model,points,checker.tree,np.array([.1,0,0]),contact=False)
    assert hit['geometry']=='TEST_left_finger' and hit['hit_count']==len(expected[1])
    assert hit['margin_m']==.004 and not hit['contact_allowed']
    # At the same predicted object position the existing finger exception
    # applies only after the defined contact time.
    reference.contact_s=2.
    assert checker.check(np.zeros(7),reference,1.) is None
    assert not np.shares_memory(checker.points,points)


def test_state_rejection_distinguishes_posture_and_tcp_height_from_collision():
    env=SimpleNamespace(model=SimpleNamespace(jnt_range=np.tile([-1.,1.],(7,1))),
        arm_joint_ids=np.arange(7), robot_controller=SimpleNamespace(config=SimpleNamespace(joint_command_margin=.02)),
        ee_site_id=0, belt_height=.5)
    data=SimpleNamespace(site_xpos=np.array([[0.,0.,.49]]),site_xmat=np.array([np.eye(3).reshape(-1)]))
    assert _rejected_state_details(env,np.zeros(7),data)['reason']=='tcp_below_belt'
    data.site_xpos[0,2]=.6
    assert _rejected_state_details(env,np.zeros(7),data)['reason']=='approach_axis_not_downward'
    joints=np.zeros(7);joints[5]=.99
    assert _rejected_state_details(env,joints,data)=={'reason':'joint_margin','joint_number':6,'margin_rad':1.-.99,'required_rad':.02}
