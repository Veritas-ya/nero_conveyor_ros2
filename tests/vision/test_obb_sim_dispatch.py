from types import SimpleNamespace
import numpy as np
import pytest
from experiments.graspgenx_sim.obb_sim_dispatch import OBBSimulationDispatch, SimulationDispatchRejected


def make():
    q=np.zeros(7)
    actuator=SimpleNamespace(commanded_joint_positions=q,commanded_joint_velocities=q,
        preview_joint_step=lambda target,previous,velocity,dt:(target,))
    env=SimpleNamespace(control_dt=.02,robot_controller=actuator,_action_queue=[])
    gripper=SimpleNamespace(target_open_fraction=1., command=lambda value:None)
    robot=SimpleNamespace(gripper=gripper,apply_action=lambda action,dt:SimpleNamespace(joint_target=action['arm'],time_parameterized=True))
    bundle=SimpleNamespace(execution_start=1.,end=5.,sample_command_full=lambda t:(q,np.zeros(7),np.zeros(7)),phase=lambda t:'approach')
    policy=SimpleNamespace(config=SimpleNamespace(planning_mode='full_preplanned'),
        _full_pick_bundle=bundle,_full_pick_abort_reason=None,stage=SimpleNamespace(value='pregrasp'),
        full_pick_dispatch_reason=lambda obs,target,clock:None)
    ctx=SimpleNamespace(policy=policy,robot_controller=robot,settings=SimpleNamespace(open_width_m=.1),motion_validation_rejection={},
        validate_command=lambda state,command:True,validate_gripper_command=lambda state,width:True)
    return env,ctx


def test_dispatch_samples_application_clock_and_rejects_scene_failure():
    env,ctx=make();seen=[]
    ctx.policy.full_pick_dispatch_reason=lambda obs,target,clock:seen.append(clock) or 'scene_review_expired_or_missing_for_plan'
    dispatch=OBBSimulationDispatch(env,ctx,10.)
    with pytest.raises(SimulationDispatchRejected,match='scene_review_expired'):
        dispatch.prepare({'arm':np.ones(7),'gripper':1.},{'timestamp':2.015},{'joints':np.zeros(7),'width_m':.1},application_sim_s=12.015)
    assert seen==pytest.approx([2.015]) and dispatch.count==0


def test_dispatch_first_close_commits_only_after_geometry_acceptance():
    env,ctx=make();calls=[]
    ctx.policy.dispatch_closure_status=lambda action,obs:{'ready':True,'gripper':-1.}
    def commit(action,obs):calls.append(True);action.pop('_dispatch_closure_intent');action['gripper']=-1.;return True
    ctx.policy.commit_dispatch_closure=commit
    d=OBBSimulationDispatch(env,ctx,10.); state={'joints':np.zeros(7),'width_m':.1};obs={'timestamp':2.}
    ctx.validate_gripper_command=lambda state,width:False
    with pytest.raises(SimulationDispatchRejected,match='gripper_geometry'):
        d.prepare({'arm':np.zeros(7),'gripper':1.,'_dispatch_closure_intent':{}},obs,state,application_sim_s=12.)
    assert calls==[]
    ctx.validate_gripper_command=lambda state,width:True
    value=d.prepare({'arm':np.zeros(7),'gripper':1.,'_dispatch_closure_intent':{}},obs,state,application_sim_s=12.)
    assert value['gripper']==-1. and len(calls)==1


def test_dispatch_refuses_invalidated_bundle():
    env,ctx=make();d=OBBSimulationDispatch(env,ctx,10.);ctx.policy._full_pick_bundle=None
    with pytest.raises(SimulationDispatchRejected,match='missing_certified_bundle'):
        d.prepare({'arm':np.zeros(7),'gripper':1.},{'timestamp':2.},{'joints':np.zeros(7),'width_m':.1},application_sim_s=12.015)


def test_retimer_rejection_records_difference_and_never_sends():
    env,ctx=make()
    env.robot_controller.preview_joint_step=lambda target,previous,velocity,dt:(target+np.full(7,.001),)
    dispatch=OBBSimulationDispatch(env,ctx,10.)
    with pytest.raises(SimulationDispatchRejected,match='simulation_retimer'):
        dispatch.prepare({'arm':np.zeros(7),'gripper':1.},{'timestamp':2.},
            {'joints':np.zeros(7),'width_m':.1},application_sim_s=12.)
    assert dispatch.count==0
    assert dispatch.last_guard_diagnostics['maximum_joint_difference_rad']==pytest.approx(.001)
    assert dispatch.last_guard_diagnostics['phase']=='approach'


def test_reserved_endpoint_float_roundoff_does_not_leave_workspace():
    from mujoco_conveyor_sorting.runtime.nero_moving_context import NeroMovingPolicyContext
    ctx=SimpleNamespace(data=SimpleNamespace(time=0.),_target={'object_id':1},_reserved=True,
        _motion_snapshot=SimpleNamespace(valid=True,current_grasp=np.eye(4),velocity_source='test'),
        _motion_velocity=lambda:np.zeros(3),_window=lambda current,velocity,now:{'start_time':0.,'end_time':10.},
        _measured_plane_normal=None,moving_approach_mode='continuous',
        policy=SimpleNamespace(config=SimpleNamespace(trajectory_replan_position_tolerance=.004,trajectory_replan_orientation_tolerance=.01)),
        s_range=(-.2,.2),_coordinate=lambda xyz:float(xyz[0]),_planned_time=None,_planned_pose=None,
        _intercept_updates=[],_refresh_prediction=lambda now:None,
        _last_observation={'objects':[],'predictions':{1:{}}})
    pose=np.array([np.nextafter(-.2,-1.),0,0,1,0,0,0])
    accepted=NeroMovingPolicyContext.refine_selected_grasp_plan(ctx,1,1.,requested_time=1.,requested_pose=pose)
    assert accepted['accepted']
    pose[0]=-.200001
    rejected=NeroMovingPolicyContext.refine_selected_grasp_plan(ctx,1,1.,requested_time=1.,requested_pose=pose)
    assert rejected['reason']=='trajectory_outside_reachable_workspace'
