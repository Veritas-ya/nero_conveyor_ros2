"""Adapter boundaries, not physics-success scores or relaxed control limits."""
from collections import deque
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from experiments.continuous_intercept_sim.full_preplanned import (
    FullPickPhysicalMixin, TargetGeometry, tcp_prediction,completed_lift_hold,
    object_offset_at_capture,matrix,FullPickMirrorMixin,
)
from experiments.continuous_intercept_sim.run import action_record


def test_tcp_observation_uses_original_exposure_and_fixed_grasp_offset():
    prediction=dict(observed_pose=np.array([.1,.2,.3,1.,0.,0.,0.]),
        current_estimated_pose=np.array([.12,.2,.3,1.,0.,0.,0.]),
        predicted_grasp_pose=np.array([.2,.2,.3,1.,0.,0.,0.]),
        observed_timestamp=7.5,estimated_velocity=np.array([.04,0.,0.]),visible=False)
    current=np.array([.12,.2,.35,0.,1.,0.,0.])
    out=tcp_prediction(prediction,current)
    np.testing.assert_allclose(out['observed_pose'],[.1,.2,.35,0.,1.,0.,0.],atol=1e-12)
    np.testing.assert_allclose(out['predicted_ee_grasp_pose'],[.2,.2,.35,0.,1.,0.,0.],atol=1e-12)
    np.testing.assert_array_equal(out['current_ee_grasp_pose'],current)
    assert out['observed_timestamp']==7.5 and out['visible'] is False
    np.testing.assert_array_equal(prediction['observed_pose'],[.1,.2,.3,1.,0.,0.,0.])
    np.testing.assert_array_equal(out['estimated_velocity'],prediction['estimated_velocity'])


def test_changing_grasp_yaw_preserves_original_measured_object_pose_and_clock():
    measured=np.array([.1,.09,.5,np.cos(.6),0.,0.,np.sin(.6)])
    prediction=dict(object_observed_pose=measured.copy(),observed_timestamp=7.5)
    expected=measured.copy();expected[0]+=.2*(8.-7.5)
    for angle in (0.,1.2,-1.2,np.pi):
        tcp=np.array([.2,.09,.535,0.,np.cos(angle/2),np.sin(angle/2),0.])
        request=SimpleNamespace(target_pose=tcp,planned_target_time_s=8.,continuation_velocity_m_s=(.2,0.,0.))
        offset=object_offset_at_capture(prediction,request)
        np.testing.assert_allclose(matrix(tcp)@offset,matrix(expected),atol=1e-12)
        np.testing.assert_array_equal(prediction['object_observed_pose'],measured)
        assert prediction['observed_timestamp']==7.5


def test_worker_snapshot_uses_adopted_decorated_observation_not_parent_dictionary():
    original={'timestamp':3.24,'objects':[dict(object_id=0,recommended_grip_width=0.)],
        'dispatch_clock':{},'predictions':{0:dict(observed_pose=np.array([.1,.09,.5,1.,0.,0.,0.]),
            current_estimated_pose=np.array([.12,.09,.5,1.,0.,0.,0.]),
            current_ee_grasp_pose=np.array([.12,.09,.55,0.,1.,0.,0.]),
            predicted_grasp_pose=np.array([.2,.09,.5,1.,0.,0.,0.]),estimated_velocity=np.array([.2,0.,0.]))}}
    class Parent:
        def adopt_snapshot(self,snap):
            self._comparison_observation=original
            return original
    class Mirror(FullPickMirrorMixin,Parent):
        control_latency=.014
        control_dt=.02
        policy=SimpleNamespace(_full_pick_catalog_quaternion=None)
        def adapt_grasp_observation(self,obs,snap):
            from copy import deepcopy
            return deepcopy(obs)
    mirror=Mirror();obs=mirror.adopt_snapshot({'sim_s':3.24})
    assert mirror._comparison_observation is obs and obs is not original
    assert 'tcp_to_object' in mirror._comparison_observation['predictions'][0]
    assert 'tcp_to_object' not in original['predictions'][0]


def geometry():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom name="conveyor_surface" type="plane" size="1 1 .1"/>
      <body name="link1" pos="0 0 .05"><joint name="joint1" type="slide" axis="1 0 0"/>
        <geom name="robot" type="sphere" size=".03" mass="1"/><site name="ee"/></body>
      <body name="object_0" pos=".2 0 .05"><freejoint/>
        <geom name="object" type="sphere" size=".03" mass=".1"/></body>
    </worldbody></mujoco>''')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    env=SimpleNamespace(model=model,data=data,arm_qpos_adrs=[0],ee_site_id=model.site('ee').id,
        _robot_collision_body_ids={model.body('link1').id},belt_geom_id=model.geom('conveyor_surface').id,
        _objects=[SimpleNamespace(logical_id=0,qpos_adr=1,geom_ids=[model.geom('object').id])])
    checker=TargetGeometry(env);checker.offsets={0:np.eye(4)}
    return env,checker


def request(env,x=.2,begin=0.,end=0.,velocity=(0.,0.,0.),phase='approach'):
    return SimpleNamespace(target_id=0,generation=7,request_id=9,scene_epoch=1,
        start_q=env.data.qpos[env.arm_qpos_adrs].copy(),full_qpos=env.data.qpos.copy(),
        full_sample_start_time_s=begin,full_sample_time_s=end,planned_target_time_s=1.,
        intercept_request_time_s=0.,continuation_velocity_m_s=velocity,
        target_pose=np.array([x,0.,.05,1.,0.,0.,0.]),full_phase=phase,
        planning_mode='full_preplanned',intercept_position_tolerance_m=.004)


def test_static_loaded_geometry_is_bound_to_one_request_and_follows_candidate_tcp():
    env,checker=geometry();req=request(env);req.planned_target_time_s=None
    original=env.data.qpos.copy()
    assert not checker(req,np.zeros(1))
    assert checker.last_rejection['reason']=='missing_simulated_capture_time'
    checker.bind_loaded_request(req)
    assert checker(req,np.array([.1]))
    np.testing.assert_allclose(checker.data.qpos[1:4],[.3,0.,.05],atol=1e-12)
    np.testing.assert_array_equal(env.data.qpos,original)
    req.request_id+=1
    assert not checker(req,np.array([.1]))
    assert checker.last_rejection['reason']=='missing_simulated_capture_time'


def test_static_loaded_binding_still_rejects_payload_world_collision():
    env,checker=geometry();req=request(env);req.planned_target_time_s=None
    checker.bind_loaded_request(req)
    checker.loaded_requests[checker.request_key(req)][2,3]-=.1
    assert not checker(req,np.zeros(1))
    assert checker.last_rejection['reason']=='simulated_payload_world_contact'


def test_target_collision_is_checked_at_all_sweep_samples_without_physical_pose_writes():
    env,checker=geometry();original=env.data.qpos.copy()
    assert checker(request(env),np.zeros(1))
    assert not checker(request(env,x=.04),np.zeros(1))
    assert checker.last_rejection['reason']=='simulated_target_robot_contact'
    # Future endpoint is clear, but the earlier part of this immutable sweep
    # places the sphere at x=.05 and intersects the arm.
    assert not checker(request(env,x=.15,begin=0.,end=1.,velocity=(.1,0.,0.)),np.zeros(1))
    np.testing.assert_array_equal(env.data.qpos,original)


def test_geometry_does_not_reuse_a_same_joint_collision_result_for_a_new_target():
    env,checker=geometry();q=np.zeros(1)
    assert checker(request(env,x=.2),q)
    assert not checker(request(env,x=.04),q)
    assert checker(request(env,x=.2),q)


def test_conditional_payload_contacts_world_in_private_model():
    env,checker=geometry()
    # Permit this test hand contact; a known payload attached to the TCP is
    # then moved below its support by an offset. It must hit the plane.
    checker.fingers.add(env.model.body('link1').id)
    checker.offsets[0][2,3]=-.04
    assert not checker(request(env,phase='lift'),np.zeros(1))
    assert checker.last_rejection['reason']=='simulated_payload_world_contact'


def test_actual_application_clock_and_fixed_failure_hold_do_not_follow_feedback():
    applied=[];intervals=[]
    class Controller:
        commanded_joint_positions=np.zeros(7)
        commanded_joint_velocities=np.zeros(7)
        def preview_joint_step(self,q,previous,velocity,dt):
            intervals.append(dt);return q+.1, None
        def apply_action(self,action,**kwargs):applied.append((action,kwargs))
    class Physical(FullPickPhysicalMixin):pass
    env=Physical();env.data=SimpleNamespace(time=1.04,qpos=np.arange(7,dtype=float))
    env.arm_qpos_adrs=np.arange(7);env.control_dt=.02;env.action_type='joint_position'
    env.last_actuator_application_sim_s=1.;env.robot_controller=Controller()
    bundle=SimpleNamespace(start=1.,end=4.,lift_time=3.,
        sample_full=lambda t:(np.full(7,t),np.ones(7),np.zeros(7)),
        sample_command_full=lambda t:(np.full(7,t),np.ones(7),np.zeros(7)),command_curves=None)
    env._action_queue=deque([(1.04,dict(arm=np.full(7,99.),gripper=-1.,
        _full_pick_bundle=bundle,_full_pick_lift_admitted=False,_full_pick_config=None))])
    env._apply_due_actions()
    assert intervals==pytest.approx([.04])
    assert applied[0][1]['dt']==pytest.approx(.04)
    row=env.full_pick_dispatch_rows[0]
    np.testing.assert_array_equal(row['reference_q'],np.full(7,1.04))
    assert not row['candidate_sent']
    anchor=applied[0][0]['arm'].copy()
    env.data.time=1.06;env.data.qpos+=.2
    env._action_queue.append((1.06,dict(arm=np.ones(7),gripper=1.)))
    env._apply_due_actions()
    np.testing.assert_array_equal(applied[1][0]['arm'],anchor)
    assert applied[1][0]['gripper']==-1.


def test_action_log_references_certificate_instead_of_copying_or_serializing_trajectory():
    bundle=SimpleNamespace(request=SimpleNamespace(request_id=3))
    out=action_record(dict(arm=np.zeros(7),gripper=1.,_full_pick_bundle=bundle,
                           _full_pick_config=object()))
    assert out['full_pick_request_id']==3
    assert '_full_pick_bundle' not in out and '_full_pick_config' not in out


def test_policy_abort_invalidates_queued_certificates_and_keeps_one_fixed_hold():
    class Base:
        def _queue_action(self,action):self._action_queue.append((10.,dict(action)))
    class Physical(FullPickPhysicalMixin,Base):pass
    env=Physical();env._action_queue=deque([(8.,{'_full_pick_bundle':object()})])
    anchor=np.arange(7,dtype=float)
    env._queue_action(dict(arm=anchor,gripper=-1.,_full_pick_stop_reason='task_failed'))
    assert len(env._action_queue)==1
    assert '_full_pick_bundle' not in env._action_queue[0][1]
    env._queue_action(dict(arm=anchor+.5,gripper=1.,_full_pick_stop_reason='task_failed'))
    np.testing.assert_array_equal(env._full_pick_fault_hold,anchor)
    assert env._full_pick_fault_gripper==-1.


def test_complete_lift_holds_certified_endpoint_without_an_old_curve_clock():
    endpoint=np.arange(7,dtype=float)
    policy=SimpleNamespace(_full_pick_bundle=SimpleNamespace(lift=SimpleNamespace(q1=endpoint)),
                           _grasp_gripper_action=-1.)
    action=completed_lift_hold(policy)
    np.testing.assert_array_equal(action['arm'],endpoint)
    assert action['gripper']==-1.
    assert action['_full_pick_stop_reason']=='completed_lift_holding'
    assert '_full_pick_bundle' not in action
    action['arm']+=.5
    np.testing.assert_array_equal(endpoint,np.arange(7,dtype=float))


def test_future_composite_before_splice_resamples_certified_prefix_at_application_time():
    # A proposal is already installed but its suffix has not begun. The
    # physical owner must sample its exact old prefix, not forward prepared q.
    applied=[];sampled=[]
    class Controller:
        commanded_joint_positions=np.zeros(7)
        commanded_joint_velocities=np.zeros(7)
        def preview_joint_step(self,q,*args):
            sampled.append(q.copy());return q+.1,None  # force original retimer rejection
        def apply_action(self,a,**kw):applied.append(a)
    class Physical(FullPickPhysicalMixin):pass
    env=Physical();env.data=SimpleNamespace(time=1.04,qpos=np.zeros(7))
    env.arm_qpos_adrs=np.arange(7);env.control_dt=.02;env.action_type='joint_position'
    env.robot_controller=Controller();env.last_actuator_application_sim_s=1.
    bundle=SimpleNamespace(start=1.2,execution_start=1.,end=4.,lift_time=3.,
        sample_full=lambda t:(np.full(7,t),np.ones(7),np.zeros(7)),
        sample_command_full=lambda t:(np.full(7,t),np.ones(7),np.zeros(7)),command_curves=None)
    env._action_queue=deque([(1.04,dict(arm=np.full(7,99.),gripper=1.,
        _full_pick_bundle=bundle,_full_pick_lift_admitted=False,_full_pick_config=None))])
    env._apply_due_actions()
    assert len(sampled)==1
    np.testing.assert_array_equal(sampled[0],np.full(7,1.04))
    assert env.full_pick_dispatch_fault=='controller_retimer_would_change_certified_clock'
    np.testing.assert_array_equal(applied[0]['arm'],np.zeros(7))


def test_preparing_original_geometry_reuses_private_checker_without_moving_physics():
    from experiments.continuous_intercept_sim.run import make_environment
    from mujoco_conveyor_sorting import ConveyorSortingEnv
    class Physical(FullPickPhysicalMixin,ConveyorSortingEnv):pass
    env=make_environment(1543,.04,.1,-.4,cls=Physical,category_id=0)
    try:
        q,v,ctrl=env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy()
        checker=env.prepare_full_pick_geometry()
        assert checker is env.prepare_full_pick_geometry()
        assert checker.model is not env.model and checker.data is not env.data
        np.testing.assert_array_equal(env.data.qpos,q)
        np.testing.assert_array_equal(env.data.qvel,v)
        np.testing.assert_array_equal(env.data.ctrl,ctrl)
        assert env.data.time==0.
    finally:env.close()
