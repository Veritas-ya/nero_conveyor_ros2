from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.two_second_grasp.pipeline import close_ramp_action, feed_delay, predicted_loaded_snapshot, matching_drop_candidate
from experiments.two_second_grasp.physics_trace import PhysicsAccelerationTrace
from experiments.two_second_grasp.target_interpolator import SimulationTargetInterpolator
from experiments.two_second_grasp.early_orientation import candidate_orientations, final_orientation_candidates
from experiments.two_second_grasp.early_observation_quality import EarlyObservationQuality, axial_difference


def test_lift_arrival_waits_for_transfer_start_tolerance():
    from experiments.two_second_grasp.hardware_conditions import HardwareConditions
    conditions=HardwareConditions.__new__(HardwareConditions)
    actual=np.zeros(7);actual[6]=.0103
    conditions.env=SimpleNamespace(get_joint_positions=lambda:actual)
    conditions.settings=SimpleNamespace(move_timeout_s=5.)
    conditions.arrivals={}
    target=np.zeros(7)
    # One degree would accept this offset; the transfer start requires .005.
    assert not conditions.arrival('lift',target,0.,.1,tolerance_rad=.005)
    assert not conditions.arrival('lift',target,0.,.3,tolerance_rad=.005)
    actual[6]=.004
    assert not conditions.arrival('lift',target,0.,.4,tolerance_rad=.005)
    assert conditions.arrival('lift',target,0.,.51,tolerance_rad=.005)
    with pytest.raises(ValueError, match='tolerance cannot change'):
        conditions.arrival('lift',target,0.,.52,tolerance_rad=.01)


def test_feed_waits_for_fast_belt_and_keeps_slow_belt_running_upstream():
    assert feed_delay(-1.28,-.55,.20,4.8) == pytest.approx((1.15,3.65))
    delay, travel=feed_delay(-1.28,-.35,.0957,4.8)
    assert delay == 0. and travel > 4.8


def test_close_ramp_is_monotonic_bounded_and_starts_open():
    command=[close_ramp_action(t,.4) for t in np.linspace(0.,.4,41)]
    assert command[0]==pytest.approx(1.) and command[-1]==pytest.approx(-1.)
    assert all(a>=b for a,b in zip(command,command[1:]))
    assert close_ramp_action(0.,0.)==-1.


def test_early_orientations_keep_four_obb_variants_and_prefer_short_axis():
    xs,ys,zs=np.meshgrid(np.linspace(-.04,.04,5),np.linspace(-.015,.015,5),[.5,.52])
    cloud=np.column_stack((xs.ravel(),ys.ravel(),zs.ravel()))
    candidates=candidate_orientations(cloud,[0.,1.,0.,0.])
    assert len(candidates)==4
    assert [row['axis'] for row in candidates]==['short','short','long','long']
    assert all(np.isclose(np.linalg.norm(row['quaternion_wxyz']),1.) for row in candidates)
    assert candidates[0]['rotation_angle_rad']<=candidates[1]['rotation_angle_rad']


def test_early_frame_quality_requires_distinct_stable_uncropped_observations():
    xs,ys,zs=np.meshgrid(np.linspace(-.04,.04,6),np.linspace(-.015,.015,6),[.5,.52,.53])
    cloud=np.column_stack((xs.ravel(),ys.ravel(),zs.ravel()))
    mask=np.zeros((40,40),bool);mask[10:22,10:22]=True
    depth=np.ones((40,40));quality=EarlyObservationQuality()
    for stamp in (0.,.04,.08):
        assert not quality.update(identity=(0,1,1),stamp=stamp,mask=mask,depth=depth,cloud=cloud)['ready']
    assert quality.update(identity=(0,1,1),stamp=.12,mask=mask,depth=depth,cloud=cloud)['ready']
    # A different track must acquire its own frame history.
    assert not quality.update(identity=(0,2,1),stamp=.16,mask=mask,depth=depth,cloud=cloud)['ready']
    border=mask.copy();border[0,12]=True
    assert quality.update(identity=(0,2,1),stamp=.20,mask=border,depth=depth,cloud=cloud)['reason']=='frame_quality'
    assert not quality.update(identity=(0,2,1),stamp=.24,mask=mask,depth=depth,cloud=cloud)['ready']


def test_square_candidates_can_keep_early_heading_at_new_frame_center():
    xs,ys,zs=np.meshgrid(np.linspace(-.02,.02,7),np.linspace(-.02,.02,7),[.5,.52,.53])
    cloud=np.column_stack((xs.ravel()+.3,ys.ravel()-.1,zs.ravel()))
    quat=np.array([0.,np.cos(.32),np.sin(.32),0.])
    candidates=final_orientation_candidates(cloud,quat,early_square=True)
    assert len(candidates)==5
    assert candidates[0]['source']=='stable_early_direction'
    np.testing.assert_allclose(candidates[0]['pose'][:3,3],[.3,-.1,.515],atol=1e-6)
    np.testing.assert_allclose(candidates[0]['pose'][:3,:3],Rotation.from_quat(quat[[1,2,3,0]]).as_matrix())
    assert axial_difference(0.,np.pi/2,np.pi/2)==pytest.approx(0.)
    early=candidate_orientations(cloud,quat)
    assert early[0]['axis']=='hold_current'
    np.testing.assert_allclose(early[0]['quaternion_wxyz'],quat)


def test_nonsquare_later_candidates_prefer_early_direction_and_keep_fallbacks():
    xs,ys,zs=np.meshgrid(np.linspace(-.04,.04,7),np.linspace(-.015,.015,7),[.5,.52,.53])
    cloud=np.column_stack((xs.ravel(),ys.ravel(),zs.ravel()))
    early=candidate_orientations(cloud,[0.,1.,0.,0.])[-1]['quaternion_wxyz']
    candidates=final_orientation_candidates(cloud,early,early_square=False)
    assert len(candidates)==4
    assert candidates[0]['rotation_difference_rad']==pytest.approx(0.,abs=1e-8)
    assert {row['source'] for row in candidates}=={'short','long'}


def test_future_snapshot_preserves_measured_payload_attachment_without_live_mutation():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="arm"><joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".01" mass="1"/><site name="tcp" pos="0 0 .2"/></body>
      <body name="object" pos=".03 0 .2"><freejoint name="free"/>
      <geom type="sphere" size=".01" mass="1"/></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    env=SimpleNamespace(model=model,arm_qpos_adrs=np.array([0]),
        ee_site_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'tcp'))
    runtime=SimpleNamespace(qpos_adr=1)
    original=data.qpos.copy()
    future,pose=predicted_loaded_snapshot(env,original,runtime,np.array([.2]))
    np.testing.assert_array_equal(data.qpos,original)
    np.testing.assert_allclose(future[0],.2)
    np.testing.assert_allclose(pose[:3],[.2,0.,.2])
    np.testing.assert_allclose(future[1:4],[.23,0.,.2])
    np.testing.assert_allclose(future[4:8],[1.,0.,0.,0.])


def test_prefetched_bin_candidate_must_still_be_free_at_the_same_pose():
    pose=[.1,.2,.3,1.,0.,0.,0.]
    assert matching_drop_candidate(pose, [[.9,.2,.3,1.,0.,0.,0.],pose])==1
    assert matching_drop_candidate(pose, [[.101,.2,.3,1.,0.,0.,0.]]) is None
    assert matching_drop_candidate(pose, [[.1,.2,.3,0.,0.,0.,1.]]) is None
    # The two signs represent the same orientation.
    assert matching_drop_candidate(pose, [[.1,.2,.3,-1.,0.,0.,0.]])==0


def test_physics_trace_uses_actual_substep_velocity_change():
    model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep="0.002"/>
      <worldbody><body><joint type="hinge"/><geom type="sphere" size=".01" mass="1"/>
      </body></worldbody></mujoco>''')
    # The production observer expects seven arm coordinates.
    env=SimpleNamespace(model=model,arm_dof_adrs=np.zeros(7,dtype=int))
    data=mujoco.MjData(model)
    data.qfrc_applied[0]=.01
    trace=PhysicsAccelerationTrace(env,lambda:'test')
    before=data.qvel[env.arm_dof_adrs].copy()
    mujoco.mj_step(model,data)
    expected=(data.qvel[0]-before[0])/model.opt.timestep
    trace(model,data,before)
    report=trace.report()
    assert report['substeps']==1
    assert report['per_joint_absolute_peaks']['velocity_difference'][0]['value_rad_s2']==pytest.approx(expected)
    assert report['per_joint_absolute_peaks']['velocity_difference'][0]['phase']=='test'


def test_report_distinguishes_recomputed_qacc_from_substep_velocity_difference():
    from experiments.two_second_grasp.simulation_report import summarize
    rows=[{'sim_s':t,'clipped':False,'applied_qd_rad_s':[0.]*7,
           'actual_qdd_rad_s2':[69.]*7} for t in (0.,.01)]
    trace={'physics_timestep_s':.002,'per_joint_absolute_peaks':{
        'velocity_difference':[{'value_rad_s2':-12.} for _ in range(7)]}}
    summary=summarize({'command_rows':rows,'physics_substep_trace':trace})
    assert summary['peak_post_forward_qacc_rad_s2']==[69.]*7
    assert summary['peak_physics_substep_velocity_difference_rad_s2']==[12.]*7
    assert summary['physics_substep_timestep_s']==.002


def test_sim_inner_target_interpolates_accepted_target_across_substeps():
    ctrl=np.zeros(7)
    env=SimpleNamespace(data=SimpleNamespace(ctrl=ctrl,time=0.),control_dt=.01,
        model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
        robot_controller=SimpleNamespace(actuator_ids=np.arange(7),_last_joint_target=np.zeros(7)))
    interpolator=SimulationTargetInterpolator(env)
    env.robot_controller._last_joint_target=np.ones(7)
    values=[]
    for i in range(5):
        env.data.time=i*.002
        interpolator.apply()
        values.append(env.data.ctrl[0])
    np.testing.assert_allclose(values,[.2,.4,.6,.8,1.])
