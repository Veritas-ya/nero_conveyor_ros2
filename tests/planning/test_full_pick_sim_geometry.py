"""Native geometry-only update must retain full-forward contact results."""
import mujoco
import numpy as np
from types import SimpleNamespace
from mujoco_conveyor_sorting.controllers.planning_worker import MuJoCoGlobalPlanningEngine
from experiments.continuous_intercept_sim.run import make_environment
from experiments.continuous_intercept_sim.full_preplanned import TargetGeometry
from experiments.continuous_intercept_sim.full_preplanned import finish_contact_lift_experiment,completed_lift_hold
import pytest

def contacts(data):
    return sorted((int(c.geom1),int(c.geom2),round(float(c.dist),12))
        for c in data.contact[:data.ncon])

def test_geometry_update_matches_forward_on_varied_arm_and_workpiece_positions():
    env=make_environment(1543,.2,18.,-1.28)
    try:
        geometry=TargetGeometry(env)
        reference=mujoco.MjData(geometry.model)
        rng=np.random.default_rng(9831)
        initial=env.data.qpos.copy()
        obj=next(r for r in env._objects if r.logical_id is not None)
        collision_cases=0
        for index in range(120):
            q=initial.copy()
            q[env.arm_qpos_adrs]+=rng.uniform(-.4,.4,7)
            q[obj.qpos_adr:obj.qpos_adr+3]=[rng.uniform(-.22,.3),rng.uniform(-.12,.12),rng.uniform(.49,.7)]
            q[7:9]=[.01*(index%6),-.01*(index%6)]
            geometry.data.qpos[:]=q;reference.qpos[:]=q
            geometry._update_geometry();mujoco.mj_forward(geometry.model,reference)
            np.testing.assert_allclose(geometry.data.geom_xpos,reference.geom_xpos,atol=1e-12)
            np.testing.assert_allclose(geometry.data.site_xmat,reference.site_xmat,atol=1e-12)
            assert contacts(geometry.data)==contacts(reference)
            flags=geometry.target_geom_masks[obj.logical_id]
            c=geometry.data.contact;n=geometry.data.ncon
            vector=np.flatnonzero((c.dist[:n]<=1e-6)&(flags[c.geom1[:n]]|flags[c.geom2[:n]]))
            scalar=[i for i,v in enumerate(c[:n]) if v.dist<=1e-6
                and (flags[v.geom1] or flags[v.geom2])]
            assert vector.tolist()==scalar
            collision_cases+=any(float(c.dist)<0 for c in reference.contact[:reference.ncon])
        assert collision_cases>0
    finally:env.close()

def test_redundant_joint_speed_keeps_original_cartesian_translation_and_rotation():
    env=make_environment(1543,.2,18.,-1.28)
    try:
        engine=MuJoCoGlobalPlanningEngine(env.model,joint_names=env.arm_joint_names,
            ee_site_name=env.ee_site_name,ik_config=env.robot_config.get('ik',{}),
            robot_collision_body_ids=tuple(env._robot_collision_body_ids))
        q=env.data.qpos[env.arm_qpos_adrs].copy()
        request=SimpleNamespace(continuation_velocity_m_s=(.2,0.,0.))
        base=engine._full_terminal_velocity(request,q)
        candidate=engine._full_terminal_velocity(request,q,nullspace_speed=1.)
        jacobian=engine.ik.compute_jacobian(q)
        np.testing.assert_allclose(jacobian@candidate,np.r_[.2,0.,0.,0.,0.,0.],atol=1e-10)
        np.testing.assert_allclose(jacobian@(candidate-base),np.zeros(6),atol=1e-10)
        assert abs(np.linalg.norm(candidate-base)-1.)<1e-10
    finally:env.close()

@pytest.mark.parametrize('handed,stage,cleared,abort,expected',[
    (True,'lifting',True,None,True),(True,'lifting',False,None,False),
    (True,'closing',False,None,False),(False,'lifting',True,None,False),
    (True,'lifting',True,'fault',False),(True,'transferring',False,None,True)])
def test_single_pick_scope_ends_only_after_original_measured_lift_checks(handed,stage,cleared,abort,expected):
    q=np.arange(7,dtype=float);events=[]
    policy=SimpleNamespace(_full_pick_contact_handed_off=handed,_full_pick_abort_reason=abort,
        _lift_clearance_complete=cleared,_full_pick_finished=False,stage=SimpleNamespace(value=stage),
        _full_pick_note=lambda *a,**k:events.append((a,k)),_grasp_gripper_action=-.7,
        _full_pick_bundle=SimpleNamespace(lift=SimpleNamespace(q1=-q)))
    mirror=SimpleNamespace(data=SimpleNamespace(time=8.4),robot_controller=SimpleNamespace(commanded_joint_positions=q))
    assert finish_contact_lift_experiment(policy,mirror)==expected
    assert policy._full_pick_finished==expected
    if expected:
        np.testing.assert_array_equal(completed_lift_hold(policy)['arm'],q)
        assert events[0][1]['independent_retention_score_pending']

@pytest.mark.parametrize('category',[6,9,12,14])
def test_explicit_spawn_uses_original_factory_and_requested_category(category):
    env=make_environment(1543,.2,18.,-1.28,category_id=category,spawn_y=.06,spawn_yaw=.2)
    try:
        active=[r for r in env._objects if r.logical_id is not None]
        assert len(active)==1 and active[0].category_id==category
        position=env.data.qpos[active[0].qpos_adr:active[0].qpos_adr+7]
        np.testing.assert_allclose(position[:2],[-1.28,.06])
        np.testing.assert_allclose(position[3:],[np.cos(.1),0.,0.,np.sin(.1)])
        assert not env.grasp_assist
    finally:env.close()
