from types import SimpleNamespace
import numpy as np
from experiments.continuous_intercept_sim.sorting_cycles import SortingCycles
from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselineSortingPolicy
from experiments.continuous_intercept_sim.full_preplanned import TargetGeometry


def test_feed_once_after_normal_return_not_release_or_timeout():
    book=SortingCycles(2);feeds=[]
    for stage,target,reason in [('closing',0,''),('releasing',0,''),('returning',0,''),('waiting',None,'return_timeout')]:
        assert not book.observe(stage,target,1.,lambda:feeds.append(1),reason=reason)
    assert not feeds and not book.returns
    book.observe('returning',1,2.,lambda:feeds.append(1),reason='')
    assert not book.observe('waiting',None,3.,lambda:feeds.append(1),reason='wait_pose_reached')
    book.observe('waiting',None,4.,lambda:feeds.append(1),reason='wait_pose_reached')
    assert len(feeds)==1 and len(book.returns)==1
    book.observe('returning',2,5.,lambda:feeds.append(1),reason='')
    assert book.observe('waiting',None,6.,lambda:feeds.append(1),reason='wait_pose_reached')
    assert len(feeds)==1 and len(book.returns)==2


def test_next_target_clears_previous_full_certificate_without_resetting_dispatch():
    policy=BaselineSortingPolicy();marker=object()
    policy._full_pick_bundle=marker;policy._full_pick_contact_handed_off=True
    policy._full_pick_abort_reason='previous';policy._full_pick_hold_q=np.ones(7)
    dispatch=SimpleNamespace(commanded_joint_positions=np.arange(7.))
    policy.env=SimpleNamespace(robot_controller=dispatch)
    policy._clear_target()
    assert policy._full_pick_bundle is None and not policy._full_pick_contact_handed_off
    assert policy._full_pick_abort_reason is None and policy._full_pick_hold_q is None
    np.testing.assert_array_equal(dispatch.commanded_joint_positions,np.arange(7.))


def test_recycled_slot_registers_new_identity_without_reusing_collision_conclusions():
    from test_full_pick_physics_adapter import geometry,request
    env,checker=geometry();old=request(env);new=request(env)
    env._objects[0].logical_id=11;new.target_id=11;new.request_id+=1
    checker.register_targets(env)
    checker.offsets[11]=np.eye(4)
    assert checker(new,np.zeros(1))
    checker.bind_static_request(new);new.planned_target_time_s=None
    assert checker(new,np.zeros(1))
    old.planned_target_time_s=None
    assert not checker(old,np.zeros(1))


def test_unchanged_physics_snapshot_is_not_copied_while_waiting():
    import threading
    from experiments.continuous_intercept_sim.run import PhysicsRunner
    class Payload:
        def __deepcopy__(self,memo):raise AssertionError('unchanged frame copied')
    runner=PhysicsRunner.__new__(PhysicsRunner);runner.lock=threading.Lock()
    runner.last={'sim_s':2.,'payload':Payload()}
    assert runner.snapshot(after_sim=2.) is None
    runner.last={'sim_s':2.02,'payload':np.arange(3.)}
    new=runner.snapshot(after_sim=2.)
    assert new['sim_s']==2.02
    new['payload'][0]=99
    assert runner.last['payload'][0]==0


def test_new_factory_identity_clears_old_appointment_but_same_id_preserves_policy_state(tmp_path):
    from experiments.continuous_intercept_sim.run import make_environment,PlanningMirror,PhysicsRunner
    physical=make_environment(1701,.2,130.,-1.28,category_id=0,cycles=2)
    mirror=make_environment(1701,.2,130.,-1.28,cls=PlanningMirror,category_id=0,cycles=2)
    try:
        old=next(r for r in physical._objects if r.logical_id==0)
        copied=next(r for r in mirror._objects if r.logical_id==0)
        copied.planned_grasp_time=7.23;copied.planned_grasp_pose=np.ones(7)
        copied.selected_at=3.26;copied.failure_reason='old'
        physical._deactivate_object(old)
        assert physical.spawn_object(category_id=0,x=-1.28)==1
        runner=PhysicsRunner(physical,tmp_path,.1,video=False)
        mirror.policy=SimpleNamespace(_grasp_target_pose=lambda *a,**k:None)
        mirror.adopt_snapshot(runner.snapshot())
        assert copied.logical_id==1 and copied.planned_grasp_time is None
        assert copied.planned_grasp_pose is None and copied.selected_at is None
        assert copied.failure_reason is None
        copied.state='PLANNING';copied.planned_grasp_time=60.
        mirror.adopt_snapshot(runner.snapshot())
        assert copied.state=='PLANNING' and copied.planned_grasp_time==60.
    finally:
        mirror.close();physical.close()


def test_sorted_body_is_not_recycled_before_independent_stability(tmp_path):
    from experiments.continuous_intercept_sim.run import make_environment,PhysicsRunner
    env=make_environment(1701,.2,130.,-1.28,category_id=0,cycles=2)
    try:
        runner=PhysicsRunner(env,tmp_path,.1,video=False,evaluate_sorting=True)
        runtime=next(r for r in env._objects if r.logical_id==0)
        runtime.state='SORTED';runtime.state_changed_at=0.;env.data.time=4.
        original=env.data.qpos.copy()
        env.recycle_terminal_objects()
        assert runtime.logical_id==0
        np.testing.assert_array_equal(env.data.qpos,original)
        runner.sorting_evaluator.records[0]['success']=True
        env.recycle_terminal_objects()
        assert runtime.logical_id is None
        runner.sorting_evaluator.close()
    finally:env.close()



def test_random_factory_uses_original_reproducible_deck_and_each_objects_own_shape():
    from experiments.continuous_intercept_sim.run import make_environment
    env=make_environment(1701,.2,900.,-1.28,category_id=None,spawn_y=.09,spawn_yaw=-1.2,cycles=10,random_objects=True)
    try:
        expected=np.resize(env._automatic_category_manifest,10).tolist();actual=[];widths=[]
        for index in range(10):
            runtime=next(r for r in env._objects if r.logical_id is not None)
            actual.append(int(runtime.category_id));widths.append(float(runtime.metadata['recommended_grip_width']))
            assert runtime.logical_id==index
            assert runtime.bin_id==env._object_spec_by_category[runtime.category_id].bin_id
            np.testing.assert_allclose(env.data.qpos[runtime.qpos_adr:runtime.qpos_adr+2],[-1.28,.09])
            assert all(env.model.geom_contype[g]==env._base_geom_properties[g]['contype'] for g in runtime.geom_ids)
            env._deactivate_object(runtime)
            if index<9:assert env.spawn_object(x=-1.28)==index+1
        assert actual==expected and len(set(actual))==len(env.active_category_ids) and len(set(widths))>1
        assert not env.grasp_assist and env._current_belt_speed==.2
    finally:env.close()


def test_new_class_restores_its_own_collision_shape_mass_and_display_color(tmp_path):
    from experiments.continuous_intercept_sim.run import make_environment,PlanningMirror,PhysicsRunner
    physical=make_environment(1701,.2,90.,-1.28,category_id=0,cycles=2)
    mirror=make_environment(1701,.2,90.,-1.28,cls=PlanningMirror,category_id=0,cycles=2)
    try:
        checker=TargetGeometry(mirror)
        old=next(r for r in physical._objects if r.logical_id==0)
        physical._deactivate_object(old)
        assert physical.spawn_object(category_id=9,x=-1.28)==1
        current=physical._logical_to_slot[1]
        runner=PhysicsRunner(physical,tmp_path,.1,video=False)
        mirror.policy=SimpleNamespace(_grasp_target_pose=lambda *a,**k:None)
        mirror.adopt_snapshot(runner.snapshot());checker.register_targets(mirror)
        copied=mirror._logical_to_slot[1]
        assert copied.category_id==9 and checker.targets[1][0]==current.qpos_adr
        assert mirror.model.body_mass[copied.body_id]==physical.model.body_mass[current.body_id]
        for geom in copied.geom_ids:
            assert mirror.model.geom_contype[geom]==physical.model.geom_contype[geom]>0
            assert checker.model.geom_contype[geom]==physical.model.geom_contype[geom]
            assert runner.snapshot()['geom_rgba'][geom,3]>0
        assert runner.rows[-1]['object_id']==1
    finally:mirror.close();physical.close()


def test_random_cli_rejects_fixed_objects_grasp_before_creating_output(tmp_path):
    import pytest
    from experiments.continuous_intercept_sim.run import main
    for extra in (['--category-id','0'],['--network-grasp-replay','other_object.json']):
        out=tmp_path/'never_created'
        with pytest.raises(SystemExit) as failure:main(['--random-objects','--output',str(out),*extra])
        assert failure.value.code==2 and not out.exists()
