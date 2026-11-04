"""Physics evidence and independent-clock tests for the isolated comparison."""
import time
import numpy as np

from experiments.continuous_intercept_sim.metrics import contact_summary


def sample(t, *, command=-1., z=.5, left=0., right=0., support=True):
    return {'sim_s': t, 'object_pose': [0.,0.,z,1.,0.,0.,0.],
        'gripper_command': command, 'left_normal_force_n': left,
        'right_normal_force_n': right, 'object_belt_contact': support}


def test_close_command_does_not_count_as_physical_grasp():
    result=contact_summary([sample(0),sample(1)])
    assert result['closure_command_sent']
    assert not result['physical_lift_and_hold']


def test_height_alone_or_supported_contact_does_not_count():
    rows=[sample(0),sample(.3),*[sample(t,z=.55,left=1.,right=1.) for t in (.4,.7,1.1)]]
    assert not contact_summary(rows)['physical_lift_and_hold']
    for row in rows:
        row['right_normal_force_n']=0.
        row['object_belt_contact']=False
    assert not contact_summary(rows)['physical_lift_and_hold']


def test_lift_and_hold_requires_unbroken_bilateral_unsupported_contact():
    rows=[sample(0),sample(.3),*[sample(t,z=.55,left=1.,right=1.,support=False) for t in (.4,.7,1.1)]]
    assert contact_summary(rows)['physical_lift_and_hold']
    rows[3]['right_normal_force_n']=0.
    assert not contact_summary(rows)['physical_lift_and_hold']


def test_physics_progresses_while_policy_owner_is_busy(tmp_path):
    from experiments.continuous_intercept_sim.run import make_environment,PhysicsRunner
    env=make_environment(1543,.04,2.,-.4)
    runner=PhysicsRunner(env,tmp_path,.8)
    try:
        runner.start()
        # This is a deliberate policy-side calculation delay. The independent
        # physics thread must advance contact dynamics and the moving object.
        time.sleep(.4)
        snap=runner.snapshot()
        assert snap['sim_s'] >= .20
        assert snap['object_pose'][0] > -.395
        assert np.linalg.norm(snap['qpos']-env.model.qpos0)>0
        assert not env.grasp_assist
        assert runner.error is None
        assert runner.commands == []
    finally:
        runner.stop()
        env.close()


def test_unbounded_policy_diagnostic_is_recorded_without_aborting_results(tmp_path):
    import json
    from experiments.continuous_intercept_sim.run import write_json
    path=tmp_path/'diagnostic.json'
    write_json(path,{'minimum_unknown':float('inf'),'sample':np.array([float('nan'),1.])})
    assert json.loads(path.read_text()) == {'minimum_unknown':'inf','sample':['nan',1.]}


def test_mirror_grasp_feedback_reads_only_actual_physics_confirmation():
    from experiments.continuous_intercept_sim.run import PlanningMirror
    mirror=PlanningMirror.__new__(PlanningMirror)
    assert not mirror.is_object_grasped(0)
    mirror._physical_grasp_confirmations={0:False,1:True}
    assert not mirror.is_object_grasped(0)
    assert mirror.is_object_grasped(1)
    assert not mirror.is_object_grasped(2)
    mirror._physical_grasp_confirmations={1:False}
    assert not mirror.is_object_grasped(1)


def test_viewer_mutations_cannot_change_physics_or_control(tmp_path):
    from experiments.continuous_intercept_sim.run import make_environment, PhysicsRunner
    from experiments.continuous_intercept_sim.viewer import PhysicsViewer
    env = make_environment(1543, .04, 2., -.4)
    try:
        runner = PhysicsRunner(env, tmp_path, 2.)
        display = PhysicsViewer(runner)
        display._copy_snapshot()
        qpos, ctrl, mass = env.data.qpos.copy(), env.data.ctrl.copy(), env.model.body_mass.copy()
        # MuJoCo GUI interactions can modify the displayed data/model. Those
        # edits must have no route back to contact physics or actuator targets.
        display.data.qpos[0] += .1
        display.data.ctrl[:] = 0.
        display.model.body_mass[:] *= 2.
        assert np.array_equal(env.data.qpos, qpos)
        assert np.array_equal(env.data.ctrl, ctrl)
        assert np.array_equal(env.model.body_mass, mass)
        assert runner.pending is None
        display._copy_snapshot()
        assert np.array_equal(display.data.qpos, qpos)
        assert np.array_equal(display.data.ctrl, ctrl)
    finally:
        env.close()


def test_closed_window_stops_physics_instead_of_leaving_it_running():
    from types import SimpleNamespace
    import threading
    from experiments.continuous_intercept_sim.viewer import PhysicsViewer
    display = PhysicsViewer.__new__(PhysicsViewer)
    display.done = threading.Event()
    display.runner = SimpleNamespace(done=threading.Event())
    display.closed=threading.Event();display.closed.set()
    display.closed_by_user = False
    display._loop()
    assert display.closed_by_user
    assert display.runner.done.is_set()
