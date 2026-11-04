from types import SimpleNamespace
import numpy as np
import pytest

from experiments.graspgenx_sim.retry_candidates import (rebuild_after_rejection, evaluated_variants,
    conveyor_world_x_axis_in_base, remaining_window_check)
from experiments.graspgenx_sim import candidates
from nero_core.measured_motion import MeasuredMotionSession
from test_measured_motion_session import IDENTITY, detection, report


def item(seq, x):
    stamp = float(seq)
    return SimpleNamespace(detection=detection(x), capture_s=stamp, sequence=seq,
        clock_epoch=IDENTITY.clock_epoch, velocity_measurement=report(seq, stamp))


def publish(session, entry):
    return session.update(entry.detection, identity=IDENTITY, timestamp=entry.capture_s,
        sequence=entry.sequence, clock_epoch=entry.clock_epoch,
        source_timestamp=entry.capture_s, velocity_measurement=entry.velocity_measurement)


class ShadowContext:
    def __init__(self): self.closed = False
    def close(self): self.closed = True


def test_replay_retires_old_task_and_tracks_next_candidate():
    old = MeasuredMotionSession(IDENTITY, max_coast_s=1.)
    first, later = item(1, 0.), item(2, .1)
    assert publish(old, first) and publish(old, later)
    context = ShadowContext()
    new = rebuild_after_rejection(context, old, IDENTITY, first, [first, later], 2.,
        'full pick approach failed complete geometry/TCP checks')
    assert context.closed and old.snapshot(2.) is None
    selected = {**first.detection, 'network_candidate': {'source_index': 2}, 'network_provenance': {},
        'network_capture_center_base_m': np.median(first.detection['object_points'], axis=0),
        'network_capture_grasp_base': np.eye(4)}
    bound = new.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    assert bound['grasp'][0, 3] == pytest.approx(.1)
    assert new.snapshot(2.).velocity_source == 'nero7_xy_kalman'
    with pytest.raises(ValueError, match='inactive'):
        old.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)


def test_cancel_expiry_or_non_geometric_failure_never_retries():
    first = item(1, 0.)
    old = MeasuredMotionSession(IDENTITY, max_coast_s=1.)
    assert publish(old, first)
    context = ShadowContext()
    with pytest.raises(ValueError, match='not eligible'):
        rebuild_after_rejection(context, old, IDENTITY, first, [], 1., 'full_pick_delivery_missed_start')
    assert not context.closed
    with pytest.raises(ValueError, match='expired'):
        rebuild_after_rejection(context, old, IDENTITY, first, [], 4.,
            'full pick closing failed complete geometry/TCP checks')
    old.cancel()
    with pytest.raises(ValueError, match='cancelled'):
        rebuild_after_rejection(context, old, IDENTITY, first, [], 1.,
            'full pick closing failed complete geometry/TCP checks')


def test_budget_retry_preserves_explicit_frozen_tracker_and_requires_opt_in():
    old=MeasuredMotionSession(IDENTITY,max_coast_s=None,max_prediction_horizon_s=None)
    first,later=item(1,0.),item(2,.1)
    assert publish(old,first) and publish(old,later)
    with pytest.raises(ValueError,match='not eligible'):
        rebuild_after_rejection(ShadowContext(),old,IDENTITY,first,[later],5.,
            'planning_candidate_budget_exceeded')
    replacement=rebuild_after_rejection(ShadowContext(),old,IDENTITY,first,[later],5.,
        'planning_candidate_budget_exceeded',allow_budget_retry=True)
    assert replacement.tracker.max_coast_s is None
    assert replacement.tracker.predictor.max_prediction_horizon is None
    assert replacement.snapshot(5.).valid


def test_replay_rejects_changed_identity_and_clock():
    old = MeasuredMotionSession(IDENTITY, max_coast_s=1.)
    first = item(1, 0.)
    assert publish(old, first)
    bad = item(2, .1)
    bad.clock_epoch = 5
    with pytest.raises(ValueError, match='epoch changed'):
        rebuild_after_rejection(ShadowContext(), old, IDENTITY, first, [bad], 1.,
            'full pick approach failed complete geometry/TCP checks')


def test_bounded_batches_keep_unexamined_variants_and_score_order(monkeypatch):
    monkeypatch.setattr(candidates, '_assets', lambda *_: (np.eye(4), np.zeros((1,3)), {}))
    good_ik = SimpleNamespace(success=True, joint_positions=np.zeros(7), position_error=0.,
                              orientation_error=0., iterations=1, message='ok')
    env = SimpleNamespace(belt_height=0., scene_config={'workspace': {}}, robot_config={},
        get_joint_positions=lambda: np.zeros(7), solve_ik_result=lambda *a, **k: good_ik,
        is_arm_configuration_safe=lambda *a, **k: True)
    poses = np.repeat(np.eye(4)[None,:,:], 2, axis=0)
    poses[:,2,3] = .1
    config = {'maximum_ik_candidates': 1, 'maximum_accepted_candidates': 1,
              'maximum_approach_tilt_deg': None}
    first, rejected = candidates.select_candidates(env, poses, [.9,.8], [0,0,0], config)
    assert [(row['source_index'], row['variant']) for row in first] == [(0, 'original')]
    examined = evaluated_variants(first,rejected)
    assert (0,'parallel_jaw_z_pi') not in examined and (1,'original') not in examined
    second, rejected2 = candidates.select_candidates(env, poses, [.9,.8], [0,0,0], config,
                                                       skip_variants=examined)
    assert [(row['source_index'], row['variant']) for row in second] == [(0, 'parallel_jaw_z_pi')]
    examined |= evaluated_variants(second,rejected2)
    third, _ = candidates.select_candidates(env, poses, [.9,.8], [0,0,0], config,
                                             skip_variants=examined)
    assert [(row['source_index'], row['variant']) for row in third] == [(1, 'original')]


def test_retry_models_keep_independent_fixed_start_limits():
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, PickRejected
    from mujoco_conveyor_sorting.runtime.nero_policy_context import _copy_geometry
    original = RobotModel()
    prototype = _copy_geometry(original)
    first = _copy_geometry(prototype)
    second = _copy_geometry(prototype)
    q1 = np.zeros(7)
    q2 = q1.copy(); q2[0] = .01
    first.configure_start_joint_tolerance(q1)
    with pytest.raises(PickRejected, match='起始关节容差已固定'):
        first.configure_start_joint_tolerance(q2)
    second.configure_start_joint_tolerance(q2)
    assert prototype._start_joint_tolerance_configuration is None
    np.testing.assert_allclose(second.nominal_limits, first.nominal_limits)


def test_world_grasp_bounds_stay_in_world_coordinates_after_base_transform():
    transform=np.eye(4)
    transform[:3,:3]=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    transform[:3,3]=[.3,-.4,.1]
    direction,origin=conveyor_world_x_axis_in_base(transform)
    world_point=np.array([.17,.22,.5])
    base_point=transform[:3,:3] @ world_point+transform[:3,3]
    assert (base_point-origin) @ direction == pytest.approx(world_point[0])
    invalid=transform.copy();invalid[0,0]=2
    with pytest.raises(ValueError,match='rigid'):
        conveyor_world_x_axis_in_base(invalid)


def test_remaining_window_rejects_impossible_schedule_without_relaxing_gate():
    config=SimpleNamespace(full_plan_compute_budget_s=2.,trajectory_minimum_duration=.1,
        descent_lead_time=.8,intercept_deadline_margin=.25)
    policy=SimpleNamespace(config=config,_execution_dt=lambda _: .02)
    obs={'timestamp':1., 'predictions':{'part':{'grasp_window':{'feasible':True,'end_time':5.}}}}
    ready=remaining_window_check(obs,'part',policy)
    assert ready['available'] and ready['minimum_required_s']==pytest.approx(3.17)
    assert not remaining_window_check(obs,'part',policy,budget_override=3.)['available']
    obs['predictions']['part']['grasp_window']['end_time']=1.2
    assert remaining_window_check(obs,'part',policy)['reason']=='insufficient_time_for_minimum_certified_schedule'


def test_late_certificate_retry_requires_explicit_opt_in():
    captured=item(1,0.)
    old=MeasuredMotionSession(IDENTITY,max_coast_s=1.)
    assert publish(old,captured)
    ctx=ShadowContext()
    new=rebuild_after_rejection(ctx,old,IDENTITY,captured,[],1.,
        'full_pick_delivery_missed_start',allow_late_delivery=True)
    assert ctx.closed and old.snapshot(1.) is None and new.snapshot(1.).valid
