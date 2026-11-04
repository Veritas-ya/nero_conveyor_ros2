import numpy as np
import pytest
from nero_core.two_second.synchronized_reference import SynchronizedIntercept, schedule_forward_join


def make_reference(*, start=None, velocity=None):
    return SynchronizedIntercept(capture_s=0., start_s=4., duration_s=4.,
        capture_tcp_world=np.array([-.55, .03, .55]),
        velocity_world=np.array([.058, 0., 0.]) if velocity is None else velocity,
        start_tcp_world=np.array([.04, 0., .70]) if start is None else start,
        start_quaternion_wxyz=np.array([1.,0.,0.,0.]),
        target_quaternion_wxyz=np.array([1.,0.,0.,0.]),
        follow_s=.8, brake_s=1.5, prealign_s=2.)


def test_sync_segment_keeps_constant_object_offset_and_equal_velocity():
    ref = make_reference()
    for t in np.linspace(ref.prealign_s, ref.duration_s, 25):
        pos, vel, acc = ref.position_derivatives(t)
        target = ref.capture_tcp_world+ref.velocity_world*(ref.start_s+t-ref.capture_s)
        np.testing.assert_allclose(pos,target,atol=1e-12)
        np.testing.assert_allclose(vel,ref.velocity_world,atol=1e-12)
        np.testing.assert_allclose(acc,0.,atol=1e-12)
    np.testing.assert_allclose(ref.sample(ref.duration_s)[0],ref.contact_xyz)


@pytest.mark.parametrize('sync_duration', [0., .2, .5])
def test_short_sync_and_closure_follow_are_distinct(sync_duration):
    from dataclasses import replace
    ref = replace(make_reference(), duration_s=2.+sync_duration)
    assert ref.duration_s-ref.prealign_s == pytest.approx(sync_duration)
    assert ref.follow_s == pytest.approx(.8)
    for t in np.linspace(ref.prealign_s, ref.duration_s+ref.follow_s, 41):
        pos, vel, acc = ref.position_derivatives(t)
        target = ref.capture_tcp_world+ref.velocity_world*(ref.start_s+t-ref.capture_s)
        np.testing.assert_allclose(pos,target,atol=1e-12)
        np.testing.assert_allclose(vel,ref.velocity_world,atol=1e-12)
        np.testing.assert_allclose(acc,0.,atol=1e-12)


def test_closure_at_join_end_preserves_position_velocity_and_acceleration():
    from dataclasses import replace
    ref=replace(make_reference(),duration_s=2.)
    assert ref.duration_s == ref.prealign_s
    for boundary in (ref.duration_s, ref.follow_end_s):
        before=ref.position_derivatives(boundary-1e-7)
        after=ref.position_derivatives(boundary+1e-7)
        for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=1e-5)
    schedule=schedule_forward_join(capture_s=0.,grasp_position=[-.4,0.,.5],velocity=[.058,0.,0.],
        standby_position=[-.1,0.,.7],direction=[1,0,0],now_s=.1,planning_reserve_s=1.,
        advance_s=.1,join_s=2.,synchronized_s=0.,follow_s=.4,follow_bounds=[-.2,.28])
    assert schedule['contact_s'] == pytest.approx(schedule['sync_start_s'])


def test_rotate_first_holds_tcp_position_and_joins_smoothly():
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    target=Rotation.from_euler('z',45,degrees=True).as_quat()[[3,0,1,2]]
    ref=replace(make_reference(), duration_s=4., prealign_s=4., rotate_first_s=2.,
                target_quaternion_wxyz=target)
    for t in np.linspace(0.,2.,11):
        xyz, v, a=ref.position_derivatives(t)
        np.testing.assert_allclose(xyz,ref.start_tcp_world,atol=1e-12)
        np.testing.assert_allclose(v,0.,atol=1e-12)
        np.testing.assert_allclose(a,0.,atol=1e-12)
    np.testing.assert_allclose(np.abs(np.dot(ref.sample(2.)[1],target)),1.,atol=1e-12)
    np.testing.assert_allclose(np.abs(np.dot(ref.sample(3.)[1],target)),1.,atol=1e-12)
    for boundary in (2.,4.,ref.follow_end_s):
        for a,b in zip(ref.position_derivatives(boundary-1e-7),ref.position_derivatives(boundary+1e-7)):
            np.testing.assert_allclose(a,b,atol=1e-5)


@pytest.mark.parametrize('boundary', [2.,4.,4.8])
def test_join_follow_and_brake_are_continuous_in_position_velocity_acceleration(boundary):
    ref=make_reference()
    before=ref.position_derivatives(boundary-1e-7)
    after=ref.position_derivatives(boundary+1e-7)
    for a,b in zip(before,after):np.testing.assert_allclose(a,b,atol=1e-5)


def test_downstream_waiting_pose_still_requires_upstream_connection():
    ref=make_reference()
    synchronized_start=ref.position_derivatives(ref.prealign_s)[0]
    assert synchronized_start[0] < ref.start_tcp_world[0]
    velocities=[ref.position_derivatives(t)[1][0] for t in np.linspace(0.,2.,101)]
    assert min(velocities) < 0
    assert np.allclose([ref.position_derivatives(t)[1][0] for t in np.linspace(2.,4.,21)],.058)


def test_schedule_waits_for_object_and_keeps_join_forward():
    start=np.array([.0405077,0.,.7]);grasp=np.array([-.5494306,.03,.55535]);v=np.array([.058,0.,0.])
    schedule=schedule_forward_join(capture_s=137.902,grasp_position=grasp,velocity=v,
        standby_position=start,direction=[1,0,0],now_s=138.2,planning_reserve_s=2.5,
        advance_s=.1,join_s=2.,synchronized_s=2.,follow_s=.8,follow_bounds=[-.2,.28])
    ref=SynchronizedIntercept(137.902,schedule['join_start_s'],4.,grasp,v,start,
        np.array([1.,0.,0.,0.]),np.array([1.,0.,0.,0.]),.8,1.5,2.)
    assert min(ref.position_derivatives(t)[1][0] for t in np.linspace(0.,4.,401)) >= -1e-10
    assert schedule['contact_coordinate_m'] == pytest.approx(.2145077)
    assert schedule['follow_end_coordinate_m'] == pytest.approx(.2609077)
    assert schedule['join_start_s']-138.2 >= 2.6


def test_schedule_rejects_when_follow_cannot_fit():
    with pytest.raises(ValueError,match='do not fit'):
        schedule_forward_join(capture_s=0.,grasp_position=[0.,0.,0.],velocity=[.1,0.,0.],
            standby_position=[.2,0.,0.],direction=[1,0,0],now_s=0.,planning_reserve_s=1.,
            advance_s=.1,join_s=2.,synchronized_s=2.,follow_s=.8,follow_bounds=[-.2,.28])


@pytest.mark.parametrize('speed,expected', [(.0957, 2.), (.2, 1.5)])
def test_speed_and_window_select_join_duration_together(speed, expected):
    valid = []
    for join in (2., 1.75, 1.5, 1.25):
        try:
            schedule = schedule_forward_join(capture_s=0., grasp_position=[-.55,0.,.55],
                velocity=[speed,0.,0.], standby_position=[.0405077,0.,.7], direction=[1,0,0],
                now_s=.05, planning_reserve_s=1.5, advance_s=.1, join_s=join,
                synchronized_s=0., follow_s=.4, follow_bounds=[-.2,.28])
            valid.append((join, schedule))
        except ValueError:
            continue
    join, schedule = valid[0]
    assert join == expected
    assert schedule['join_start_s'] >= 1.65 - 1e-9
    assert -.2 <= schedule['contact_coordinate_m'] <= schedule['follow_end_coordinate_m'] <= .28


def test_synchronized_admission_accepts_late_entry_but_rejects_old_and_far_tracks():
    from nero_core.two_second.reference import EntryNotReady
    from nero_core.two_second.synchronized_reference import admit_synchronized_observation
    # The old entry gate requires 3.5 seconds before the upstream boundary.
    # Here the target is already inside but a downstream schedule may fit.
    assert admit_synchronized_observation(10., -.15, .05, [-.2,.28], 10.1, .35) == pytest.approx(9.)
    with pytest.raises(EntryNotReady, match='too far upstream'):
        admit_synchronized_observation(10., -.6, .05, [-.2,.28], 10.1, .35)
    with pytest.raises(EntryNotReady, match='passed'):
        admit_synchronized_observation(10., .27, .05, [-.2,.28], 11., .35)


def test_delayed_freeze_tracks_until_ten_centimetres_before_entry():
    from nero_core.two_second.reference import EntryNotReady
    from nero_core.two_second.synchronized_reference import admit_synchronized_observation
    with pytest.raises(EntryNotReady, match='too far upstream'):
        admit_synchronized_observation(10., -.301, .0415, [-.2,.28], 10.05, .10)
    entry = admit_synchronized_observation(10., -.299, .0415, [-.2,.28], 10.05, .10)
    assert entry == pytest.approx(10.+.099/.0415)


def test_adaptive_freeze_advances_with_speed_and_stays_bounded():
    from types import SimpleNamespace
    from nero_core.two_second.synchronized_reference import freeze_window_for_speed
    cfg=SimpleNamespace(follow_bounds=np.array([-.2,.28]), settings=SimpleNamespace(full_plan_close_window_s=.8),
        raw={'freeze_upstream_distance_m':.1,'synchronized_adaptive_freeze':True,
             'synchronized_max_freeze_upstream_distance_m':.35,'maximum_observation_age_s':1.,
             'planning_reserve_s':2.5,'command_advance_s':.1,'prealign_orientation_s':2.,
             'synchronized_duration_s':.2})
    slow=freeze_window_for_speed(cfg,.0415)
    medium=freeze_window_for_speed(cfg,.09563)
    fast=freeze_window_for_speed(cfg,.11102)
    assert slow['distance_m'] == pytest.approx(.1)
    assert medium['distance_m'] == pytest.approx(.151158)
    assert fast['distance_m'] == pytest.approx(.252732)
    assert fast['budget_horizon_s'] == pytest.approx(6.6)
    capped=freeze_window_for_speed(cfg,.3)
    assert capped['distance_m'] == pytest.approx(.35) and capped['capped']


def test_hardware_planner_never_falls_back_to_closer_long_axis():
    from scipy.spatial.transform import Rotation
    from nero_core.two_second.smooth_planning import order_obb_grasps
    poses=[]
    for angle in (90., 270., 0., 180.):
        pose=np.eye(4)
        pose[:3,:3]=Rotation.from_euler('z',angle,degrees=True).as_matrix()
        poses.append(pose)
    synchronized=order_obb_grasps(poses,np.eye(3),synchronized=True)
    classic=order_obb_grasps(poses,np.eye(3),synchronized=False)
    assert {index for index,_ in synchronized} == {0,1}
    assert {index for index,_ in classic} == {0,1}
