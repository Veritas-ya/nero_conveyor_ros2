"""Actual Nero7 tracker with fake observations; no device or physics claim."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nero_core.measured_motion import MeasuredMotionSession
from mujoco_conveyor_sorting.runtime.types import TargetIdentity, RGBDFrame, CameraIntrinsics, SampleHeader

IDENTITY = TargetIdentity('s', 'o', 't', 1., 4, track_id=7, track_generation=2)

def detection(x):
    pose = np.eye(4); pose[0, 3] = x
    return {'target_identity': IDENTITY, 'grasp': pose, 'height': .03, 'width': .04,
            'object_points': np.array([[x, 0., .01], [x, .01, .02]])}

def report(seq, stamp):
    return {'clock_epoch': 4, 'geometry_sequence': seq, 'geometry_device_timestamp': stamp,
            'geometry_host_capture_timestamp': stamp, 'speed_valid': True,
            'motion_source': 'nero7_xy_kalman', 'velocity_base_m_s': [.1, 0., 0.],
            'device_timestamp': stamp, 'host_capture_timestamp': stamp, 'sequence': seq,
            'track_id': 7, 'track_generation': 2, 'sample_count': 5, 'span_s': .4,
            'speed_std_m_s': .001}

def update(session, seq, x):
    t = float(seq)
    return session.update(detection(x), identity=IDENTITY, timestamp=t, sequence=seq,
                          clock_epoch=4, source_timestamp=t, velocity_measurement=report(seq,t))

def test_network_offset_tracks_new_observation_and_retains_actual_time():
    s = MeasuredMotionSession(IDENTITY, max_coast_s=None, max_prediction_horizon_s=None)
    assert update(s, 1, 0.)
    initial = detection(0.)
    grasp = np.eye(4); grasp[:3, 3] = [.02, 0., .03]
    selected = {**initial, 'network_candidate': {'score': .8},
                'network_capture_grasp_base': grasp, 'network_capture_center_base_m': [0., .005, .015],
                'network_provenance': {}}
    assert update(s, 2, .1)
    bound = s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    np.testing.assert_allclose(bound['grasp'][:3, 3], [.12, 0., .03], atol=1e-8)
    assert update(s, 3, .2)
    snap = s.snapshot(3.5, robot_arrival_time=.2)
    np.testing.assert_allclose(snap.current_grasp[:3, 3], [.27, 0., .03], atol=1e-8)
    np.testing.assert_allclose(snap.grasp[:3, 3], [.29, 0., .03], atol=1e-8)
    assert snap.observed_timestamp == 3. and snap.observation_age_s == .5
    assert snap.velocity_source == 'nero7_xy_kalman' and snap.target_identity == IDENTITY

def test_reset_cancel_and_track_change_reject_late_results():
    s = MeasuredMotionSession(IDENTITY)
    assert update(s, 1, 0.)
    bad = report(2, 2.); bad['track_generation'] = 3
    with pytest.raises(ValueError, match='identity changed'):
        s.update(detection(.1), identity=IDENTITY, timestamp=2., sequence=2,
                 clock_epoch=4, source_timestamp=2., velocity_measurement=bad)
    with pytest.raises(ValueError, match='epoch changed'):
        s.update(detection(.1), identity=IDENTITY, timestamp=2., sequence=2,
                 clock_epoch=5, source_timestamp=2., velocity_measurement=None)
    assert s.snapshot(2.) is None
    with pytest.raises(ValueError, match='inactive'):
        update(s, 2, .1)
    with pytest.raises(ValueError, match='inactive'):
        s.bind_selected_grasp({}, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)

def test_task_switch_and_nominal_velocity_are_rejected():
    with pytest.raises(ValueError, match='nominal'):
        MeasuredMotionSession(IDENTITY, nominal_speed_m_s=.1)
    s = MeasuredMotionSession(IDENTITY)
    with pytest.raises(ValueError, match='inactive'):
        s.update(detection(0.), identity=replace(IDENTITY, task_id='new'), timestamp=1.,
                 sequence=1, clock_epoch=4, source_timestamp=1., velocity_measurement=report(1,1.))


def test_reject_result_from_different_capture():
    s = MeasuredMotionSession(IDENTITY)
    assert update(s, 1, 0.)
    fake = detection(.5)
    with pytest.raises(ValueError, match='accepted capture'):
        s.bind_selected_grasp(fake, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    with pytest.raises(ValueError, match='accepted capture'):
        s.bind_selected_grasp(detection(0.), identity=IDENTITY, capture_timestamp=1., capture_sequence=2)


def test_measured_session_constructs_original_full_planner_without_device():
    from scipy.spatial.transform import Rotation
    from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, PickSettings
    from mujoco_conveyor_sorting.runtime.nero_moving_context import NeroMovingPolicyContext
    geometry = RobotModel()
    joints = np.array([.0142155141, .5678943524, -.0210999179,
                       1.8755820277, .0708611646, .0282067099, .6941062082])
    geometry.set_state(joints, .1)
    flange = geometry.flange()
    state = {'joints': joints.copy(), 'pose': np.r_[flange[:3, 3],
        Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')], 'width_m': .1,
        'force_n': 0., 'joint_stamp': 1., 'gripper_stamp': 1.}
    observed = detection(0.)
    observed['frame'] = RGBDFrame(header=SampleHeader(sample_time=1., received_time=1., sequence=1, clock_epoch=4),
        depth_m=np.ones((2,2)), depth_intrinsics=CameraIntrinsics(2,2,100.,100.,.5,.5),
        camera_to_world_cv=np.eye(4), serial_number='test-camera', calibration_id='test-calibration')
    observed['belt_region'] = {'plane_to_base': np.eye(4), 'bounds_xy_m': [[-1.,1.],[-1.,1.]]}
    observed['scene_state'] = state
    observed['scene_points'] = np.empty((0, 3))
    observed['grasp'] = geometry.tcp().copy()
    observed['object_points'] = observed['grasp'][:3, 3] + np.array([[0.,0.,0.],[0.,.01,.01]])
    s = MeasuredMotionSession(IDENTITY)
    assert s.update(observed, identity=IDENTITY, timestamp=1., sequence=1,
                    clock_epoch=4, source_timestamp=1., velocity_measurement=report(1,1.))
    selected = {**observed, 'network_candidate': {'score': .8}, 'network_provenance': {},
        'network_capture_center_base_m': np.median(observed['object_points'], axis=0),
        'network_capture_grasp_base': observed['grasp'].copy()}
    bound = s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    binding = s.install_latest_scene(geometry, identity=IDENTITY)
    settings = PickSettings(speed_percent=100, planning_mode='full_preplanned',
                            moving_approach_mode='continuous', selection_policy='first_safe',
                            replan_policy='event_triggered')
    monitor_options = {'read_frame': lambda _: None, 'calibrate_frame': lambda f: f,
        'capture_monotonic': lambda f: f.header.sample_time, 'belt_region': observed['belt_region']}
    with pytest.raises(ValueError, match='stale'):
        s.create_full_context(geometry, state, settings, identity=IDENTITY,
            scene_binding=replace(binding, scene_digest=b'wrong'), scene_monitor_options=monitor_options, policy_clock_origin=1.,
            belt_direction=[1.,0.,0.], belt_origin=[0.,0.,0.], s_range=[-1.,1.])
    corrupted = dict(binding.capture_detection)
    corrupted['grasp'] = corrupted['grasp'].copy()
    corrupted['grasp'][0, 3] += .01
    with pytest.raises(ValueError, match='grasp changed'):
        s.create_full_context(geometry, state, settings, identity=IDENTITY,
            scene_binding=replace(binding, capture_detection=corrupted), scene_monitor_options=monitor_options,
            policy_clock_origin=1., belt_direction=[1.,0.,0.], belt_origin=[0.,0.,0.], s_range=[-1.,1.])
    # A newer frame arrives after the scene snapshot was filtered. The scene
    # retains the captured target; the tracker must publish the later measurement.
    current = dict(observed)
    current['object_points'] = observed['object_points'] + [.05, 0., 0.]
    current['grasp'] = observed['grasp'].copy()
    current['grasp'][0, 3] += .05
    current['frame'] = replace(observed['frame'], header=SampleHeader(
        sample_time=1.5, received_time=1.5, sequence=2, clock_epoch=4))
    assert s.update(current, identity=IDENTITY, timestamp=1.5, sequence=2, clock_epoch=4,
        source_timestamp=1.5, velocity_measurement=report(2,1.5))
    with pytest.raises(ValueError, match='monitor'):
        s.create_full_context(geometry, state, settings, identity=IDENTITY,
            scene_binding=binding, policy_clock_origin=1.5, belt_direction=[1.,0.,0.],
            belt_origin=[0.,0.,0.], s_range=[-1.,1.])
    bad_frame = replace(current['frame'], calibration_id='different-calibration')
    current['frame'] = bad_frame
    current['object_points'] = observed['object_points'] + [.06, 0., 0.]
    current['grasp'][0, 3] = observed['grasp'][0, 3]+.06
    current['frame'] = replace(bad_frame, header=SampleHeader(
        sample_time=1.6, received_time=1.6, sequence=3, clock_epoch=4))
    assert s.update(current, identity=IDENTITY, timestamp=1.6, sequence=3, clock_epoch=4,
        source_timestamp=1.6, velocity_measurement=report(3,1.6))
    with pytest.raises(ValueError, match='camera binding changed'):
        s.create_full_context(geometry, state, settings, identity=IDENTITY,
            scene_binding=binding, scene_monitor_options=monitor_options, policy_clock_origin=1.6,
            belt_direction=[1.,0.,0.], belt_origin=[0.,0.,0.], s_range=[-1.,1.])
    current['frame'] = replace(observed['frame'], header=SampleHeader(
        sample_time=1.7, received_time=1.7, sequence=4, clock_epoch=4))
    current['object_points'] = observed['object_points'] + [.07, 0., 0.]
    current['grasp'][0, 3] = observed['grasp'][0, 3]+.07
    assert s.update(current, identity=IDENTITY, timestamp=1.7, sequence=4, clock_epoch=4,
        source_timestamp=1.7, velocity_measurement=report(4,1.7))
    ctx = s.create_full_context(geometry, state, settings, identity=IDENTITY,
        scene_binding=binding, scene_monitor_options=monitor_options, policy_clock_origin=1.7,
        belt_direction=[1.,0.,0.], belt_origin=[0.,0.,0.], s_range=[-1.,1.])
    try:
        assert isinstance(ctx, NeroMovingPolicyContext)
        assert ctx.target_identity == IDENTITY
        assert ctx.full_plan_scene_monitor.snapshot()['reason'] == 'waiting_for_paired_frame'
        assert ctx._snapshot(.2).observed_timestamp == 1.7
        np.testing.assert_allclose(ctx.grasp_transform, bound['grasp'])
        np.testing.assert_allclose(ctx._snapshot(.2).current_grasp[0, 3], bound['grasp'][0, 3]+.09, atol=1e-8)
        worker = ctx.create_global_planning_worker()
        assert worker.__class__.__name__ == '_MovingPlanningWorker'
        assert worker.close()
    finally:
        ctx.close()
        s.cancel()


def test_selected_network_result_cannot_be_replaced_by_late_result():
    s = MeasuredMotionSession(IDENTITY)
    assert update(s, 1, 0.)
    selected = {**detection(0.), 'network_candidate': {'score': .8},
                'network_capture_grasp_base': np.eye(4),
                'network_capture_center_base_m': [0., .005, .015], 'network_provenance': {}}
    s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    with pytest.raises(ValueError, match='already bound'):
        s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)


def test_scene_installation_keeps_capture_when_new_frame_arrives():
    import threading
    s = MeasuredMotionSession(IDENTITY, max_coast_s=None)
    observed = detection(0.)
    observed.update(frame=SimpleNamespace(header=SimpleNamespace(sample_time=1., sequence=1, clock_epoch=4),
                    camera_to_world_cv=np.eye(4)), scene_points=np.zeros((1,3)), scene_state={})
    assert s.update(observed, identity=IDENTITY, timestamp=1., sequence=1, clock_epoch=4,
                    source_timestamp=1., velocity_measurement=report(1,1.))
    selected = {**observed, 'network_candidate': {'score': .8}, 'network_provenance': {},
                'network_capture_grasp_base': np.eye(4),
                'network_capture_center_base_m': [0., .005, .015]}
    s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    started, proceed = threading.Event(), threading.Event()
    class Geometry:
        def install_scene(self, points, target, state, **kwargs):
            self.scene, self.target = points.copy(), target[::3].copy()
            started.set()
            assert proceed.wait(3.)
            return {}
    errors, bindings = [], []
    def install():
        try:
            bindings.append(s.install_latest_scene(Geometry(), identity=IDENTITY))
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=install)
    thread.start()
    assert started.wait(3.)
    assert update(s, 2, .1)  # Perception is not blocked by scene installation.
    proceed.set(); thread.join(3.)
    assert not thread.is_alive()
    assert not errors and len(bindings) == 1
    assert bindings[0].capture_time == 1. and s.snapshot(2.).observed_timestamp == 2.


def test_scene_rejects_wrong_frame_epoch_before_installing():
    s = MeasuredMotionSession(IDENTITY)
    observed = detection(0.)
    observed.update(frame=SimpleNamespace(header=SimpleNamespace(sample_time=1., sequence=1, clock_epoch=5),
                    camera_to_world_cv=np.eye(4)), scene_points=np.zeros((1,3)), scene_state={})
    assert s.update(observed, identity=IDENTITY, timestamp=1., sequence=1, clock_epoch=4,
                    source_timestamp=1., velocity_measurement=report(1,1.))
    selected = {**observed, 'network_candidate': {'score': .8}, 'network_provenance': {},
                'network_capture_grasp_base': np.eye(4),
                'network_capture_center_base_m': [0., .005, .015]}
    s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    with pytest.raises(ValueError, match='frame does not match'):
        s.install_latest_scene(object(), identity=IDENTITY)


def test_cancel_during_planner_preparation_closes_late_context(monkeypatch):
    import threading
    import mujoco_conveyor_sorting.runtime.nero_moving_context as original
    s = MeasuredMotionSession(IDENTITY)
    observed = detection(0.)
    observed.update(frame=SimpleNamespace(header=SimpleNamespace(sample_time=1., sequence=1, clock_epoch=4),
                    camera_to_world_cv=np.eye(4)), scene_points=np.zeros((1,3)), scene_state={})
    assert s.update(observed, identity=IDENTITY, timestamp=1., sequence=1, clock_epoch=4,
                    source_timestamp=1., velocity_measurement=report(1,1.))
    selected = {**observed, 'network_candidate': {'score': .8}, 'network_provenance': {},
                'network_capture_grasp_base': np.eye(4),
                'network_capture_center_base_m': [0., .005, .015]}
    s.bind_selected_grasp(selected, identity=IDENTITY, capture_timestamp=1., capture_sequence=1)
    class Geometry:
        def install_scene(self, points, target, state, **kwargs):
            self.scene, self.target = points.copy(), target[::3].copy()
            return {}
    geometry = Geometry()
    binding = s.install_latest_scene(geometry, identity=IDENTITY)
    started, proceed = threading.Event(), threading.Event()
    class Context:
        closed = False
        def set_policy_clock_origin(self, origin): pass
        def close(self): self.closed = True
    context = Context()
    def build(*args, **kwargs):
        started.set()
        assert proceed.wait(3.)
        return context
    monkeypatch.setattr(original, 'build_moving_policy_context', build)
    settings = SimpleNamespace(planning_mode='full_preplanned', moving_approach_mode='continuous')
    errors = []
    def create():
        try:
            s.create_full_context(geometry, {}, settings, identity=IDENTITY,
                scene_binding=binding, scene_monitor_options={}, policy_clock_origin=1., belt_direction=[1.,0.,0.],
                belt_origin=[0.,0.,0.], s_range=[-1.,1.])
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=create)
    thread.start()
    assert started.wait(3.)
    s.cancel()  # Must not wait for planner preparation.
    proceed.set(); thread.join(3.)
    assert not thread.is_alive() and context.closed
    assert len(errors) == 1 and 'inactive' in str(errors[0])


def test_pinned_network_capture_survives_frame_eviction_but_not_cancellation():
    s = MeasuredMotionSession(IDENTITY, max_coast_s=None)
    assert update(s, 1, 0.)
    s.pin_network_capture(identity=IDENTITY, timestamp=1., sequence=1)
    selected={**detection(0.),'network_candidate':{'score':.8},'network_provenance':{},
        'network_capture_center_base_m':[0.,.005,.015], 'network_capture_grasp_base':np.eye(4)}
    for seq in range(2,72):
        assert update(s,seq,(seq-1)*.1)
    bound=s.bind_selected_grasp(selected,identity=IDENTITY,capture_timestamp=1.,capture_sequence=1)
    assert bound['network_candidate']['score']==.8
    s.cancel()
    with pytest.raises(ValueError,match='inactive'):
        s.bind_selected_grasp(selected,identity=IDENTITY,capture_timestamp=1.,capture_sequence=1)


def test_pin_cannot_be_reassigned_to_a_second_inflight_capture():
    s=MeasuredMotionSession(IDENTITY,max_coast_s=None)
    assert update(s,1,0.) and update(s,2,.1)
    s.pin_network_capture(identity=IDENTITY,timestamp=1.,sequence=1)
    with pytest.raises(ValueError,match='already pending'):
        s.pin_network_capture(identity=IDENTITY,timestamp=2.,sequence=2)


def test_unlimited_coast_preserves_original_exposure_and_velocity_without_new_frames():
    s=MeasuredMotionSession(IDENTITY,max_coast_s=None,max_prediction_horizon_s=None)
    assert update(s,1,0.)
    assert update(s,2,.1)
    initial=s.snapshot(2.)
    later=s.snapshot(7.)
    assert later.observed_timestamp==initial.observed_timestamp==2.
    assert later.observation_age_s==5.
    np.testing.assert_allclose(later.velocity,initial.velocity)
    np.testing.assert_allclose(later.current_grasp[:3,3]-initial.current_grasp[:3,3],[.5,0,0],atol=1e-9)
    assert later.target_identity==IDENTITY
