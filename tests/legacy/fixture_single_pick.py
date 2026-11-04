"""Offline only: these tests must never construct a hardware adapter."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import (
    PickCycle, PickRejected, PickSettings, RobotModel, stable_target, detect_target)
from mujoco_conveyor_sorting.runtime.nero_pick_hardware import fresh, FeedbackError
from mujoco_conveyor_sorting.runtime.types import CameraIntrinsics, RGBDFrame, SampleHeader
from mujoco_conveyor_sorting.runtime.belt_scene_config import BeltSceneFilterConfig


class Clock:
    t = 1.
    def now(self): return self.t
    def sleep(self, dt): self.t += dt


class FakeHardware:
    def __init__(self, clock, *, empty=False, stalled=False, lost_force=False):
        self.clock = clock; self.commands = []; self.q = np.zeros(7)
        self.width = .08; self.force = 0.; self.empty = empty; self.stalled = stalled
        self.lost_force = lost_force
    def snapshot(self):
        return {'pose': [self.q[0], 0, .2, 0, 0, 0], 'joints': self.q.tolist(),
                'width_m': self.width, 'force_n': self.force,
                'pose_stamp': self.clock.now(), 'gripper_stamp': self.clock.now()}
    def prepare_motion(self, speed): self.commands.append(('speed', speed))
    def grip(self, width, force):
        self.commands.append(('grip', width))
        self.width = width if width else (0. if self.empty else .04)
        self.force = 0. if width or self.empty else 1.
    def move_joints(self, target):
        self.commands.append(('move', list(target)))
        if not self.stalled: self.q = np.array(target)
        if self.lost_force and self.force: self.force = 0.
    def stop(self): self.commands.append(('stop',))


class FakeModel:
    flange_to_tcp = np.eye(4)
    def check_feedback(self, s): pass
    def check_edge(self, *args, **kwargs): pass


def fixture_cycle(**kwargs):
    clock = Clock(); hw = FakeHardware(clock, **kwargs)
    events = []
    cycle = PickCycle(hw, FakeModel(), PickSettings(move_timeout_s=3),
                      lambda e, d: events.append((e, d)), clock=clock.now, sleep=clock.sleep)
    stages = []
    for name, x in [('raise_current', .01), ('above_object', .02), ('descend', .03),
                    ('lift', .04), ('lower', .03), ('retreat', .02)]:
        t = np.eye(4); t[:3, 3] = [x, 0, .2]
        stages.append({'name': name, 'joints_rad': [[x, 0, 0, 0, 0, 0, 0]], 'target_tcp': t.tolist()})
    return cycle, hw, {'start_joints_rad': [0.]*7, 'stages': stages}, events


def test_plan_only_never_sends():
    cycle, hw, plan, _ = fixture_cycle()
    assert cycle.run(plan)['status'] == 'plan_only'
    assert hw.commands == []


def test_current_pose_cycle_releases_only_after_return():
    cycle, hw, plan, events = fixture_cycle()
    assert cycle.run(plan, execute=True)['status'] == 'cycle_completed_from_feedback'
    assert [c[0] for c in hw.commands if c[0] != 'move'] == ['speed', 'grip', 'grip', 'grip']
    assert sum(c[0] == 'move' for c in hw.commands) > 6
    assert [d['name'] for e, d in events if e == 'stage'] == [s['name'] for s in plan['stages']]
    i = [e for e, _ in events].index('open_release')
    assert [d['name'] for e, d in events[:i] if e == 'stage'][-1] == 'lower'
    assert not any(c[0] == 'stop' for c in hw.commands)


@pytest.mark.parametrize('fault', ['empty', 'stalled', 'lost_force'])
def test_fault_stops_without_release(fault):
    cycle, hw, plan, events = fixture_cycle(**{fault: True})
    with pytest.raises(PickRejected): cycle.run(plan, execute=True)
    assert hw.commands[-1] == ('stop',)
    assert not any(e == 'open_release' for e, _ in events)


def test_moved_start_rejected_before_any_command():
    cycle, hw, plan, _ = fixture_cycle(); hw.q[0] = .1
    with pytest.raises(PickRejected, match='规划期间'): cycle.run(plan, execute=True)
    assert not hw.commands


def test_camera_failure_stops_before_following_motion():
    cycle, hw, plan, events = fixture_cycle()
    def failed(): raise PickRejected('camera disconnected')
    cycle.heartbeat = failed
    with pytest.raises(PickRejected, match='camera'): cycle.run(plan, execute=True)
    assert hw.commands == []  # Camera preflight now precedes all actuation.


def test_stale_feedback_after_command_stops():
    cycle, hw, plan, _ = fixture_cycle()
    snapshot = hw.snapshot
    def stale():
        if hw.commands: raise FeedbackError('stale')
        return snapshot()
    hw.snapshot = stale
    with pytest.raises(FeedbackError): cycle.run(plan, execute=True)
    assert hw.commands[-1] == ('stop',)


def test_recorded_feedback_cannot_satisfy_settle():
    cycle, hw, plan, _ = fixture_cycle()
    snapshot = hw.snapshot
    def repeated():
        state = snapshot(); state.update(pose_stamp=1, gripper_stamp=1); return state
    hw.snapshot = repeated
    with pytest.raises(PickRejected, match='按时到位'): cycle.run(plan, execute=True)
    assert not any(c[0] == 'move' for c in hw.commands)
    assert hw.commands[-1] == ('stop',)


@pytest.mark.parametrize('stamp', [None, .79, 1.01, float('nan')])
def test_bad_timestamp_rejected(stamp):
    msg = None if stamp is None else SimpleNamespace(timestamp=stamp, msg={})
    with pytest.raises(FeedbackError): fresh(msg, 'test', 1.)


@pytest.mark.parametrize('value', [.001, .1, float('nan')])
def test_lift_validation(value):
    with pytest.raises(ValueError): PickSettings(lift_m=value).validate()


def test_moving_target_rejected():
    a = {'center': np.zeros(3), 'height': .05, 'width': .04}
    b = dict(a, center=np.array([.02, 0, 0]))
    with pytest.raises(PickRejected): stable_target([a, a, b])


@pytest.fixture(scope='module')
def model(): return RobotModel(belt_scene_filter_config=BeltSceneFilterConfig(enabled=False))


def test_nominal_tcp_and_flange_frame(model):
    # Drawing-sized gripper: 142.5 mm fingertip, TCP 10 mm behind that plane.
    # link7 is the SDK frame; the adapter offset is part of this transform.
    np.testing.assert_allclose(model.flange_to_tcp[:3, 3], [.1695, 0, -.0235], atol=1e-6)
    assert model.names == [f'joint{i}' for i in range(1, 8)]


def compiled_mesh_in_gripper_base(model, geom_name):
    """Undo MuJoCo's compiled mesh frame before measuring CAD dimensions."""
    mj_model, data = model.model, model.data
    geom_id = mj_model.geom(geom_name).id
    mesh_id = int(mj_model.geom_dataid[geom_id])
    begin = int(mj_model.mesh_vertadr[mesh_id])
    count = int(mj_model.mesh_vertnum[mesh_id])
    vertices = np.asarray(mj_model.mesh_vert[begin:begin+count], dtype=float)
    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    base_id = mj_model.body('gripper_base').id
    return (world-data.xpos[base_id]) @ data.xmat[base_id].reshape(3, 3)


@pytest.mark.parametrize('q,width', [
    (np.zeros(7), 0.),
    (np.array([1.74, -.86, 2.08, 1.65, 1.03, .41, 1.09]), .1),
])
def test_compiled_gripper_matches_drawing_dimensions_and_feedback_tcp(model, q, width):
    # Measure compiled geometry in its installation frame, including mesh
    # recentering/rotation and articulated finger transforms, not XML labels.
    model.set_state(q, width)
    palm = compiled_mesh_in_gripper_base(model, 'nero_gripper_base_collision')
    assert palm[:, 2].min() == pytest.approx(0., abs=2e-6)
    assert np.min(np.abs(palm[:, 2]-.0505)) < 2e-6
    assert palm[:, 2].max() == pytest.approx(.0505+.019, abs=2e-6)
    base_id = model.model.body('gripper_base').id
    base_rotation = model.data.xmat[base_id].reshape(3, 3)
    for geom_name in ('left_finger_geom', 'right_finger_geom'):
        finger = compiled_mesh_in_gripper_base(model, geom_name)
        tip = finger[:, 2].max()
        assert tip == pytest.approx(.0505+.019+.073, abs=2e-6)
        assert tip-palm[:, 2].max() == pytest.approx(.073, abs=2e-6)
    for site_name in ('ee_site', 'grasp_attach_site',
                      'left_finger_pad_site', 'right_finger_pad_site'):
        site_id = model.model.site(site_name).id
        local = (model.data.site_xpos[site_id]-model.data.xpos[base_id]) @ base_rotation
        assert local[2] == pytest.approx(.1425-.010, abs=1e-8)
    # Execution reconstructs TCP from SDK-frame flange feedback with this
    # nominal transform; it must reach the same site that IK and previews use.
    np.testing.assert_allclose(model.flange() @ model.flange_to_tcp, model.tcp(), atol=1e-12)


def test_ik_and_path_validator_use_identical_joint_bounds(model):
    np.testing.assert_array_equal(model.ik.lower_limits, model.command_lower)
    np.testing.assert_array_equal(model.ik.upper_limits, model.command_upper)
    np.testing.assert_allclose(model.command_upper, model.limits[:, 1]-.02)
    np.testing.assert_allclose(model.ik.center_positions, .5*(model.command_lower+model.command_upper))


def test_limit_error_identifies_joint_and_boundary_without_widening(model):
    q = np.array([1.74, -.86, 2.08, 1.65, 1.03, .41, 1.09])
    q[6] = model.command_upper[6]+.0002
    with pytest.raises(PickRejected, match=r'joint7=.*允许'):
        model.check_state(q, .1)


def test_bounded_ik_fallback_reaches_pose_without_changing_limits(monkeypatch):
    model = RobotModel(belt_scene_filter_config=BeltSceneFilterConfig(enabled=False))
    q = np.array([1.74, -.86, 2.08, 1.65, 1.03, .41, 1.09])
    model.set_state(q, .1); target = model.tcp().copy()
    quat = Rotation.from_matrix(target[:3, :3]).as_quat()[[3, 0, 1, 2]]
    nominal = model.limits.copy()
    monkeypatch.setattr(model.ik, 'solve', lambda *a, **kw: SimpleNamespace(success=False))
    solved = model.solve_pose(target[:3, 3], quat, q+np.array([.015, -.01, 0, .01, 0, 0, 0]))
    assert solved.success, solved
    assert np.all(solved.joint_positions >= model.command_lower)
    assert np.all(solved.joint_positions <= model.command_upper)
    np.testing.assert_array_equal(model.limits, nominal)


def test_bounded_ik_does_not_mark_unreachable_target_success(monkeypatch):
    model = RobotModel(belt_scene_filter_config=BeltSceneFilterConfig(enabled=False))
    monkeypatch.setattr(model.ik, 'solve', lambda *a, **kw: SimpleNamespace(success=False))
    solved = model.solve_pose([2, 2, 2], [1, 0, 0, 0], np.zeros(7))
    assert not solved.success
    assert solved.position_error > 1


def test_real_model_current_state_plan(model):
    q = np.array([1.740543896, -.862720048, 2.082230799, 1.653253916, 1.025960646, .409752901, 1.088976081])
    model.set_state(q, .1)
    flange = model.flange(); tcp = model.tcp()
    pose = np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')]
    state = {'joints': q.tolist(), 'pose': pose.tolist(), 'width_m': .1}
    grasp = tcp.copy(); grasp[2, 3] -= .04
    observation = {'grasp': grasp}
    model.install_scene(np.empty((0, 3)), np.empty((0, 3)), state)
    plan = model.plan(state, observation, PickSettings())
    np.testing.assert_allclose(plan['start_joints_rad'], q)
    assert [s['name'] for s in plan['stages']] == ['raise_current', 'above_object', 'descend', 'lift', 'lower', 'retreat']
    assert all(s['joints_rad'] for s in plan['stages'])


def test_collision_with_observed_point_rejected(model):
    q = np.array([1.74, -.86, 2.08, 1.65, 1.03, .41, 1.09])
    model.set_state(q, .1)
    g = model.model.geom('nero_link3_collision').id
    model.scene = model.data.geom_xpos[g][None].copy()
    from scipy.spatial import cKDTree
    model.tree = cKDTree(model.scene)
    with pytest.raises(PickRejected, match='障碍'): model.check_state(q, .1)


def load_entry():
    path = Path(__file__).resolve().parents[1]/'scripts/test_pick_and_place.py'
    spec = importlib.util.spec_from_file_location('pick_entry_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def synthetic_frame(two=False):
    # Downward camera 1 m above green plane; zero distortion. Two separated
    # raised squares exercise ambiguity rejection through the ACTUAL detector.
    k = CameraIntrinsics(width=320, height=240, fx=300, fy=300, cx=159.5, cy=119.5)
    rgb = np.zeros((240, 320, 3), np.uint8); rgb[:] = [30, 150, 40]
    depth = np.ones((240, 320), np.float32)
    rgb[108:125, 150:167] = 120; depth[108:125, 150:167] = .95
    if two:
        rgb[108:125, 210:227] = 120; depth[108:125, 210:227] = .95
    x = np.diag([1., -1, -1, 1]); x[2, 3] = 1.
    return RGBDFrame(header=SampleHeader(1, 1, 1), depth_m=depth, rgb=rgb,
                     depth_intrinsics=k, color_intrinsics=k, camera_to_world_cv=x,
                     serial_number='test', calibration_id='synthetic')


def test_actual_detector_single_and_ambiguous():
    entry = load_entry()
    ob = detect_target(synthetic_frame(), np.eye(4), fit_plane=entry.fit_plane,
                       green_belt_points=entry.green_belt_points)
    assert abs(ob['height']-.05) < .001
    assert abs(ob['grasp'][2, 3]-.035) < .001
    assert ob['category'].startswith('unknown')
    with pytest.raises(PickRejected, match='2个候选'):
        detect_target(synthetic_frame(True), np.eye(4), fit_plane=entry.fit_plane,
                      green_belt_points=entry.green_belt_points)


def synthetic_rectangle(length, width, yaw_deg=0):
    frame = synthetic_frame()
    depth = np.ones_like(frame.depth_m)
    rgb = np.full_like(frame.rgb, [30, 150, 40])
    # Define the visible top rectangle in metres in the base XY plane.
    y, x = np.indices(depth.shape)
    points_x = (x-frame.depth_intrinsics.cx)*.95/frame.depth_intrinsics.fx
    points_y = -(y-frame.depth_intrinsics.cy)*.95/frame.depth_intrinsics.fy
    yaw = np.deg2rad(yaw_deg)
    along = points_x*np.cos(yaw)+points_y*np.sin(yaw)
    across = -points_x*np.sin(yaw)+points_y*np.cos(yaw)
    mask = (np.abs(along)<length/2) & (np.abs(across)<width/2)
    depth[mask] = .95; rgb[mask] = 120
    from dataclasses import replace
    return replace(frame, depth_m=depth, rgb=rgb)


@pytest.mark.parametrize('yaw', [0, 30, 90])
def test_eighty_mm_long_part_fits_across_short_side(yaw):
    entry = load_entry()
    ob = detect_target(synthetic_rectangle(.0802, .0564, yaw), np.eye(4),
                       fit_plane=entry.fit_plane, green_belt_points=entry.green_belt_points,
                       target_shape='auto')
    assert ob['footprint_m'][0] > .075
    assert .05 < ob['width'] < .07
    assert ob['open_clearance_per_side_m'] > .015
    assert ob['target_geometry_source'] == 'depth_foreground_envelope'
    # Verify that the ACTUAL jaw axis follows the short side, modulo sign.
    short_axis = np.array([-np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw)), 0])
    assert abs(np.dot(ob['grasp'][:3, 1], short_axis)) > .98


def test_jaw_width_over_eighty_mm_is_rejected():
    entry = load_entry()
    with pytest.raises(PickRejected, match='实际夹持方向宽'):
        detect_target(synthetic_rectangle(.12, .09, 30), np.eye(4),
                      fit_plane=entry.fit_plane, green_belt_points=entry.green_belt_points)


def test_uncertain_yaw_checks_actual_width_instead_of_short_side():
    entry = load_entry()
    # Square at 45 deg has uncertain PCA yaw. Fixed yaw's jaw span exceeds
    # 80 mm despite both principal extents fitting the opening margin.
    with pytest.raises(PickRejected, match='实际夹持方向宽'):
        detect_target(synthetic_rectangle(.066, .066, 45), np.eye(4),
                      fit_plane=entry.fit_plane, green_belt_points=entry.green_belt_points)


def test_low_boundary_extension_saved_before_rejection(tmp_path):
    entry = load_entry()
    frame = synthetic_rectangle(.10, .05)
    depth = frame.depth_m.copy()
    _, x = np.indices(depth.shape)
    side = (depth < 1) & (np.abs(x-frame.depth_intrinsics.cx) > 8)
    depth[side] = .985  # 15 mm-high spurious boundary around 50 mm-high top.
    from dataclasses import replace
    frame = replace(frame, depth_m=depth)
    stem = tmp_path/'synthetic'
    with pytest.raises(PickRejected, match='深度轮廓随高度明显变化'):
        detect_target(frame, np.eye(4), fit_plane=entry.fit_plane,
                      green_belt_points=entry.green_belt_points,
                      diagnostic=lambda ob: entry.save_target_diagnostic(stem, frame, np.eye(4), ob))
    import json
    review = json.loads((tmp_path/'synthetic_target_geometry.json').read_text())
    assert review['shape_consistency']['status'] == 'inconsistent'
    assert (tmp_path/'synthetic_target.png').is_file()


def test_depth_cylinder_checks_and_refines_low_boundary_tail():
    entry = load_entry(); frame = synthetic_frame()
    y, x = np.indices(frame.depth_m.shape)
    px = (x-frame.depth_intrinsics.cx)*.95/frame.depth_intrinsics.fx
    py = (y-frame.depth_intrinsics.cy)*.95/frame.depth_intrinsics.fy
    circle = px*px+py*py < .025**2
    tail = (np.abs(px)<.05) & (np.abs(py)<.008) & ~circle
    depth = np.ones_like(frame.depth_m); depth[circle] = .95; depth[tail] = .985
    rgb = np.full_like(frame.rgb, [30, 150, 40]); rgb[circle | tail] = 120
    from dataclasses import replace
    frame = replace(frame, depth_m=depth, rgb=rgb)
    ob = detect_target(frame, np.eye(4), fit_plane=entry.fit_plane,
                       green_belt_points=entry.green_belt_points, target_shape='cylinder')
    assert ob['shape_consistency']['status'] == 'inconsistent'
    assert ob['cylinder_fit']['accepted']
    assert ob['target_geometry_source'] == 'depth_cylinder_candidate'
    assert .045 < ob['cylinder_fit']['diameter_m'] < .055
    assert ob['width'] < .065
    assert np.linalg.norm(ob['center'][:2]) < .002


def test_requested_cylinder_does_not_accept_rectangular_part():
    entry = load_entry()
    with pytest.raises(PickRejected, match='深度圆柱拟合未通过'):
        detect_target(synthetic_rectangle(.0802, .0564), np.eye(4),
                      fit_plane=entry.fit_plane, green_belt_points=entry.green_belt_points,
                      target_shape='cylinder')


def test_simple_localizes_despite_low_boundary_extension(monkeypatch):
    entry = load_entry(); frame = synthetic_rectangle(.10, .05)
    depth = frame.depth_m.copy()
    _, x = np.indices(depth.shape)
    depth[(depth < 1) & (np.abs(x-frame.depth_intrinsics.cx) > 8)] = .985
    from dataclasses import replace
    frame = replace(frame, depth_m=depth)
    import mujoco_conveyor_sorting.runtime.depth_shape as shape
    def forbidden(*args, **kwargs): raise AssertionError('simple must not run circle fits')
    monkeypatch.setattr(shape, 'fit_upright_cylinder', forbidden)
    ob = detect_target(frame, np.eye(4), fit_plane=entry.fit_plane,
                       green_belt_points=entry.green_belt_points, target_shape='simple')
    assert ob['shape_consistency']['status'] == 'not_required_in_simple_mode'
    assert ob['target_geometry_source'] == 'upper_depth_geometry'
    assert ob['width'] < .065
    assert ob['localization_points'] < len(ob['object_points'])
    assert len(ob['object_points']) == ob['mask'].sum()  # Full collision data retained.
    assert np.linalg.norm(ob['center'][:2]) < .003


def test_entry_explicit_simple_samples_once_without_shape_stability_gate(tmp_path, monkeypatch):
    entry = load_entry(); frame = synthetic_frame()
    class Camera:
        calls = 0
        def read_latest(self, **kwargs): self.calls += 1; return frame
        def get_depth_to_color_cv(self): return np.eye(4)
    camera = Camera()
    calibration = {'serial_number': frame.serial_number,
                   'camera_to_base_cv': frame.camera_to_world_cv.tolist(),
                   'captured_pnp_calibration_reference': {
                       'image_size': [320, 240], 'K': frame.color_intrinsics.matrix.tolist(), 'D': [0.]*5}}
    monkeypatch.setattr(entry.time, 'sleep', lambda _: None)
    def forbidden(*args): raise AssertionError('simple must not run multi-frame shape checks')
    monkeypatch.setattr(entry, 'stable_target', forbidden)
    ob, sequence = entry.record_observation(camera, calibration, tmp_path, target_shape='simple')
    assert camera.calls == 1
    assert sequence == frame.header.sequence
    assert ob['target_geometry_source'] == 'upper_depth_geometry'


def test_depth_color_transform_order_and_resolution():
    entry = load_entry(); f = synthetic_frame()
    x = np.eye(4); x[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
    d2c = np.eye(4); d2c[0, 3] = .02
    calibration = {'serial_number': 'test', 'camera_to_base_cv': x.tolist(),
                   'captured_pnp_calibration_reference': {'K': f.color_intrinsics.matrix.tolist(),
                       'D': [0.]*5, 'image_size': [320, 240]}}
    out = entry.calibrate_frame(f, calibration, d2c)
    np.testing.assert_allclose(out.camera_to_world_cv, x@d2c)
    calibration['captured_pnp_calibration_reference']['image_size'] = [640, 480]
    with pytest.raises(PickRejected, match='分辨率'): entry.calibrate_frame(f, calibration, d2c)
