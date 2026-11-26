"""Historical exposure pairing and geometry reuse; no camera/CAN connection."""
from copy import deepcopy
from threading import RLock

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.feedback_pairing import JointFeedbackHistory
from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
from mujoco_conveyor_sorting.runtime import self_filter
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel
from test_single_exposure_hardware import fixture


def delayed_capture_state():
    """Reproduce the measured 282.9 ms mismatch, with injected CAN brackets."""
    capture = 38677.97801613808
    current = 38678.26089191437
    hw = Hardware.__new__(Hardware)
    hw.lock = RLock()
    hw._joint_feedback_history = JointFeedbackHistory()
    for t in np.arange(capture-.005, current+.006, .005):
        hw._joint_feedback_history.observe_packet(tuple(range(7)), [t-capture]*7, t, epoch=3)
    state = dict(joints=[current-capture]*7, width_m=.1,
        joint_stamp_monotonic=current, pose_stamp_monotonic=38678.24726629257,
        feedback_time={name: dict(clock_epoch=3, mapping_quality='mapped') for name in
                       ('joint_12', 'joint_34', 'joint_56', 'joint_7')})
    hw.snapshot_nowait = lambda: state
    return hw, state, capture


def test_283ms_current_packet_delay_does_not_reject_exposure_paired_joints(monkeypatch):
    hw, current, capture = delayed_capture_state()
    original = deepcopy(current)
    v, _, _, _ = fixture(monkeypatch)
    v.enable_lookahead()
    v.capture_state_reader = hw.snapshot_for_capture
    try:
        state = v._paired_state(capture)
        assert state is not None
        np.testing.assert_allclose(state['joints'], np.zeros(7), atol=1e-12)
        assert mapped_stamp(state)-capture == pytest.approx(.28287577629)
        assert state['capture_joint_pairing']['joint_interpolation_stamp'] == capture
        assert state['capture_joint_pairing']['clock_epoch'] == 3
        assert state['capture_gripper_state_basis'] == 'current_measured_width_not_interpolated'
        assert state['feedback_time'] == original['feedback_time']
        assert current == original
        assert v.progress.snapshot()['frame']['capture_feedback_basis'] == 'measured_historical_joint_brackets'
    finally:
        v.close()


@pytest.mark.parametrize('fault', ['outside_bracket', 'wide_bracket', 'wrong_exposure',
                                  'epoch_mismatch', 'extrapolation', 'missing_axis'])
def test_historical_pairing_must_keep_actual_brackets_and_epoch(monkeypatch, fault):
    hw, _, capture = delayed_capture_state()
    state = hw.snapshot_for_capture(capture)
    evidence = state['capture_joint_pairing']
    if fault == 'outside_bracket': evidence['bracket_bounds_s'][0] = [capture+.001, capture+.002]
    if fault == 'wide_bracket': evidence['bracket_bounds_s'][0] = [capture-.03, capture+.03]
    if fault == 'wrong_exposure': evidence['joint_interpolation_stamp'] += .001
    if fault == 'epoch_mismatch': state['feedback_time']['joint_34']['clock_epoch'] += 1
    if fault == 'extrapolation': evidence['extrapolated'] = True
    if fault == 'missing_axis': evidence['bracket_bounds_s'].pop()
    v, _, _, _ = fixture(monkeypatch)
    v.enable_lookahead();v.capture_state_reader = lambda _: state
    try:
        assert v._paired_state(capture) is None
        assert v.progress.snapshot()['last_blocker']['reason'] == 'capture_joint_pairing_invalid'
    finally:
        v.close()


def test_capture_adapter_does_not_bridge_clock_reset():
    hw, _, capture = delayed_capture_state()
    hw._joint_feedback_history.observe_packet(tuple(range(7)), [0.]*7, capture+.3, epoch=4)
    assert hw.snapshot_for_capture(capture) is None


def test_moving_self_filter_reuses_only_local_hull_not_point_results(monkeypatch):
    model = RobotModel()
    original = self_filter.HalfspaceIntersection
    intersections = []
    def count(*args, **kwargs):
        intersections.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(self_filter, 'HalfspaceIntersection', count)
    first = model.self_filter_bodies_for_state(dict(joints=np.zeros(7), width_m=.1))
    builds = len(intersections)
    assert builds > 0
    q = np.array([.1, -.2, .1, .3, -.1, .2, .1])
    moved = model.self_filter_bodies_for_state(dict(joints=q, width_m=.04))
    assert len(intersections) == builds
    assert any(not np.array_equal(a.position, b.position) for a, b in zip(first, moved))
    # The pre-optimization body construction is the numerical reference.
    baseline = [self_filter.ConvexBody(b.geom_id, b.name, model.meshes[b.geom_id][0],
        model.mesh_vertices[b.geom_id], b.position, b.rotation, b.padding_m) for b in moved]
    for a, b in zip(moved, baseline):
        np.testing.assert_array_equal(a.padded_equations, b.padded_equations)
        assert a.radius == b.radius and a.padded_radius == b.padded_radius
    rng = np.random.default_rng(42)
    points = np.concatenate([b.vertices@b.rotation.T+b.position for b in moved])
    points = points[::7] + rng.normal(0., .01, points[::7].shape)
    origin = np.array([-.8, .2, .9])
    a = model.self_filter.classify(points, moved, origin)
    b = model.self_filter.classify(points, baseline, origin)
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.removed_mask, b.removed_mask)
    # A new point set is classified afresh, never an old collision conclusion.
    outside = model.self_filter.classify(np.array([[5., 5., 5.]]), moved, origin)
    assert not outside.removed_mask.any()
    before = len(intersections)
    model.mesh_vertices[next(iter(model.meshes))][0, 0] += .00001
    model.self_filter_bodies_for_state(dict(joints=q, width_m=.04))
    assert len(intersections) > before
