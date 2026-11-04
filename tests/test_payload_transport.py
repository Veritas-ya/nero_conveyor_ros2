"""Payload timing and unchanged retention guards; no physical devices."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import yaml
from scipy.interpolate import BPoly
from nero_core.two_second.config import CellConfig
from nero_core.two_second.motion_profile import select_motion_profile
from nero_core.two_second.command_history import CommandHistory
from nero_core.two_second.trajectory import TimedPath, extrema
from mujoco_conveyor_sorting.runtime.single_pick import PickCycle, PickSettings, PickRejected
from test_two_second_hardware import site, executor_fixture


def profile_config():
    return SimpleNamespace(settings=PickSettings(speed_percent=100),
        raw={'loaded_joint_acceleration_limits_rad_s2': [2.]*7,
             'loaded_acceleration_limit_source': 'test_only_prior_payload_profile'},
        velocity_limits=np.ones(7)*2., acceleration_limits=np.ones(7)*8.)


def test_payload_selection_keeps_manufacturer_envelope_and_empty_motion_unchanged():
    config = profile_config()
    loaded = select_motion_profile(config, payload=True)
    empty = select_motion_profile(config)
    np.testing.assert_array_equal(config.acceleration_limits, 8.)
    np.testing.assert_array_equal(empty.acceleration_limits, 8.)
    np.testing.assert_array_equal(loaded.acceleration_limits, 2.)
    assert loaded.name == 'loaded' and not loaded.acceleration_limits.flags.writeable
    del config.raw['loaded_joint_acceleration_limits_rad_s2']
    np.testing.assert_array_equal(select_motion_profile(config, payload=True).acceleration_limits, 8.)


@pytest.mark.parametrize('value,source', [([2.]*7, ''), ([9.]*7, 'test'),
    ([0.]*7, 'test'), ([float('nan')]*7, 'test'), (None, 'test'), ([2.]*6, 'test')])
def test_invalid_payload_profile_is_rejected_before_opening_devices(site, value, source):
    path, data = site
    data.update(loaded_joint_acceleration_limits_rad_s2=value, loaded_acceleration_limit_source=source)
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        CellConfig.load(path)


def test_real_cell_reports_separate_8_capability_and_2_payload_request():
    config = CellConfig.load(Path(__file__).parents[1]/
        'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    np.testing.assert_array_equal(config.acceleration_limits, 8.)
    np.testing.assert_array_equal(select_motion_profile(config, payload=True).acceleration_limits, 2.)
    assert config.settings.speed_percent == 100 and config.settings.force_n == 2.


def test_actual_dispatch_history_applies_the_selected_payload_profile():
    config = profile_config()
    config.raw['runtime_command_acceleration_check_enabled'] = True  # explicit opt-in regression
    history = CommandHistory(config)
    history.record(1., .01, np.zeros(7), {})
    history.record(1.01, .01, np.ones(7)*.0002, {})
    q = np.ones(7)*.0008  # a=4 passes capability8, exceeds payload request2
    history.validate(q, 1.02)
    with pytest.raises(PickRejected, match='加速度') as error:
        history.validate(q, 1.02, limits=select_motion_profile(config, payload=True))
    assert error.value.details['acceleration_limits_rad_s2'] == [2.]*7


def test_hardware_sender_enforces_payload_limits_before_dispatch():
    executor, device, _ = executor_fixture()
    executor.config.raw.update(profile_config().raw)
    executor.command_history.check_command_acceleration = True  # explicit opt-in regression
    executor.config.velocity_limits[:] = 2.
    executor.config.acceleration_limits[:] = 8.
    executor.holding.held_width = device.width
    device.force = 2.
    curve = BPoly.from_derivatives([0., 1.], [[np.zeros(7), np.zeros(7), np.zeros(7)],
                                            [np.ones(7)*.6, np.zeros(7), np.zeros(7)]])
    path = TimedPath(curve, executor.owner, 1, 'injected_payload_curve', 1., 0.)
    with pytest.raises(PickRejected, match='加速度') as error:
        executor._stream(path, origin=1.1, epoch=0, carrying=True)
    assert error.value.details['acceleration_limits_rad_s2'] == [2.]*7
    assert len(device.commands) > 0  # checks sampled motion rather than just a config label


def test_native_payload_route_retimes_and_certifies_with_the_loaded_profile():
    from nero_core.two_second.planning import MeasuredPlanner
    config = CellConfig.load(Path(__file__).parents[1]/
        'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    planner = MeasuredPlanner(config)
    planner.identity = (0, 1, 2)
    planner.plane = np.eye(4); planner.plane[2, 3] = -1.
    planner.support = np.array([0., 0., 1.]); planner.support_origin = planner.plane[:3, 3].copy()
    planner.belt_bounds = np.array([[-5., 5.], [-5., 5.]])
    planner.payload_local = np.empty((0, 3))  # explicit empty-cloud geometry fixture
    goal = config.standby.copy(); goal[0] += .1
    state = {'joints': config.standby.copy(), 'width_m': .1}
    loaded = planner.route(state, stage='test_payload', goal_joints=goal, payload=True)
    unloaded = planner.route(state, stage='test_empty', goal_joints=goal)
    assert loaded.diagnostics['motion_profile'] == 'loaded'
    assert loaded.diagnostics['joint_acceleration_limits_rad_s2'] == [2.]*7
    assert np.max(abs(np.asarray(extrema(loaded.curve, 2)))) < 2.
    assert loaded.duration > unloaded.duration
    for derivative in (1, 2):
        np.testing.assert_allclose(loaded.curve([0., loaded.duration], derivative), 0., atol=1e-8)


def test_3mm_retention_guard_is_preserved_and_reports_the_actual_width():
    cycle = PickCycle(None, None, PickSettings(speed_percent=100), lambda *a: None)
    cycle.held_width = .05479
    with pytest.raises(PickRejected, match='夹持开口异常') as error:
        cycle.check_holding({'width_m': .05786, 'force_n': 2.})
    assert error.value.details['width_change_m'] == pytest.approx(.00307)
    assert error.value.details['maximum_width_change_m'] == .003
    cycle.check_holding({'width_m': .05480, 'force_n': 2.})


def test_diagnostic_rotation_drift_uses_rotation_composition_across_pi():
    from experiments.two_second_grasp.simulation_report import summarize_retention
    base = {'phase': 'transfer', 'gripper_width_m': .05, 'confirmed_width_m': .05,
            'object_in_tcp_xyz_m': [0., 0., 0.], 'left_force_n': 1., 'right_force_n': 1.}
    report = summarize_retention([
        {**base, 'object_in_tcp_rotation_vector_rad': [0., 0., np.pi-.01]},
        {**base, 'object_in_tcp_rotation_vector_rad': [0., 0., -np.pi+.01]}])
    assert report['transfer']['maximum_relative_rotation_change_rad'] == pytest.approx(.02)
