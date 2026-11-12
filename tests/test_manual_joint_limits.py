"""User-supplied manual bounds must agree in simulation and hardware mirror."""
from pathlib import Path
import mujoco
import numpy as np
import pytest
import yaml
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel


ROOT=Path(__file__).resolve().parents[1]/'src/nero_core/mujoco_conveyor_sorting'
DEGREES=np.array([[-157,157],[-102,102],[-160,160],[-60,125],
                  [-160,160],[-44,57],[-97,97]],dtype=float)


def test_manual_ranges_match_simulation_model_actuators_and_config():
    expected=np.deg2rad(DEGREES)
    scene=mujoco.MjModel.from_xml_path(str(ROOT/'models/scene.xml'))
    joints=[scene.joint(f'joint{i}').id for i in range(1,8)]
    actuators=[scene.actuator(f'joint{i}').id for i in range(1,8)]
    np.testing.assert_allclose(scene.jnt_range[joints],expected,atol=1e-12,rtol=0)
    np.testing.assert_allclose(scene.actuator_ctrlrange[actuators],expected,atol=1e-12,rtol=0)
    cfg=yaml.safe_load((ROOT/'configs/robot.yaml').read_text())['robot']['joint_limits']
    np.testing.assert_allclose(np.column_stack([cfg['lower'],cfg['upper']]),expected,atol=1e-12,rtol=0)


def test_hardware_kinematic_mirror_uses_same_ranges_with_existing_margin():
    model=RobotModel()
    expected=np.deg2rad(DEGREES)
    np.testing.assert_allclose(model.limits,expected,atol=1e-12,rtol=0)
    assert model.joint_margin_rad==.02
    np.testing.assert_allclose(model.command_lower,expected[:,0]+.02,atol=1e-12,rtol=0)
    np.testing.assert_allclose(model.command_upper,expected[:,1]-.02,atol=1e-12,rtol=0)


def test_j6_original_range_survives_start_configuration_and_matches_ik():
    margins = [.02]*7
    margins[5] = 0.
    model = RobotModel(joint_position_margins_rad=margins)
    expected = np.deg2rad(DEGREES)
    np.testing.assert_allclose(model.command_lower, expected[:, 0]+margins, atol=1e-12)
    np.testing.assert_allclose(model.command_upper, expected[:, 1]-margins, atol=1e-12)
    np.testing.assert_allclose(model.ik.lower_limits, model.command_lower)
    np.testing.assert_allclose(model.ik.upper_limits, model.command_upper)
    start = np.zeros(7)
    start[0] = model.command_upper[0]+.001
    start[5] = np.deg2rad(56.5)
    report = model.configure_start_joint_tolerance(start)
    assert report['expanded_joints'] == ['joint1']
    np.testing.assert_allclose([model.command_lower[5], model.command_upper[5]],
                               expected[5], atol=1e-12)
    assert model.command_upper[0] == pytest.approx(start[0]+.003)
    model.configure_start_joint_tolerance(start)
    np.testing.assert_allclose(model.nominal_command_upper, expected[:, 1]-margins)


@pytest.mark.parametrize('margins', [[0.]*6, [-.001]*7, [float('nan')]*7,
                                    [float('inf')]*7, [2.]*7])
def test_invalid_position_margins_cannot_extend_model_ranges(margins):
    with pytest.raises(ValueError, match='关节位置余量'):
        RobotModel(joint_position_margins_rad=margins)
