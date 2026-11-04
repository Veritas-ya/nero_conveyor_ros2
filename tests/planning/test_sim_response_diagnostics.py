"""Admission diagnostics explain ordinary fallback without granting a certificate."""
from dataclasses import replace

from test_joint_response import _axis
from mujoco_conveyor_sorting.controllers.joint_response import JointResponseModel
from experiments.continuous_intercept_sim.response_diagnostics import response_model_status


def test_missing_model_explicitly_explains_ordinary_reference():
    result = response_model_status(None)
    assert not result['configured'] and not result['eligible']
    assert result['reason'] == 'no_explicit_model_file'
    assert result['command_reference'] == 'nominal'


def test_holdout_failure_names_joint_and_does_not_hide_values():
    axes = [_axis(.01, .1) for _ in range(7)]
    axes[5] = replace(axes[5], holdout_rmse_rad=1.)
    model = JointResponseModel(tuple(axes), {'runtime_eligible': True})
    result = response_model_status(model, path='explicit.json')
    assert not result['eligible'] and result['seven_axes']
    assert result['axes'][5]['checks']['holdout_not_worse'] is False
    assert result['axes'][5]['values']['holdout_rmse_rad'] == 1.
    assert result['axes'][0]['checks']['holdout_not_worse'] is True


def test_eligible_model_still_requires_full_command_certificate():
    model = JointResponseModel(tuple(_axis(.01, .1) for _ in range(7)),
        {'runtime_eligible': True})
    result = response_model_status(model)
    assert result['eligible']
    assert result['reason'] == 'eligible_requires_command_certificate'
    assert result['command_reference'] == 'planner_decides_after_full_certificate'
