from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.graspgenx_sim.simulation_envelope import align_simulation_command_envelope


@dataclass(frozen=True)
class Config:
    minimum_joint_limit_margin: float = .035


def controller(margin):
    return SimpleNamespace(joint_lower_limits=np.full(7,-2.),joint_upper_limits=np.full(7,2.),
        command_joint_lower_limits=np.full(7,-2.+margin),command_joint_upper_limits=np.full(7,2.-margin),
        config=SimpleNamespace(joint_command_margin=margin))


def test_common_envelope_tightens_without_widening():
    p=controller(.02);a=controller(.055)
    ctx=SimpleNamespace(robot_controller=p,policy=SimpleNamespace(config=Config()))
    report=align_simulation_command_envelope(ctx,a,np.zeros(7))
    np.testing.assert_allclose(p.command_joint_upper_limits,np.full(7,1.945))
    assert ctx.policy.config.minimum_joint_limit_margin==.055
    assert report['source'].startswith('intersection')


def test_feedback_and_changed_hard_limits_fail_closed():
    p=controller(.02);a=controller(.055)
    ctx=SimpleNamespace(robot_controller=p,policy=SimpleNamespace(config=Config()))
    with pytest.raises(ValueError,match='outside'):
        align_simulation_command_envelope(ctx,a,np.full(7,1.99))
    a.joint_upper_limits[0]=1.9
    with pytest.raises(ValueError,match='hard joint limits differ'):
        align_simulation_command_envelope(ctx,a,np.zeros(7))
