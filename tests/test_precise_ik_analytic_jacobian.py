"""Check the bounded IK analytical derivative against independent finite differences."""
from pathlib import Path
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation
from types import SimpleNamespace

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.precise_ik import PrecisePoseSolver, _right_jacobian_inverse


@pytest.fixture(scope='module')
def calibrated():
    root=Path(__file__).parents[1]
    config=CellConfig.load(root/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    model=calibrated_model(config);model.set_state(config.standby,.1)
    return config,model


@pytest.mark.parametrize('rotvec',[[0.,0.,0.],[.2,-.1,.3],[2.,-.4,.7],[1e-7,-1e-7,2e-7]])
def test_rotation_differential_matches_finite_difference(rotvec):
    r=Rotation.from_rotvec(rotvec)
    def f(x): return (r*Rotation.from_rotvec(x)).as_rotvec()
    numerical=approx_derivative(f,np.zeros(3),method='3-point')
    np.testing.assert_allclose(_right_jacobian_inverse(rotvec),numerical,atol=2e-8,rtol=2e-8)


def test_bounded_fallback_uses_derivative_and_finds_pose(calibrated,monkeypatch):
    cfg,model=calibrated;model.set_state(cfg.standby,.1)
    solver=PrecisePoseSolver(model)
    monkeypatch.setattr(solver.ik,'solve',lambda *a,**kw:SimpleNamespace(success=False))
    desired_q=cfg.standby+np.array([.015,-.02,.01,.02,-.015,.01,.01])
    model.set_state(desired_q,.1);pose=model.tcp().copy()
    model.set_state(cfg.standby,.1)
    quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
    result=solver.solve(pose[:3,3],quat,cfg.standby)
    assert result.success
    model.set_state(result.joint_positions,.1);actual=model.tcp()
    np.testing.assert_allclose(actual[:3,3],pose[:3,3],atol=1e-7)
    assert Rotation.from_matrix(actual[:3,:3].T@pose[:3,:3]).magnitude()<1e-6
