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


@pytest.mark.parametrize('angle', [0., 1e-13, 1e-8, .7, np.pi-1e-8])
def test_site_error_is_shortest_world_rotation_for_either_quaternion_sign(calibrated, angle):
    cfg, model = calibrated
    solver = PrecisePoseSolver(model).ik
    pose = solver.forward_pose(cfg.standby)
    current = Rotation.from_quat(pose[[4, 5, 6, 3]])
    axis = np.array([1., -2., 3.])/np.sqrt(14.)
    target = (Rotation.from_rotvec(axis*angle)*current).as_quat()[[3, 0, 1, 2]]
    for sign in (-1, 1):
        np.testing.assert_allclose(solver._orientation_error(sign*target), axis*angle, atol=3e-14)


@pytest.mark.parametrize('oriented,centering,avoidance', [(False, 0., 0.), (True, 0., 0.),
                                                         (True, .002, .003)])
def test_ik_kinematics_matches_full_forward_without_moving_live_state(
        calibrated, monkeypatch, oriented, centering, avoidance):
    from dataclasses import replace
    cfg, model = calibrated
    solver = PrecisePoseSolver(model).ik
    solver.config = replace(solver.config, joint_centering_gain=centering,
                            joint_limit_avoidance_gain=avoidance)
    pose = solver.forward_pose(cfg.standby + np.array([.03, -.02, .02, .01, -.01, -.02, .01]))
    live = model.data.qpos.copy()
    args = (pose[:3], pose[3:] if oriented else None)
    fast = solver.solve(*args, qpos_seed=cfg.standby)
    monkeypatch.setattr(model.mj, 'mj_kinematics', model.mj.mj_forward)
    full = solver.solve(*args, qpos_seed=cfg.standby)
    np.testing.assert_array_equal(fast.joint_positions, full.joint_positions)
    np.testing.assert_array_equal(model.data.qpos, live)
    assert (fast.success, fast.iterations, fast.position_error, fast.orientation_error) == (
        full.success, full.iterations, full.position_error, full.orientation_error)
