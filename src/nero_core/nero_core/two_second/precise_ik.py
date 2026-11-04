"""Private high-precision DLS/bounded IK used by the smooth reference."""
from dataclasses import replace
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK, IKResult
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def _right_jacobian_inverse(rotvec):
    """Differential of log(R exp(delta)) for a world-frame rotation vector."""
    x, y, z = np.asarray(rotvec, dtype=float)
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = float(np.linalg.norm(rotvec))
    factor = (1.-angle/(2.*np.tan(angle/2.)))/(angle*angle) if angle > 1e-5 else 1./12.
    return np.eye(3)+.5*skew+factor*(skew@skew)


class PrecisePoseSolver:
    def __init__(self, model):
        self.model = model
        self.data = model.mj.MjData(model.model)
        self.data.qpos[:] = model.data.qpos
        config = replace(model.ik.config, position_tolerance=1e-7,
            orientation_tolerance=1e-6, joint_centering_gain=0., max_iterations=8)
        self.ik = DampedLeastSquaresIK(model.model, self.data, model.names, 'ee_site', config)
        self.ik.lower_limits = model.command_lower.copy()
        self.ik.upper_limits = model.command_upper.copy()

    def solve(self, xyz, quat, seed, *, cancelled=lambda: False):
        if cancelled():
            raise PickRejected('平滑逆解已取消或超出预算')
        self.data.qpos[:] = self.model.data.qpos
        solved = self.ik.solve(xyz, quat, qpos_seed=seed, cancelled=cancelled)
        if solved.success:
            return solved
        target = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        def residual(q):
            if cancelled():
                raise PickRejected('平滑有界逆解已取消或超出预算')
            self.data.qpos[self.model.indices] = q
            self.model.mj.mj_kinematics(self.model.model, self.data)
            actual = self.data.site_xmat[self.model.tcp_id].reshape(3, 3)
            return 1000*np.r_[self.data.site_xpos[self.model.tcp_id]-xyz,
                self.ik.config.orientation_weight*Rotation.from_matrix(target@actual.T).as_rotvec()]
        def jacobian(q):
            if cancelled():
                raise PickRejected('平滑有界逆解已取消或超出预算')
            self.data.qpos[self.model.indices] = q
            self.model.mj.mj_kinematics(self.model.model, self.data)
            self.model.mj.mj_comPos(self.model.model, self.data)
            jp = np.zeros((3, self.model.model.nv))
            jr = np.zeros((3, self.model.model.nv))
            self.model.mj.mj_jacSite(self.model.model, self.data, jp, jr, self.model.tcp_id)
            actual = self.data.site_xmat[self.model.tcp_id].reshape(3, 3)
            error = Rotation.from_matrix(target@actual.T).as_rotvec()
            dofs = self.ik.dof_indices
            return 1000*np.vstack((jp[:, dofs],
                -self.ik.config.orientation_weight*_right_jacobian_inverse(error)@jr[:, dofs]))
        fit = least_squares(residual, np.clip(seed, self.ik.lower_limits, self.ik.upper_limits),
            bounds=(self.ik.lower_limits, self.ik.upper_limits), method='dogbox',
            jac=jacobian, max_nfev=150, ftol=1e-12, xtol=1e-12, gtol=1e-10)
        error = residual(fit.x)/1000
        position = float(np.linalg.norm(error[:3]))
        angle = float(np.linalg.norm(error[3:])/self.ik.config.orientation_weight)
        success = position <= self.ik.config.position_tolerance and angle <= self.ik.config.orientation_tolerance
        full = self.data.qpos.copy()
        full[self.model.indices] = fit.x
        return IKResult(fit.x.copy(), full, success, fit.nfev, position, angle,
                        f'smooth bounded IK: position={position:.6g}m rotation={angle:.6g}rad')
