"""Jacobian damped-least-squares inverse kinematics for MuJoCo.

The solver works on a private :class:`mujoco.MjData` instance.  Calling
``solve`` therefore never teleports the simulated robot or alters contacts in
the environment; it only returns a joint target for the regular actuators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ._math import (
    as_vector,
    config_from_mapping,
    normalized_quat,
    quat_error_vector,
)

try:  # Import lazily enough that policy-only use does not require MuJoCo.
    import mujoco
except ImportError:  # pragma: no cover - exercised on lightweight clients.
    mujoco = None  # type: ignore[assignment]


@dataclass(frozen=True)
class IKConfig:
    """Numerical and convergence parameters for DLS IK."""

    damping: float = 0.04
    step_size: float = 0.7
    max_iterations: int = 120
    position_tolerance: float = 5e-4
    orientation_tolerance: float = 2e-3
    position_weight: float = 1.0
    orientation_weight: float = 0.35
    max_joint_update: float | tuple[float, ...] = 0.12
    joint_centering_gain: float = 0.002
    joint_limit_avoidance_gain: float = 0.0

    def __post_init__(self) -> None:
        if self.damping <= 0.0:
            raise ValueError("IK damping must be positive")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("IK step_size must be in (0, 1]")
        if self.max_iterations < 1:
            raise ValueError("IK max_iterations must be at least one")
        if self.position_tolerance <= 0.0 or self.orientation_tolerance <= 0.0:
            raise ValueError("IK tolerances must be positive")
        if self.position_weight <= 0.0 or self.orientation_weight <= 0.0:
            raise ValueError("IK weights must be positive")
        if (
            self.joint_centering_gain < 0.0
            or self.joint_limit_avoidance_gain < 0.0
        ):
            raise ValueError("joint nullspace gains cannot be negative")


@dataclass(frozen=True)
class IKResult:
    """Rich result returned by :meth:`DampedLeastSquaresIK.solve`."""

    joint_positions: np.ndarray
    full_qpos: np.ndarray
    success: bool
    iterations: int
    position_error: float
    orientation_error: float
    message: str

    @property
    def qpos(self) -> np.ndarray:
        """Alias used by a few trajectory/IK libraries."""

        return self.joint_positions

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.joint_positions, dtype=dtype)


def _require_mujoco() -> Any:
    if mujoco is None:
        raise ImportError(
            "MuJoCo is required for inverse kinematics. Install the `mujoco` "
            "Python package before constructing DampedLeastSquaresIK."
        )
    return mujoco


def _name2id(model: Any, object_type: Any, name: str) -> int:
    mj = _require_mujoco()
    identifier = int(mj.mj_name2id(model, object_type, name))
    if identifier < 0:
        object_label = str(object_type).split(".")[-1]
        raise ValueError(f"MuJoCo {object_label} named {name!r} does not exist")
    return identifier


class DampedLeastSquaresIK:
    """Iterative pose IK over a configured set of 1-DoF arm joints."""

    def __init__(
        self,
        model: Any,
        data: Any,
        joint_names: Iterable[str],
        ee_site_name: str,
        config: IKConfig | Mapping[str, Any] | None = None,
    ) -> None:
        mj = _require_mujoco()
        self.model = model
        self.data = data
        self.config = config_from_mapping(IKConfig, config)
        self.joint_names = tuple(str(name) for name in joint_names)
        if not self.joint_names:
            raise ValueError("joint_names cannot be empty")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must not contain duplicates")

        self.joint_ids = np.asarray(
            [
                _name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
                for name in self.joint_names
            ],
            dtype=int,
        )
        # Compare integers: NumPy scalars and MuJoCo's pybind enums do not
        # consistently compare equal across supported dependency versions.
        supported = (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE))
        unsupported = [
            name
            for name, joint_id in zip(self.joint_names, self.joint_ids)
            if int(model.jnt_type[joint_id]) not in supported
        ]
        if unsupported:
            raise ValueError(
                "DLS IK supports hinge/slide arm joints only; unsupported: "
                + ", ".join(unsupported)
            )
        self.qpos_indices = np.asarray(model.jnt_qposadr[self.joint_ids], dtype=int)
        self.dof_indices = np.asarray(model.jnt_dofadr[self.joint_ids], dtype=int)
        self.ee_site_name = str(ee_site_name)
        self.ee_site_id = _name2id(
            model, mj.mjtObj.mjOBJ_SITE, self.ee_site_name
        )
        self.lower_limits, self.upper_limits = self._read_joint_limits()
        self.center_positions = self._joint_centers()
        self._scratch = mj.MjData(model)
        self._jacobian_position = np.zeros((3, model.nv), dtype=float)
        self._jacobian_rotation = np.zeros((3, model.nv), dtype=float)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def _read_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = np.full(self.dof, -np.inf, dtype=float)
        upper = np.full(self.dof, np.inf, dtype=float)
        for index, joint_id in enumerate(self.joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                lower[index], upper[index] = self.model.jnt_range[joint_id]
        return lower, upper

    def _joint_centers(self) -> np.ndarray:
        centers = np.zeros(self.dof, dtype=float)
        limited = np.isfinite(self.lower_limits) & np.isfinite(self.upper_limits)
        centers[limited] = 0.5 * (
            self.lower_limits[limited] + self.upper_limits[limited]
        )
        return centers

    def _seed_full_qpos(self, qpos_seed: Any | None) -> np.ndarray:
        if qpos_seed is None:
            return np.array(self.data.qpos, dtype=float, copy=True)
        seed = np.asarray(qpos_seed, dtype=float).reshape(-1)
        if not np.all(np.isfinite(seed)):
            raise ValueError("qpos_seed must contain only finite values")
        if seed.size == self.model.nq:
            return seed.copy()
        if seed.size == self.dof:
            full = np.array(self.data.qpos, dtype=float, copy=True)
            full[self.qpos_indices] = seed
            return full
        raise ValueError(
            f"qpos_seed must have {self.dof} arm values or {self.model.nq} "
            f"full-model values, got {seed.size}"
        )

    def _site_quaternion(self) -> np.ndarray:
        quat = np.empty(4, dtype=float)
        _require_mujoco().mju_mat2Quat(
            quat, self._scratch.site_xmat[self.ee_site_id]
        )
        return normalized_quat(quat)

    def _clip_joint_positions(self, positions: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(positions, self.lower_limits), self.upper_limits)

    def compute_jacobian(self, qpos: Any | None = None) -> np.ndarray:
        """Return the current 6-by-N site Jacobian for configured joints."""

        mj = _require_mujoco()
        self._scratch.qpos[:] = self._seed_full_qpos(qpos)
        self._scratch.qvel[:] = 0.0
        # A site Jacobian only depends on position-stage kinematics and the
        # subtree centres of mass.  ``mj_forward`` additionally runs
        # collision detection, constraint assembly and velocity/acceleration
        # stages which do not affect ``mj_jacSite``.  Angular trajectory
        # timing samples this method 33 times per edge, so using the minimal
        # equivalent MuJoCo stages keeps that safety calculation off the
        # control-loop critical path without changing its result.
        mj.mj_kinematics(self.model, self._scratch)
        mj.mj_comPos(self.model, self._scratch)
        mj.mj_jacSite(
            self.model,
            self._scratch,
            self._jacobian_position,
            self._jacobian_rotation,
            self.ee_site_id,
        )
        return np.vstack(
            (
                self._jacobian_position[:, self.dof_indices],
                self._jacobian_rotation[:, self.dof_indices],
            )
        ).copy()

    def forward_pose(self, qpos: Any | None = None) -> np.ndarray:
        """Return the configured site pose for an arm/full-model state."""

        mj = _require_mujoco()
        self._scratch.qpos[:] = self._seed_full_qpos(qpos)
        self._scratch.qvel[:] = 0.0
        mj.mj_forward(self.model, self._scratch)
        quaternion = np.empty(4, dtype=float)
        mj.mju_mat2Quat(
            quaternion,
            self._scratch.site_xmat[self.ee_site_id],
        )
        return np.r_[
            self._scratch.site_xpos[self.ee_site_id].copy(),
            normalized_quat(quaternion),
        ]

    def solve(
        self,
        target_position: Any,
        target_quaternion: Any | None = None,
        *,
        qpos_seed: Any | None = None,
        target_quat: Any | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> IKResult:
        """Solve for an end-effector pose without changing live simulation data.

        Args:
            target_position: Desired world-frame XYZ position.
            target_quaternion: Optional desired world-frame quaternion in
                MuJoCo ``[w, x, y, z]`` order.
            qpos_seed: Either N configured arm positions or a full ``model.nq``
                vector.  The live state is used when omitted.
            target_quat: Backward-compatible alias for ``target_quaternion``.
        """

        if target_quaternion is not None and target_quat is not None:
            raise ValueError("pass only one of target_quaternion and target_quat")
        if target_quat is not None:
            target_quaternion = target_quat
        target_pos = as_vector(target_position, 3, name="target_position")
        target_quaternion = (
            None
            if target_quaternion is None
            else normalized_quat(target_quaternion, name="target_quaternion")
        )
        full_qpos = self._seed_full_qpos(qpos_seed)
        full_qpos[self.qpos_indices] = self._clip_joint_positions(
            full_qpos[self.qpos_indices]
        )

        max_update = np.asarray(self.config.max_joint_update, dtype=float)
        if max_update.ndim == 0:
            max_update = np.full(self.dof, float(max_update))
        max_update = max_update.reshape(-1)
        if max_update.size != self.dof or np.any(max_update <= 0.0):
            raise ValueError(
                "max_joint_update must be a positive scalar or one value per joint"
            )

        mj = _require_mujoco()
        position_error_norm = np.inf
        orientation_error_norm = np.inf if target_quaternion is not None else 0.0
        iteration = 0
        success = False

        for iteration in range(1, self.config.max_iterations + 1):
            if cancelled is not None and cancelled():
                break
            self._scratch.qpos[:] = full_qpos
            self._scratch.qvel[:] = 0.0
            mj.mj_forward(self.model, self._scratch)

            position_error = (
                target_pos - self._scratch.site_xpos[self.ee_site_id]
            )
            position_error_norm = float(np.linalg.norm(position_error))
            if target_quaternion is None:
                orientation_error = np.empty(0, dtype=float)
                orientation_error_norm = 0.0
            else:
                orientation_error = quat_error_vector(
                    target_quaternion, self._site_quaternion()
                )
                orientation_error_norm = float(np.linalg.norm(orientation_error))

            if (
                position_error_norm <= self.config.position_tolerance
                and orientation_error_norm <= self.config.orientation_tolerance
            ):
                success = True
                break

            mj.mj_jacSite(
                self.model,
                self._scratch,
                self._jacobian_position,
                self._jacobian_rotation,
                self.ee_site_id,
            )
            position_jacobian = self._jacobian_position[:, self.dof_indices]
            if target_quaternion is None:
                jacobian = self.config.position_weight * position_jacobian
                error = self.config.position_weight * position_error
            else:
                rotation_jacobian = self._jacobian_rotation[:, self.dof_indices]
                jacobian = np.vstack(
                    (
                        self.config.position_weight * position_jacobian,
                        self.config.orientation_weight * rotation_jacobian,
                    )
                )
                error = np.r_[
                    self.config.position_weight * position_error,
                    self.config.orientation_weight * orientation_error,
                ]

            regularized = (
                jacobian @ jacobian.T
                + self.config.damping**2 * np.eye(jacobian.shape[0])
            )
            try:
                solved_error = np.linalg.solve(regularized, error)
                jacobian_inverse = jacobian.T @ np.linalg.solve(
                    regularized, np.eye(jacobian.shape[0])
                )
            except np.linalg.LinAlgError:
                # The damping should make this rare, but lstsq is a safe,
                # deterministic fallback for numerically pathological models.
                solved_error = np.linalg.lstsq(regularized, error, rcond=None)[0]
                jacobian_inverse = jacobian.T @ np.linalg.pinv(regularized)
            joint_update = jacobian.T @ solved_error

            if self.config.joint_centering_gain > 0.0:
                current = full_qpos[self.qpos_indices]
                center_error = self.center_positions - current
                limited = np.isfinite(self.lower_limits) & np.isfinite(
                    self.upper_limits
                )
                center_error[~limited] = 0.0
                nullspace = np.eye(self.dof) - jacobian_inverse @ jacobian
                joint_update += (
                    self.config.joint_centering_gain
                    * nullspace
                    @ center_error
                )

            if self.config.joint_limit_avoidance_gain > 0.0:
                current = full_qpos[self.qpos_indices]
                limited = np.isfinite(self.lower_limits) & np.isfinite(
                    self.upper_limits
                )
                half_range = 0.5 * (
                    self.upper_limits - self.lower_limits
                )
                normalized = np.zeros(self.dof, dtype=float)
                normalized[limited] = (
                    current[limited] - self.center_positions[limited]
                ) / half_range[limited]
                # Negative log-barrier gradient, projected into the task
                # nullspace.  It is negligible near mid-range and increases
                # smoothly before a wrist reaches its hard stop.
                denominator = np.maximum(
                    1.0 - normalized * normalized,
                    0.04,
                )
                avoidance = -normalized / denominator
                avoidance[~limited] = 0.0
                avoidance = np.clip(avoidance, -25.0, 25.0)
                nullspace = np.eye(self.dof) - jacobian_inverse @ jacobian
                joint_update += (
                    self.config.joint_limit_avoidance_gain
                    * nullspace
                    @ avoidance
                )

            joint_update = np.clip(
                self.config.step_size * joint_update, -max_update, max_update
            )
            updated = full_qpos[self.qpos_indices] + joint_update
            full_qpos[self.qpos_indices] = self._clip_joint_positions(updated)

        # Report errors for the returned configuration (the last loop error is
        # measured immediately before its update).
        self._scratch.qpos[:] = full_qpos
        self._scratch.qvel[:] = 0.0
        mj.mj_forward(self.model, self._scratch)
        position_error_norm = float(
            np.linalg.norm(target_pos - self._scratch.site_xpos[self.ee_site_id])
        )
        if target_quaternion is not None:
            orientation_error_norm = float(
                np.linalg.norm(
                    quat_error_vector(
                        target_quaternion, self._site_quaternion()
                    )
                )
            )
        success = (
            position_error_norm <= self.config.position_tolerance
            and orientation_error_norm <= self.config.orientation_tolerance
        )
        was_cancelled = bool(cancelled is not None and cancelled())
        message = (
            "converged"
            if success
            else (
                "cancelled"
                if was_cancelled
                else (
                    "maximum iterations reached "
                    f"(position error={position_error_norm:.6g}, "
                    f"orientation error={orientation_error_norm:.6g})"
                )
            )
        )
        return IKResult(
            joint_positions=full_qpos[self.qpos_indices].copy(),
            full_qpos=full_qpos.copy(),
            success=success,
            iterations=iteration,
            position_error=position_error_norm,
            orientation_error=orientation_error_norm,
            message=message,
        )

    def solve_joint_positions(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Convenience wrapper returning only the N arm joint positions."""

        return self.solve(*args, **kwargs).joint_positions


# Short aliases make configuration-driven construction less verbose.
JacobianIKSolver = DampedLeastSquaresIK
IKSolver = DampedLeastSquaresIK


def damped_least_squares_ik(
    model: Any,
    data: Any,
    joint_names: Iterable[str],
    ee_site_name: str,
    target_position: Any,
    target_quaternion: Any | None = None,
    *,
    qpos_seed: Any | None = None,
    config: IKConfig | Mapping[str, Any] | None = None,
) -> IKResult:
    """Stateless convenience function for one IK solve."""

    solver = DampedLeastSquaresIK(
        model=model,
        data=data,
        joint_names=joint_names,
        ee_site_name=ee_site_name,
        config=config,
    )
    return solver.solve(
        target_position,
        target_quaternion,
        qpos_seed=qpos_seed,
    )
