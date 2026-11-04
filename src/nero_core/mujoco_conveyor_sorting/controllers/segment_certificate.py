"""Conservative certificates for linear joint-space segments.

This module never replaces an unknown condition with ``True``.  A certificate
requires lower distance bounds for *every* configured collision pair, plus
height, orientation and Jacobian-condition bounds.  Uncertified intervals go
back to the caller's original dense oracle.  Moving obstacles need a swept
union or an explicit displacement bound; a single frozen object pose does not
certify an actually moving object.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def native_collision_pairs(model, robot_body_ids: Sequence[int], *,
                           ignore_object_contacts: bool = True) -> tuple[tuple[int, int], ...]:
    """Configuration-independent superset of the worker's native contacts.

    Mirrors MuJoCo's bitmask, weld/parent and body-exclusion filters; explicit
    geom pairs bypass those filters.  The worker ignores ``object_*`` contacts
    because its measured object geometry is checked separately.  Sleep and
    broadphase-distance filters are intentionally omitted: they depend on the
    current state and cannot remove a pair from a whole-segment proof.

    A custom contact callback can replace geom-level mask filtering, so masks
    are not used when such a callback is installed.  Flex collision is not
    represented by geom pairs; callers must retain dense validation for flex
    models instead of interpreting this list as complete coverage.

    Filter source: MuJoCo engine_collision_driver.c, filterBodyPair,
    filterCollisionPair, and explicit-pair merge.  No runtime engine callback
    or model option is modified.
    """
    import mujoco

    if int(getattr(model, "nflex", 0)):
        raise ValueError("flex contacts need the original dense validator")
    robots = set(int(b) for b in robot_body_ids)
    explicit = {tuple(sorted((int(a), int(b))))
                for a, b in zip(model.pair_geom1, model.pair_geom2)}
    excluded = set(int(value) for value in model.exclude_signature)
    flags = int(model.opt.disableflags)
    no_parent_filter = bool(flags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
    callback = mujoco.get_mjcb_contactfilter()
    result = []
    for first in range(model.ngeom):
        body1 = int(model.geom_bodyid[first])
        for second in range(first+1, model.ngeom):
            body2 = int(model.geom_bodyid[second])
            robot1, robot2 = body1 in robots, body2 in robots
            if not (robot1 or robot2):
                continue
            other = body2 if robot1 else body1
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other) or ""
            if ignore_object_contacts and name.startswith("object_"):
                continue
            pair = (first, second)
            if pair not in explicit:
                weld1, weld2 = int(model.body_weldid[body1]), int(model.body_weldid[body2])
                if weld1 == weld2:
                    continue
                parent1 = int(model.body_weldid[int(model.body_parentid[weld1])])
                parent2 = int(model.body_weldid[int(model.body_parentid[weld2])])
                if (not no_parent_filter and weld1 != 0 and weld2 != 0
                        and (weld1 == parent2 or weld2 == parent1)):
                    continue
                low, high = sorted((body1, body2))
                if (low << 16)+high in excluded:
                    continue
                if callback is None and not (
                    int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
                    or int(model.geom_contype[second]) & int(model.geom_conaffinity[first])):
                    continue
            result.append(pair)
    return tuple(result)


def _geom_projection_interval(model, data, geom: int,
                              axis: np.ndarray) -> tuple[float, float]:
    """Support interval of a native convex geom on a unit world axis."""
    import mujoco

    kind = int(model.geom_type[geom])
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local_axis = rotation.T @ axis
    center = float(data.geom_xpos[geom] @ axis)
    size = model.geom_size[geom]
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom])
        start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        projection = model.mesh_vert[start:start+count].astype(float) @ local_axis
        return center+float(np.min(projection)), center+float(np.max(projection))
    if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        radius = float(size[0])
    elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius = float(size[0]+size[1]*abs(local_axis[2]))
    elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
        radius = float(np.abs(local_axis) @ size)
    elif kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        radius = float(np.linalg.norm(size*local_axis))
    elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius = float(size[0]*np.linalg.norm(local_axis[:2])
                       +size[1]*abs(local_axis[2]))
    else:
        return -np.inf, np.inf
    return center-radius, center+radius


def native_separation_lower_bound(model, data, first: int, second: int,
                                  *, search_distance_m: float = 0.10,
                                  margin_m: float = 1e-6) -> float:
    """Certified convex separation using support planes, not GJK distance.

    MuJoCo's distance result selects a promising axis only.  Projecting the
    complete native geoms onto *any* axis gives a conservative separation
    lower bound, so finite GJK termination error cannot overestimate clearance.
    The primitive ``margin_m`` is subtracted once, matching the caller's
    collision threshold.  Unsupported shapes are inconclusive.
    """
    import mujoco

    first, second = int(first), int(second)
    plane = int(mujoco.mjtGeom.mjGEOM_PLANE)
    plane_first, plane_second = (int(model.geom_type[g]) == plane
                                 for g in (first, second))
    if plane_first and plane_second:
        return -np.inf
    if plane_first or plane_second:
        ground, other = (first, second) if plane_first else (second, first)
        axis = data.geom_xmat[ground].reshape(3, 3)[:, 2]
        low, _ = _geom_projection_interval(model, data, other, axis)
        gap = low-float(data.geom_xpos[ground] @ axis)
    else:
        witness = np.zeros(6)
        mujoco.mj_geomDistance(model, data, first, second,
                              float(search_distance_m), witness)
        axis = witness[3:]-witness[:3]
        length = float(np.linalg.norm(axis))
        if length <= np.finfo(float).tiny or not np.isfinite(length):
            # When GJK returns only distmax, center direction remains a valid
            # projection axis; its tightness affects speed, never correctness.
            axis = data.geom_xpos[second]-data.geom_xpos[first]
            length = float(np.linalg.norm(axis))
        if length <= np.finfo(float).tiny or not np.isfinite(length):
            return -np.inf
        axis /= length
        a0, a1 = _geom_projection_interval(model, data, first, axis)
        b0, b1 = _geom_projection_interval(model, data, second, axis)
        gap = max(b0-a1, a0-b1)
    if not np.isfinite(gap):
        return -np.inf
    # Outward floating-point allowance for projections and support reductions.
    scale = 1+np.max(np.abs(data.geom_xpos[[first, second]]))
    scale += max(float(model.geom_rbound[first]), float(model.geom_rbound[second]))
    return float(gap-margin_m-128*np.finfo(float).eps*scale)


@dataclass(frozen=True)
class CollisionPair:
    """One complete collision constraint; distances already include its margin.

    ``second_geom=None`` denotes a fixed point cloud/world obstacle.  For a
    moving obstacle, ``obstacle_displacement`` must bound the obstacle over the
    *whole* segment.  Its half-segment bound is deliberately not inferred.
    """

    name: str
    first_geom: int
    second_geom: int | None = None
    obstacle_displacement_m: float = 0.0


@dataclass(frozen=True)
class SegmentSample:
    q: np.ndarray
    valid: bool
    clearances_m: Mapping[str, float]
    tcp_height_m: float
    downward_alignment: float
    sigma_min: float
    sigma_max: float
    tcp_position: np.ndarray
    joint_anchors: np.ndarray
    snapshot: Any = None
    geom_centers: np.ndarray | None = None


@dataclass(frozen=True)
class SegmentConstraints:
    minimum_tcp_height_m: float
    minimum_downward_alignment: float
    maximum_jacobian_condition: float
    collision_pairs: tuple[CollisionPair, ...]
    rotation_scale_m: float = 0.30
    numeric_slack: float = 1e-10


class MujocoMotionBounds:
    """Configuration-independent L1 bounds from the compiled kinematic tree.

    A hinge contribution is ``radius * abs(delta_q)``; a slide contribution is
    ``abs(delta_q)``.  Radii contain the complete descendant geometry, all body
    translations and joint-offset translations.  Thus they bound every point,
    rather than just the geom origin or the Jacobian at one configuration.
    Geom sizes use MuJoCo's enclosing ``geom_rbound``.  Arm joints must be hinge
    or slide joints; the remaining qpos is fixed for this request.
    """

    def __init__(self, model, arm_joint_ids: Sequence[int], ee_site_id: int,
                 *, full_qpos: np.ndarray | None = None,
                 geom_radii_m: Mapping[int, float] | None = None,
                 local_bounds: bool = False) -> None:
        import mujoco

        self.model = model
        self.joint_ids = np.asarray(arm_joint_ids, dtype=int)
        self.site_id = int(ee_site_id)
        self.use_local_bounds = bool(local_bounds)
        self.qpos = np.array(model.qpos0 if full_qpos is None else full_qpos,
                             dtype=float, copy=True)
        self.kinds = np.asarray(model.jnt_type[self.joint_ids], dtype=int)
        self.hinge = self.kinds == int(mujoco.mjtJoint.mjJNT_HINGE)
        self.slide = self.kinds == int(mujoco.mjtJoint.mjJNT_SLIDE)
        if len(self.joint_ids) == 0 or not np.all(self.hinge | self.slide):
            raise ValueError("segment bounds require hinge/slide arm joints")
        if self.qpos.shape != (model.nq,) or not np.isfinite(self.qpos).all():
            raise ValueError("invalid fixed qpos for segment bounds")
        self._arm_index = {int(j): i for i, j in enumerate(self.joint_ids)}
        radii = dict(geom_radii_m or {})
        for geom, radius in radii.items():
            if not np.isfinite(radius) or radius < 0:
                raise ValueError("geom radius must be finite and nonnegative")
            # MuJoCo's native rbound can be looser than the padded hull's exact
            # vertex radius.  Enclose both without rejecting a valid tighter
            # caller envelope or assuming either one is larger.
            radii[geom] = max(radius, float(model.geom_rbound[geom]))
        self.geom_coefficients = np.array([
            self._point_coefficients(int(model.geom_bodyid[g]),
                np.linalg.norm(model.geom_pos[g])+float(
                    radii.get(g, model.geom_rbound[g])))
            for g in range(model.ngeom)]).reshape(model.ngeom, len(self.joint_ids))
        self.geom_radii = np.array([radii.get(g, model.geom_rbound[g])
                                   for g in range(model.ngeom)])
        self.geom_affected = np.array([
            [int(model.jnt_bodyid[j]) in self._body_path(int(model.geom_bodyid[g]))
             for j in self.joint_ids] for g in range(model.ngeom)], dtype=bool).reshape(
                 model.ngeom, len(self.joint_ids))
        self.tcp_coefficients = self._point_coefficients(
            int(model.site_bodyid[self.site_id]),
            float(np.linalg.norm(model.site_pos[self.site_id])))
        self.anchor_coefficients = np.array([
            self._point_coefficients(int(model.jnt_bodyid[j]),
                                     float(np.linalg.norm(model.jnt_pos[j])))
            for j in self.joint_ids])
        self.axis_ancestors = np.array([
            [float(self.hinge[i] and int(source) != int(target)
                and int(model.jnt_bodyid[source]) in self._body_path(int(model.jnt_bodyid[target]))
                and (int(model.jnt_bodyid[source]) != int(model.jnt_bodyid[target])
                     or int(source) < int(target)))
             for i, source in enumerate(self.joint_ids)]
            for target in self.joint_ids])
        self.downstream = np.array([
            [float(int(source) != int(target)
                and int(model.jnt_bodyid[source]) in self._body_path(int(model.jnt_bodyid[target]))
                and (int(model.jnt_bodyid[source]) != int(model.jnt_bodyid[target])
                     or int(source) < int(target)))
             for target in self.joint_ids]
            for source in self.joint_ids])
        self.tcp_rotation_coefficients = self._axis_coefficients(
            int(model.site_bodyid[self.site_id]))
        self.common_rigid_joint_indices = self._common_rigid_joints()
        self._root_axis_z_projection = {}
        self._root_axis_cross_z = {}
        if self.common_rigid_joint_indices:
            scratch = mujoco.MjData(model)
            scratch.qpos[:] = self.qpos
            mujoco.mj_kinematics(model, scratch)
            for index in self.common_rigid_joint_indices:
                axis = scratch.xaxis[self.joint_ids[index]]
                rounding = 64*np.finfo(float).eps
                self._root_axis_z_projection[index] = min(1., abs(float(axis[2]))+rounding)
                self._root_axis_cross_z[index] = min(1., float(np.linalg.norm(
                    np.cross(axis, [0., 0., 1.])))+rounding)

    def _body_path(self, body: int) -> list[int]:
        path = []
        while body > 0:
            path.append(body)
            body = int(self.model.body_parentid[body])
        return path

    def _common_rigid_joints(self) -> tuple[int, ...]:
        """Arm coordinates that rigidly transform the complete Jacobian.

        A joint must precede every arm column and the TCP.  Same-body joints
        use compiled application order; merely sharing a body is insufficient.
        Everything above it is fixed by this request's non-arm qpos snapshot.
        Changing it rotates/translates the entire downstream mechanism.  For
        a hinge, geometric J becomes diag(R,R) @ J, an orthogonal left factor
        even with the existing .30 rotational scale.  A slide translates all
        anchors/TCP equally and leaves J unchanged.
        """
        if self.model.nmocap or self.model.nplugin:
            return ()
        targets = [int(self.model.jnt_bodyid[j]) for j in self.joint_ids]
        targets.append(int(self.model.site_bodyid[self.site_id]))
        found = []
        for index, joint in enumerate(self.joint_ids):
            body = int(self.model.jnt_bodyid[joint])
            if (all(body in self._body_path(target) for target in targets)
                    and all(int(self.model.jnt_bodyid[other]) != body or other >= joint
                            for other in self.joint_ids)):
                found.append(index)
        return tuple(found)

    def _body_joints(self, body: int):
        start = int(self.model.body_jntadr[body])
        return range(start, start+int(self.model.body_jntnum[body]))

    def _translation_bound(self, joint: int) -> float:
        import mujoco

        kind = int(self.model.jnt_type[joint])
        if kind == int(mujoco.mjtJoint.mjJNT_HINGE):
            # The p - R p term in the body transform has norm <= 2 ||p||.
            return 2*float(np.linalg.norm(self.model.jnt_pos[joint]))
        if kind == int(mujoco.mjtJoint.mjJNT_SLIDE):
            address = int(self.model.jnt_qposadr[joint])
            reference = float(self.model.qpos0[address])
            if joint in self._arm_index:
                if not self.model.jnt_limited[joint]:
                    return float("inf")
                return float(np.max(np.abs(self.model.jnt_range[joint]-reference)))
            return abs(float(self.qpos[address])-reference)
        # A non-arm free/ball joint on the robot chain is not covered.  Infinity
        # makes the certificate inconclusive and preserves dense validation.
        return float("inf")

    def _point_coefficients(self, body: int, local_radius: float) -> np.ndarray:
        path = self._body_path(body)
        result = np.zeros(len(self.joint_ids))
        for arm, joint in enumerate(self.joint_ids):
            joint = int(joint)
            joint_body = int(self.model.jnt_bodyid[joint])
            if joint_body not in path:
                continue
            if self.slide[arm]:
                result[arm] = 1.0
                continue
            below = path[:path.index(joint_body)+1]
            # All translations below this pivot are bounded independently of
            # q.  Keeping other joints in the same body covers multi-joint
            # bodies too.  Excluding this joint is valid: its rotation preserves
            # the distance to its own pivot.
            radius = local_radius+float(np.linalg.norm(self.model.jnt_pos[joint]))
            radius += sum(float(np.linalg.norm(self.model.body_pos[b]))
                          for b in below if b != joint_body)
            radius += sum(self._translation_bound(other)
                          for b in below for other in self._body_joints(b)
                          if other != joint)
            result[arm] = radius
        return result

    def _axis_coefficients(self, body: int) -> np.ndarray:
        path = self._body_path(body)
        return np.array([float(self.hinge[i] and
            int(self.model.jnt_bodyid[j]) in path)
            for i, j in enumerate(self.joint_ids)])

    def geom_displacement(self, delta_q: np.ndarray) -> np.ndarray:
        return self.geom_coefficients @ np.abs(delta_q)

    def pair_displacement(self, first: int, second: int,
                          delta_q: np.ndarray) -> float:
        # A joint upstream of BOTH geoms applies the same rigid transform to
        # them.  Their separation is invariant to that transform, even while
        # downstream joints move.  Removing these terms is exact and avoids
        # claiming a wrist/base sweep changes an internal gripper clearance.
        return float(self.pair_coefficients(first, second) @ np.abs(delta_q))

    def pair_coefficients(self, first: int, second: int) -> np.ndarray:
        common = self.geom_affected[first] & self.geom_affected[second]
        return np.where(common, 0.,
            self.geom_coefficients[first]+self.geom_coefficients[second])

    def tcp_displacement(self, delta_q: np.ndarray) -> float:
        return float(self.tcp_coefficients @ np.abs(delta_q))

    def tcp_height_displacement(self, delta_q: np.ndarray) -> float:
        """World-Z bound; does not replace the Euclidean TCP movement bound."""
        coefficients = self.tcp_coefficients.copy()
        for index in self.common_rigid_joint_indices:
            factor = (self._root_axis_cross_z[index] if self.hinge[index]
                      else self._root_axis_z_projection[index])
            coefficients[index] *= factor
        return float(coefficients @ np.abs(delta_q))

    def _local_point_coefficients(self, centers: np.ndarray, radii: np.ndarray,
                                  global_coefficients: np.ndarray,
                                  sample: SegmentSample,
                                  delta_q: np.ndarray) -> np.ndarray:
        # At the endpoint, pivot-to-point radius is known exactly (up to the
        # enclosing point radius). Only downstream joints can change it;
        # their complete interval sweep uses the global bounds above. Taking
        # the minimum of two genuine upper bounds remains conservative.
        distance = np.linalg.norm(centers[:, None, :]-sample.joint_anchors[None, :, :], axis=2)
        downstream_motion = (global_coefficients*np.abs(delta_q)) @ self.downstream.T
        local = distance+radii[:, None]+downstream_motion
        local[:, self.slide] = 1.
        return np.minimum(global_coefficients, local)

    def local_geom_coefficients(self, sample: SegmentSample,
                                 delta_q: np.ndarray) -> np.ndarray:
        if sample.geom_centers is None:
            return self.geom_coefficients
        return self._local_point_coefficients(sample.geom_centers, self.geom_radii,
            self.geom_coefficients, sample, delta_q)

    def tcp_displacement_at(self, sample: SegmentSample,
                             delta_q: np.ndarray) -> float:
        coefficients = self._local_point_coefficients(sample.tcp_position[None],
            np.zeros(1), self.tcp_coefficients[None], sample, delta_q)[0]
        return float(coefficients @ np.abs(delta_q))

    def rotation_displacement(self, delta_q: np.ndarray) -> float:
        return float(self.tcp_rotation_coefficients @ np.abs(delta_q))

    def downward_alignment_displacement(self, delta_q: np.ndarray) -> float:
        """World-Z tool-axis change, factoring known common-axis rotations."""
        intrinsic = np.array(delta_q, dtype=float, copy=True)
        contribution = 0.
        for index in self.common_rigid_joint_indices:
            if self.hinge[index]:
                chord = 2*np.sin(min(np.pi, abs(float(intrinsic[index])))/2)
                contribution += chord*self._root_axis_cross_z[index]
            intrinsic[index] = 0.
        remaining = 2*np.sin(min(np.pi, self.rotation_displacement(intrinsic))/2)
        return min(2., float(contribution+remaining))

    def jacobian_perturbation(self, sample: SegmentSample,
                              delta_q: np.ndarray, rotation_scale_m: float) -> float:
        """Spectral-norm upper bound via column/Frobenius bounds.

        For a hinge column Jp = axis x (tcp - anchor), bound the axis chord
        and both point displacements.  Weyl's inequality then encloses every
        singular value over the interval.  This does not extrapolate the SVD
        or Jacobian at the endpoint as if it were constant.
        """

        dq = np.abs(delta_q)
        tcp = (self.tcp_displacement_at(sample, dq) if self.use_local_bounds
               else self.tcp_displacement(dq))
        anchor_coefficients = (self._local_point_coefficients(sample.joint_anchors,
            np.zeros(len(self.joint_ids)), self.anchor_coefficients, sample, dq)
            if self.use_local_bounds else self.anchor_coefficients)
        anchors = anchor_coefficients @ dq
        angle = self.axis_ancestors @ dq
        axis_chord = 2*np.sin(np.minimum(np.pi, angle)/2)
        distance = np.linalg.norm(sample.joint_anchors-sample.tcp_position, axis=1)
        translation = np.where(self.hinge,
            axis_chord*distance+tcp+anchors, axis_chord)
        rotation = np.where(self.hinge, rotation_scale_m*axis_chord, 0.0)
        return float(np.linalg.norm(np.hypot(translation, rotation)))

    def singular_value_perturbation(self, sample: SegmentSample,
                                    delta_q: np.ndarray,
                                    rotation_scale_m: float) -> float:
        """Weyl bound after factoring a proven common rigid transform.

        This bounds changes of singular values, not ||J(q)-J(sample)||: the
        latter can be nonzero under a base rotation although singular values
        remain exactly unchanged.  Keep ``jacobian_perturbation`` for callers
        that need the original matrix-difference norm guarantee.
        """
        intrinsic = np.array(delta_q, dtype=float, copy=True)
        intrinsic[list(self.common_rigid_joint_indices)] = 0.
        return self.jacobian_perturbation(sample, intrinsic, rotation_scale_m)


class ConservativeSegmentCertificate:
    """Prove all constraints from an endpoint or from the two half-intervals."""

    def __init__(self, bounds: MujocoMotionBounds, constraints: SegmentConstraints,
                 *, on_proof: Callable[[SegmentSample, np.ndarray], None] | None = None,
                 precertified_collision_pairs=None):
        self.bounds = bounds
        self.constraints = constraints
        self.on_proof = on_proof
        self.precertified_collision_pairs = precertified_collision_pairs
        names = [p.name for p in constraints.collision_pairs]
        if len(names) != len(set(names)):
            raise ValueError("collision pair names must be unique")
        self.inconclusive_counts: dict[str, int] = {}
        self._pair_motion_coefficients = None
        if hasattr(bounds, "geom_coefficients"):
            self._pair_motion_coefficients = np.array([
                bounds.geom_coefficients[pair.first_geom]
                if pair.second_geom is None else
                bounds.pair_coefficients(pair.first_geom, pair.second_geom)
                for pair in constraints.collision_pairs]).reshape(
                    len(constraints.collision_pairs), len(bounds.joint_ids))
            self._pair_first = np.array([p.first_geom for p in constraints.collision_pairs], dtype=int)
            self._pair_second = np.array([-1 if p.second_geom is None else p.second_geom
                                         for p in constraints.collision_pairs], dtype=int)
            self._pair_common = np.array([np.zeros(len(bounds.joint_ids), dtype=bool)
                if p.second_geom is None else
                bounds.geom_affected[p.first_geom] & bounds.geom_affected[p.second_geom]
                for p in constraints.collision_pairs], dtype=bool).reshape(
                    len(constraints.collision_pairs), len(bounds.joint_ids))

    def _inconclusive(self, reason: str) -> bool:
        self.inconclusive_counts[reason] = self.inconclusive_counts.get(reason, 0)+1
        return False

    def _anchor_proves(self, sample: SegmentSample, delta_q: np.ndarray,
                       *, obstacle_fraction: float = 1.0) -> bool:
        c = self.constraints
        slack = float(c.numeric_slack)
        tcp_sweep = (self.bounds.tcp_height_displacement(delta_q)
                     if hasattr(self.bounds, 'tcp_height_displacement')
                     else self.bounds.tcp_displacement(delta_q))
        if not sample.tcp_height_m-tcp_sweep >= (
                c.minimum_tcp_height_m+slack):
            return self._inconclusive("tcp_height")
        if c.minimum_downward_alignment > -1.0:
            axis_chord = (self.bounds.downward_alignment_displacement(delta_q)
                if hasattr(self.bounds, 'downward_alignment_displacement') else
                2*np.sin(min(np.pi, self.bounds.rotation_displacement(delta_q))/2))
            if not sample.downward_alignment-axis_chord >= (
                    c.minimum_downward_alignment+slack):
                return self._inconclusive("downward_alignment")
        singular_bound = getattr(self.bounds, 'singular_value_perturbation',
                                self.bounds.jacobian_perturbation)
        epsilon = singular_bound(sample, delta_q, c.rotation_scale_m)
        sigma_min = sample.sigma_min-epsilon
        sigma_max = sample.sigma_max+epsilon
        if (np.isfinite(c.maximum_jacobian_condition) and
                (not np.isfinite([sigma_min, sigma_max]).all() or sigma_min <= slack
                 or sigma_max > c.maximum_jacobian_condition*sigma_min)):
            return self._inconclusive("jacobian_condition")
        # Geometry lower bounds can be lazy.  Do not ask for convex support
        # queries while a much cheaper kinematic bound is still inconclusive.
        if self._pair_motion_coefficients is not None:
            if sample.geom_centers is not None and self.bounds.use_local_bounds:
                geom_coefficients = self.bounds.local_geom_coefficients(sample, delta_q)
                coefficients = geom_coefficients[self._pair_first].copy()
                second = self._pair_second >= 0
                coefficients[second] += geom_coefficients[self._pair_second[second]]
                coefficients[self._pair_common] = 0.
                swept_bounds = coefficients @ np.abs(delta_q)
            else:
                swept_bounds = self._pair_motion_coefficients @ np.abs(delta_q)
        else:
            motion = self.bounds.geom_displacement(delta_q)
            swept_bounds = [motion[pair.first_geom] if pair.second_geom is None else
                self.bounds.pair_displacement(pair.first_geom, pair.second_geom, delta_q)
                for pair in c.collision_pairs]
        already_proven = (self.precertified_collision_pairs(sample, np.abs(delta_q))
                         if self.precertified_collision_pairs is not None else ())
        for pair, swept in zip(c.collision_pairs, swept_bounds):
            if pair.name in already_proven:
                continue
            swept += pair.obstacle_displacement_m*obstacle_fraction
            refined_get = getattr(sample.clearances_m, 'get_for_bound', None)
            clearance = (refined_get(pair.name, swept+slack) if refined_get is not None
                         else sample.clearances_m.get(pair.name))
            if clearance is None or np.isnan(clearance) or clearance == -np.inf:
                return self._inconclusive("missing_clearance:"+pair.name)
            if not clearance > swept+slack:
                return self._inconclusive("clearance:"+pair.name)
        if self.on_proof is not None:
            self.on_proof(sample, np.abs(delta_q))
        return True

    def __call__(self, left: SegmentSample, right: SegmentSample) -> bool:
        if not left.valid or not right.valid or left.snapshot != right.snapshot:
            return False
        dq = np.asarray(right.q)-np.asarray(left.q)
        if not np.isfinite(dq).all():
            return False
        if self._anchor_proves(left, dq) or self._anchor_proves(right, -dq):
            return True
        # Each half is certified from its nearest endpoint.  Only the robot's
        # affine joint displacement scales by 1/2.  A moving obstacle's complete
        # swept bound remains unchanged unless the caller provides subsegments.
        return (self._anchor_proves(left, dq/2)
                and self._anchor_proves(right, -dq/2))


@dataclass
class SegmentValidationStats:
    samples: int = 0
    certified_segments: int = 0
    split_segments: int = 0
    dense_fallback_segments: int = 0
    invalid_samples: int = 0
    cancellations: int = 0
    reused_regions: int = 0
    endpoint_only_segments: int = 0


class AdaptiveSegmentValidator:
    """Certify easy intervals, retain the original-grid oracle in hard ones.

    Grid boundaries are integer indices into the original dense edge samples.
    Thus uncertain leaves check exactly the same sample locations as before;
    midpoint refinement cannot shift the old grid and hide a narrow obstacle.
    A certificate covers every continuous configuration in skipped intervals.
    """

    def __init__(self, evaluate: Callable[[np.ndarray], SegmentSample],
                 certify: Callable[[SegmentSample, SegmentSample], bool],
                 *, lower: np.ndarray, upper: np.ndarray, resolution: float,
                 cancelled: Callable[[], bool] | None = None,
                 dense_valid: Callable[[np.ndarray], bool] | None = None,
                 leaf_grid_steps: int = 1,
                 certified_region: Callable[[np.ndarray, np.ndarray], bool] | None = None):
        self.evaluate = evaluate
        self.certify = certify
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.resolution = float(resolution)
        self.cancelled = cancelled or (lambda: False)
        self.dense_valid = dense_valid
        self.leaf_grid_steps = max(1, int(leaf_grid_steps))
        self.certified_region = certified_region
        self.stats = SegmentValidationStats()
        if (self.lower.ndim != 1 or self.lower.shape != self.upper.shape
                or not np.isfinite([self.lower, self.upper]).all()
                or np.any(self.lower >= self.upper) or self.resolution <= 0):
            raise ValueError("invalid adaptive segment bounds/resolution")

    def validate(self, start: np.ndarray, end: np.ndarray) -> bool:
        start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        if (start.shape != self.lower.shape or end.shape != self.lower.shape
                or not np.isfinite([start, end]).all()
                or np.any(start < self.lower) or np.any(start > self.upper)
                or np.any(end < self.lower) or np.any(end > self.upper)):
            return False
        samples = max(1, int(np.ceil(np.linalg.norm(
            (end-start)/(self.upper-self.lower))/self.resolution)))
        cache: dict[int, SegmentSample] = {}

        def q_at(index):
            # Keep the original endpoints exactly, including their final ulp.
            # The former dense checker validated these arrays, not a rebuilt
            # start + (end-start)*1 that can round to a different endpoint.
            if index == 0:
                return start
            if index == samples:
                return end
            return start+(end-start)*(index/samples)

        def sample(index: int) -> SegmentSample | None:
            if self.cancelled():
                self.stats.cancellations += 1
                return None
            if index not in cache:
                q = q_at(index)
                cache[index] = self.evaluate(q)
                self.stats.samples += 1
                if not cache[index].valid:
                    self.stats.invalid_samples += 1
            return cache[index]

        def visit(first: int, last: int) -> bool:
            if (not self.cancelled() and self.certified_region is not None
                    and self.certified_region(q_at(first), q_at(last))):
                self.stats.reused_regions += 1
                return not self.cancelled()
            a, b = sample(first), sample(last)
            if a is None or b is None or not a.valid or not b.valid:
                return False
            if last-first == 1:
                # Both adjacent points of the original finite-resolution grid
                # have already passed the complete state oracle. There are no
                # interior grid points for a certificate to replace: even an
                # inconclusive certificate would return this same result.
                # Avoid its distance queries and report dense validation, not
                # a proof covering all continuous configurations in the edge.
                self.stats.dense_fallback_segments += 1
                self.stats.endpoint_only_segments += 1
                return not self.cancelled()
            if self.certify(a, b):
                self.stats.certified_segments += 1
                return not self.cancelled()
            if last-first <= self.leaf_grid_steps:
                self.stats.dense_fallback_segments += 1
                # This is the old finite-resolution oracle, not a continuous
                # certificate.  Preserve that distinction in reported stats.
                missing = [q_at(i)
                           for i in range(first+1, last) if i not in cache]
                if self.dense_valid is not None and missing:
                    return bool(self.dense_valid(np.asarray(missing))) and not self.cancelled()
                return all((s := sample(i)) is not None and s.valid
                           for i in range(first+1, last)) and not self.cancelled()
            self.stats.split_segments += 1
            middle = (first+last)//2
            return visit(first, middle) and visit(middle, last)

        return visit(0, samples)


def convex_point_clearance_lower_bound(points: np.ndarray, equations: np.ndarray,
                                        *, margin_m: float = 0.0) -> float:
    """Lower bound distance from fixed points to an expanded convex polytope.

    Every normalized supporting-plane violation is <= Euclidean separation.
    Their maximum is therefore a lower bound; the minimum over points remains
    a lower bound for the complete cloud.  Negative values are inconclusive.
    This is not a signed-distance approximation used as an exact distance.
    """

    points = np.asarray(points, dtype=float).reshape(-1, 3)
    equations = np.asarray(equations, dtype=float).reshape(-1, 4)
    if not len(points):
        return float("inf")
    normals = equations[:, :3]
    lengths = np.linalg.norm(normals, axis=1)
    if (not len(equations) or not np.isfinite(points).all()
            or not np.isfinite(equations).all()
            or np.any(lengths <= 0)):
        return float("-inf")
    # Match the original unnormalized inequality n.x+b <= margin exactly.
    # Subtracting margin after normalization would enlarge a nonunit plane by
    # the wrong amount and could overestimate clearance when ||n|| < 1.
    values = (points @ normals.T+equations[:, 3]-margin_m)/lengths
    error = 64*np.finfo(float).eps*(1+np.max(np.abs(points))
                                  +np.max(np.abs(equations))/np.min(lengths))
    return float(np.min(np.max(values, axis=1))-error)
