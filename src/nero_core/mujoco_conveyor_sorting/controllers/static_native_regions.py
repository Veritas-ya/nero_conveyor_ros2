"""Precomputed convex joint boxes for native self/fixed-world collision.

This is a conservative safety-certificate construction, not C-IRIS/SOS.  At
each valid seed, complete convex-support separation and configuration-global
motion bounds prove a whole joint box.  All native contacts at the bound
non-arm qpos are covered;
observed points, target motion and request kinematic constraints stay online.
"""
from __future__ import annotations

import time
import numpy as np
import mujoco

from .segment_certificate import (MujocoMotionBounds, native_collision_pairs,
                                  native_separation_lower_bound)

_CALLBACK_GETTERS = tuple(getattr(mujoco, name) for name in dir(mujoco)
                          if name.startswith('get_mjcb_'))


def _supported(model):
    # Callbacks may replace ordinary contact semantics.  Geometry proofs do
    # not certify arbitrary user code or non-rigid position dependencies.
    return (not any(int(getattr(model, name, 0)) for name in
                    ('nflex', 'nplugin', 'nmocap', 'ntendon'))
            and all(getter() is None for getter in _CALLBACK_GETTERS))


class StaticNativeRegions:
    """Immutable regions, matched to the complete compiled native world."""

    def __init__(self, *, model_digest, arm_joint_ids, arm_qpos_adrs,
                 robot_body_ids, full_qpos, lower, upper, boxes, report):
        self.model_digest = str(model_digest)
        self.arm_joint_ids = tuple(int(v) for v in arm_joint_ids)
        self.arm_qpos_adrs = np.asarray(arm_qpos_adrs, dtype=int).copy()
        self.robot_body_ids = tuple(sorted(int(v) for v in robot_body_ids))
        self.non_arm = np.setdiff1d(np.arange(len(full_qpos)), self.arm_qpos_adrs)
        self.non_arm_qpos = np.asarray(full_qpos, dtype=float)[self.non_arm].copy()
        self.lower, self.upper = np.asarray(lower).copy(), np.asarray(upper).copy()
        self.boxes = np.asarray(boxes, dtype=float).reshape(-1, 2, len(self.lower)).copy()
        for value in (self.arm_qpos_adrs, self.non_arm, self.non_arm_qpos,
                      self.lower, self.upper, self.boxes):
            value.setflags(write=False)
        self.report = dict(report)

    def compatible(self, model, arm_joint_ids, robot_body_ids):
        if (not _supported(model)
                or self.arm_joint_ids != tuple(int(v) for v in arm_joint_ids)
                or self.robot_body_ids != tuple(sorted(int(v) for v in robot_body_ids))):
            return False
        from .warm_roadmap import compiled_model_digest
        return compiled_model_digest(model) == self.model_digest

    def matches_qpos(self, full_qpos):
        full = np.asarray(full_qpos, dtype=float)
        return (full.shape == (len(self.non_arm) + len(self.arm_qpos_adrs),)
                and np.isfinite(full).all()
                and np.array_equal(full[self.non_arm], self.non_arm_qpos))

    def covers_box(self, lower, upper):
        low, high = np.asarray(lower), np.asarray(upper)
        if (low.shape != self.lower.shape or high.shape != low.shape
                or not np.isfinite([low, high]).all() or np.any(low > high)):
            return False
        return bool(np.any(np.all((low >= self.boxes[:, 0])
                                 & (high <= self.boxes[:, 1]), axis=1)))

    def contains(self, q):
        return self.covers_box(q, q)


def prepare_static_native_regions(model, *, arm_joint_ids, ee_site_id,
        robot_body_ids, full_qpos, lower, upper, seeds, model_digest=None,
        cancelled=None):
    """Build before recognition; a failed proof merely publishes no region.

    If L[p] bounds pair p's relative displacement and d[p] is its conservative
    clearance (already minus the original 1e-6 m rejection threshold), the
    symmetric box radius r is certified when L @ r < d for every varying pair.
    Coordinate inflation grows an inscribed box, never relaxes that inequality.
    """
    started = time.perf_counter()
    stop = cancelled or (lambda: False)
    if not _supported(model) or stop():
        return None
    joints = np.asarray(arm_joint_ids, dtype=int)
    qpos_adrs = np.asarray(model.jnt_qposadr[joints], dtype=int)
    snapshot = np.asarray(full_qpos, dtype=float).copy()
    low, high = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if (snapshot.shape != (model.nq,) or low.shape != joints.shape
            or high.shape != low.shape or not np.isfinite([low, high]).all()
            or not np.isfinite(snapshot).all() or np.any(low >= high)):
        raise ValueError('invalid static region inputs')
    for axis, joint in enumerate(joints):
        if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
            if (not model.jnt_limited[joint] or low[axis] < model.jnt_range[joint, 0]
                    or high[axis] > model.jnt_range[joint, 1]):
                return None
    bounds = MujocoMotionBounds(model, joints, int(ee_site_id), full_qpos=snapshot)
    # RobotModel's external validator also inspects contacts outside the
    # worker's robot-body subset.  Certify the stronger ALL-native scope so
    # both checks can consume the same proof, including object_* geometry
    # at its exactly bound non-arm qpos.
    pairs = native_collision_pairs(model, range(model.nbody), ignore_object_contacts=False)
    # A moving infinite plane is not enclosed by geom_rbound.  Fixed world
    # planes are supported by the existing convex-support distance routine.
    for first, second in pairs:
        for geom in (first, second):
            if (int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                    and np.any(bounds.geom_affected[geom])):
                return None
    coefficients = np.asarray([bounds.pair_coefficients(a, b)
                               for a, b in pairs]).reshape(len(pairs), len(joints))
    if not np.isfinite(coefficients).all():
        return None
    varying = np.any(coefficients > 0, axis=1)
    varying_pairs = [p for p, flag in zip(pairs, varying) if flag]
    varying_coeff = coefficients[varying]
    # Invariant pairs are checked at each seed using the original contacts;
    # their complete relative geometry is independent of every arm joint.
    bodies = frozenset(int(v) for v in robot_body_ids)
    data = mujoco.MjData(model)
    boxes, rejected = [], 0
    for seed in seeds:
        if stop():
            return None  # never publish an interrupted preprocessing result
        center = np.asarray(seed, dtype=float)
        if (center.shape != low.shape or not np.isfinite(center).all()
                or np.any(center < low) or np.any(center > high)):
            continue
        data.qpos[:] = snapshot
        data.qpos[qpos_adrs] = center
        mujoco.mj_fwdPosition(model, data)
        invalid = any(float(contact.dist) <= 1e-6
                      for contact in data.contact[:data.ncon])
        if invalid:
            rejected += 1
            continue
        gap = np.asarray([native_separation_lower_bound(model, data, a, b)
                          for a, b in varying_pairs]) - 1e-10
        if not np.isfinite(gap).all() or np.any(gap <= 0):
            rejected += 1
            continue
        cap = np.minimum(center-low, high-center)
        row_sum = varying_coeff.sum(axis=1)
        initial = (float(np.min(gap / row_sum)) * .5 if len(gap)
                   else float(np.max(cap)))
        radius = np.minimum(cap, initial)
        # Allocate remaining proven clearance to axes one by one.  All bounds
        # are global over this box, not a linearization at the seed.
        for axis in range(len(joints)):
            affected = varying_coeff[:, axis] > 0
            allowance = cap[axis]-radius[axis]
            if np.any(affected):
                slack = gap-varying_coeff @ radius
                allowance = min(allowance, float(np.min(
                    slack[affected]/varying_coeff[affected, axis])))
            radius[axis] += max(0., allowance)*(1.-1e-12)
        # Later coordinates can consume the last representable slack of an
        # earlier active face.  Shrink the published box, then retain the
        # strict full-matrix test; no boundary equality is treated as proof.
        radius *= 1.-1e-9
        if np.any(varying_coeff @ radius >= gap):
            rejected += 1
            continue
        a, b = center-radius, center+radius
        moving = radius > 0
        a[moving] = np.nextafter(a[moving], np.inf)
        b[moving] = np.nextafter(b[moving], -np.inf)
        # Check the actually represented, rounded box against the same proof;
        # adding a tiny radius to a large center must not enlarge its domain.
        actual_radius = np.maximum(center-a, b-center)
        if (np.any(a > b) or np.any(a < low) or np.any(b > high)
                or np.any(varying_coeff @ actual_radius >= gap)):
            rejected += 1
            continue
        boxes.append(np.stack((a, b)))
    if stop() or not _supported(model):
        return None
    if model_digest is None:
        from .warm_roadmap import compiled_model_digest
        model_digest = compiled_model_digest(model)
    return StaticNativeRegions(model_digest=model_digest, arm_joint_ids=joints,
        arm_qpos_adrs=qpos_adrs, robot_body_ids=bodies, full_qpos=snapshot,
        lower=low, upper=high, boxes=boxes,
        report={'method': 'convex_joint_boxes_from_conservative_motion_bounds',
                'scope': 'all_native_contacts_at_exact_fixed_non_arm_qpos',
                'regions': len(boxes), 'unproved_seeds': rejected,
                'native_pairs': len(pairs), 'varying_pairs': len(varying_pairs),
                'preprocessing_wall_s': time.perf_counter()-started})
