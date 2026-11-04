"""Immutable-request adapter between shared FK facts and segment certificates."""

from dataclasses import asdict
import time

import mujoco
import numpy as np

from .segment_certificate import (
    AdaptiveSegmentValidator, CollisionPair, ConservativeSegmentCertificate,
    MujocoMotionBounds, SegmentConstraints, SegmentSample,
    convex_point_clearance_lower_bound, native_collision_pairs,
    native_separation_lower_bound,
)


class _LazyClearances(dict):
    def __init__(self, owner, full_qpos):
        super().__init__()
        self.owner, self.full_qpos = owner, full_qpos
        self.feature = owner.engine.state_evaluator.current
        self.native_gaps = None
        self.point_spheres = {}
        self.refined = set()

    def get(self, key, default=None):
        if key not in self:
            # Endpoint A may no longer be the shared scratch state's current
            # pose. Restore the cached immutable FK facts before any distance.
            if self.owner.engine.state_evaluator.current is not self.feature:
                self.owner.engine.state_evaluator.prepare(self.full_qpos)
            self[key] = self.owner.clearance(key, self)
        return self[key]

    def get_for_bound(self, key, required):
        value = self.get(key)
        if value <= required and key not in self.refined:
            if self.owner.engine.state_evaluator.current is not self.feature:
                self.owner.engine.state_evaluator.prepare(self.full_qpos)
            began = time.perf_counter()
            self[key] = max(value, self.owner.refine_clearance(key, self, required))
            self.owner.metrics['clearance_refinements'] += 1
            self.owner.metrics['refinement_wall_s'] += time.perf_counter()-began
            self.refined.add(key)
        return self[key]


class PlanningSegmentValidator:
    """One constraint height and one immutable request/snapshot publication.

    Unknown custom validators, payload-dependent geometry and flex/plugin
    models retain the original dense checker.  A planning/released target is
    the same immutable union of points used by the existing worker.  These
    certificates describe that union, not a newly invented time sweep.
    """

    @classmethod
    def available(cls, engine):
        if (engine.state_evaluator is None or not engine.state_evaluator.cache_enabled
                or engine.model.nflex or engine.model.nplugin):
            return False
        validator = engine._state_validator
        if validator is None:
            return True
        geometry = getattr(validator, "geometry", None)
        required = ("meshes", "mesh_bounds", "collision_radii", "fingers", "scene", "tree")
        return (getattr(validator, 'segment_validation_semantics', None)
                == 'nero_observed_convex_v1' and geometry is not None
                and all(hasattr(geometry, name) for name in required))

    def __init__(self, engine, request, cancelled, *, minimum_tcp_height=None):
        self.engine, self.request, self.cancelled = engine, request, cancelled
        self.height = (request.minimum_tcp_height if minimum_tcp_height is None
                       else float(minimum_tcp_height))
        self.geometry = getattr(engine._state_validator, "geometry", None)
        collision_for = getattr(engine._state_validator, "collision_for", None)
        self.collision = collision_for(request) if callable(collision_for) else None
        self.supported = not (callable(collision_for) and (self.collision is None
            or self.collision.kind not in {"planning", "released", "live"}))
        self.metrics = dict(clearance_queries=0, clearance_wall_s=0.,
            clearance_refinements=0, refinement_wall_s=0.,
            native_support_queries=0, native_aabb_proofs=0,
            point_sphere_proofs=0, point_halfspace_queries=0)
        if not self.supported:
            return
        self.bounds = MujocoMotionBounds(engine.model, engine.ik.joint_ids,
            engine.ee_site_id, full_qpos=request.full_qpos,
            geom_radii_m=getattr(self.geometry, "collision_radii", None))
        self.native_pairs = native_collision_pairs(engine.model,
                                                  engine.robot_collision_body_ids)
        self.native_indices = {f"native:{a}:{b}": i
                               for i, (a, b) in enumerate(self.native_pairs)}
        from ..runtime.simd_broadphase import ConvexBroadPhase
        local = engine.model.geom_aabb
        self.native_kernel = ConvexBroadPhase(np.stack(
            (local[:, :3]-local[:, 3:], local[:, :3]+local[:, 3:]), axis=1))
        self._native_array = np.asarray(self.native_pairs, dtype=np.int64).reshape(-1, 2)
        self._native_has_plane = np.array([
            int(engine.model.geom_type[a]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
            or int(engine.model.geom_type[b]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
            for a, b in self.native_pairs])
        self.pairs = {}
        constraints = []
        for first, second in self.native_pairs:
            name = f"native:{first}:{second}"
            self.pairs[name] = ("native", first, second)
            constraints.append(CollisionPair(name, first, second))
        geometry = self.geometry
        if geometry is not None:
            self._geom_ids = tuple(geometry.meshes)
            self._geom_index = {g: i for i, g in enumerate(self._geom_ids)}
            self._radii = np.array([max(float(geometry.collision_radii[g]),
                float(engine.model.geom_rbound[g])) for g in self._geom_ids])
            for geom in geometry.meshes:
                name = f"scene:{geom}"
                self.pairs[name] = ("scene", geom, geometry.scene_collision_margin(geom))
                constraints.append(CollisionPair(name, geom))
                if geom in geometry.fingers:
                    continue  # Existing intentional selected-target contact.
                cloud = self.collision.non_finger if self.collision is not None else None
                target = geometry.target if cloud is None else cloud.points
                if not len(target):
                    continue
                margin = .004 if self.collision is None else self.collision.non_finger_margin
                name = f"object:{geom}"
                self.pairs[name] = ("object", geom, margin)
                constraints.append(CollisionPair(name, geom))
        self.snapshot = (id(request), id(self.collision),
                         id(getattr(geometry, "scene", None)),
                         id(getattr(geometry, "target", None)))
        self.samples = {}
        self._region_lower = request.lower_limits+request.minimum_joint_margin
        self._region_upper = request.upper_limits-request.minimum_joint_margin
        if geometry is not None:
            self._region_lower = np.maximum(self._region_lower, geometry.command_lower)
            self._region_upper = np.minimum(self._region_upper, geometry.command_upper)
            if request.stage != 'pregrasp':
                self._region_lower = np.maximum(self._region_lower, geometry.nominal_command_lower)
                self._region_upper = np.minimum(self._region_upper, geometry.nominal_command_upper)
        self._proof_boxes = []
        self._proof_box_array = None
        self.certificate = ConservativeSegmentCertificate(self.bounds,
            SegmentConstraints(self.height, request.minimum_downward_alignment,
                request.singularity_hard_stop_condition, tuple(constraints)),
                on_proof=self._remember_region,
                precertified_collision_pairs=self._precertified_native_pairs)
        self.checker = AdaptiveSegmentValidator(self.sample, self.certificate,
            lower=request.lower_limits+request.minimum_joint_margin,
            upper=request.upper_limits-request.minimum_joint_margin,
            resolution=request.planner_config.normalized_edge_resolution,
            cancelled=cancelled, certified_region=self._known_region)

    def _precertified_native_pairs(self, sample, radius):
        query = getattr(self.engine, '_static_native_region_covers', None)
        # _anchor_proves also remembers a symmetric joint box.  Certify that
        # WHOLE box here, not merely the one-sided chord's two endpoints.
        if callable(query) and query(self.request, sample.q-radius, sample.q+radius):
            return self.native_indices.keys()
        return ()

    def _remember_region(self, sample, radius):
        # The motion and Jacobian bounds use componentwise |dq| and cover this
        # entire orthotope, not merely the original line. Round the published
        # box inward so its numerical boundaries cannot enlarge that domain.
        low, high = sample.q-radius, sample.q+radius
        varying = radius > 0
        low[varying] = np.nextafter(low[varying], np.inf)
        high[varying] = np.nextafter(high[varying], -np.inf)
        if hasattr(self, '_region_lower'):
            low = np.maximum(low, self._region_lower)
            high = np.minimum(high, self._region_upper)
        if np.any(low > high):
            return
        self._proof_boxes.append(np.stack((low, high)))
        self._proof_box_array = None

    def _known_region(self, start, end):
        if not self._proof_boxes:
            return False
        if self._proof_box_array is None:
            self._proof_box_array = np.asarray(self._proof_boxes)
        boxes = self._proof_box_array
        low, high = np.minimum(start, end), np.maximum(start, end)
        return bool(np.any(np.all((low >= boxes[:, 0]) & (high <= boxes[:, 1]), axis=1)))

    def sample(self, q):
        key = np.asarray(q, dtype=float).tobytes()
        if key in self.samples:
            return self.samples[key]
        engine, request = self.engine, self.request
        valid = engine._state_valid(request, q, self.cancelled,
                                    minimum_tcp_height=self.height)
        full = np.array(request.full_qpos, copy=True)
        full[engine.arm_qpos_adrs] = q
        if valid:
            engine.state_evaluator.prepare(full)
            data = engine.validation_data
            values = engine.state_evaluator.singular_values(engine.ee_site_id,
                                                            engine.ik.dof_indices)
            axis = data.site_xmat[engine.ee_site_id].reshape(3, 3) @ request.approach_axis_local
            sample = SegmentSample(np.array(q, copy=True), True,
                _LazyClearances(self, full), float(data.site_xpos[engine.ee_site_id, 2]),
                -float(axis[2]), float(values[-1]), float(values[0]),
                data.site_xpos[engine.ee_site_id].copy(),
                data.xanchor[engine.ik.joint_ids].copy(), self.snapshot,
                data.geom_xpos.copy())
        else:
            sample = SegmentSample(np.array(q, copy=True), False, {}, np.nan,
                np.nan, np.nan, np.nan, np.full(3, np.nan),
                np.full((len(q), 3), np.nan), self.snapshot)
        self.samples[key] = sample
        return sample

    def _native_clearance(self, name, first, second, cache):
        model, data = self.engine.model, self.engine.validation_data
        if cache.native_gaps is None:
            boxes = self.native_kernel.world_aabbs(data.geom_xpos[None],
                data.geom_xmat.reshape(1, model.ngeom, 3, 3))
            gaps = self.native_kernel.pair_clearances(boxes, self._native_array)[0]
            gaps[self._native_has_plane] = -np.inf
            cache.native_gaps = gaps-1e-6-1e-12
        lower = cache.native_gaps[self.native_indices[name]]
        if lower > 0:
            self.metrics["native_aabb_proofs"] += 1
            return float(lower)
        cache.refined.add(name)
        self.metrics["native_support_queries"] += 1
        return native_separation_lower_bound(model, data, first, second)

    def _point_clearance(self, kind, geom, margin, points, tree, cache,
                         *, force_exact=False, search_m=.025):
        if not len(points):
            return np.inf
        geometry, data = self.geometry, self.engine.validation_data
        center = data.geom_xpos[geom]
        radius = max(float(geometry.collision_radii[geom]),
                     float(self.engine.model.geom_rbound[geom]))
        if kind not in cache.point_spheres:
            nearest = tree.query(data.geom_xpos[list(self._geom_ids)])[0]
            cache.point_spheres[kind] = nearest-self._radii-1e-12
        sphere_lower = float(cache.point_spheres[kind][self._geom_index[geom]])
        if sphere_lower > .005 and not force_exact:
            self.metrics["point_sphere_proofs"] += 1
            return sphere_lower
        # Far points are at least search farther than the enclosing padded
        # sphere. Only near points need the original convex support planes.
        search = max(.025, search_m)
        ids = tree.query_ball_point(center, radius+search+1e-12)
        if not ids:
            return search
        local = (points[ids]-center) @ data.geom_xmat[geom].reshape(3, 3)
        cache.refined.add(f"{kind}:{geom}")
        self.metrics["point_halfspace_queries"] += 1
        return min(search, convex_point_clearance_lower_bound(local,
            geometry.meshes[geom][0], margin_m=margin))

    def clearance(self, name, cache):
        began = time.perf_counter()
        try:
            kind, geom, extra = self.pairs[name]
            if kind == "native":
                return self._native_clearance(name, geom, extra, cache)
            if kind == "scene":
                return self._point_clearance(kind, geom, extra, self.geometry.scene,
                                             self.geometry.tree, cache)
            if self.collision is not None:
                cloud = self.collision.non_finger
                points, tree = cloud.points, cloud.tree
            else:
                from scipy.spatial import cKDTree
                points = self.geometry.target
                if not hasattr(self, "_target_tree"):
                    self._target_tree = cKDTree(points)
                tree = self._target_tree
            return self._point_clearance(kind, geom, extra, points, tree, cache)
        finally:
            self.metrics["clearance_queries"] += 1
            self.metrics["clearance_wall_s"] += time.perf_counter()-began

    def refine_clearance(self, name, cache, required):
        """Only refine coarse bounds that cannot yet prove this segment."""
        kind, geom, extra = self.pairs[name]
        if kind == 'native':
            self.metrics['native_support_queries'] += 1
            return native_separation_lower_bound(self.engine.model,
                self.engine.validation_data, geom, extra)
        if kind == 'scene':
            points, tree = self.geometry.scene, self.geometry.tree
        elif self.collision is not None:
            points, tree = self.collision.non_finger.points, self.collision.non_finger.tree
        else:
            from scipy.spatial import cKDTree
            points = self.geometry.target
            if not hasattr(self, '_target_tree'):
                self._target_tree = cKDTree(points)
            tree = self._target_tree
        return self._point_clearance(kind, geom, extra, points, tree, cache,
                                    force_exact=True, search_m=required)

    def validate(self, start, end):
        return self.checker.validate(start, end)

    def diagnostics(self):
        if not self.supported:
            return {"supported": False, "scope": "original_dense_geometry_dependent_on_q"}
        return {"supported": True, "scope": "same_immutable_worker_union_and_scene",
                'common_rigid_joint_names': [
                    self.engine.model.joint(int(self.bounds.joint_ids[index])).name
                    for index in self.bounds.common_rigid_joint_indices],
                'jacobian_certificate_bound': 'singular_value_change_after_common_rigid_factor',
                'height_certificate_bound': 'world_z_projection_of_fixed_upstream_axis',
                **asdict(self.checker.stats), **self.metrics,
                'native_broadphase': dict(self.native_kernel.diagnostics),
                'inconclusive_anchor_bounds': dict(self.certificate.inconclusive_counts)}
