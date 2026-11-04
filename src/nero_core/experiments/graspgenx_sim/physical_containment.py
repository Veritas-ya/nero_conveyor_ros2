"""Read-only collision geometry bounds for the offline placement evaluator.

MuJoCo stores centred/aligned mesh vertices.  The compiler folds the mesh
asset transform into the geom frame, so ``geom_xpos + mesh_vert @ geom_xmat.T``
is the world mapping; applying ``mesh_pos/mesh_quat`` again is incorrect.
The convex graph layout is documented in MuJoCo API types, "Convex hulls":
https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#convex-hulls
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Bounds:
    lower: np.ndarray
    upper: np.ndarray
    source: str
    fallback_reasons: tuple[str, ...] = ()

    def diagnostics(self) -> dict[str, Any]:
        # Unknown geometry is deliberately unbounded internally, but persisted
        # results must remain standards-compliant JSON rather than Infinity.
        lower = [float(value) if np.isfinite(value) else None for value in self.lower]
        upper = [float(value) if np.isfinite(value) else None for value in self.upper]
        return {"lower_world_m": lower, "upper_world_m": upper,
                "source": self.source, "fallback_reasons": list(self.fallback_reasons)}


def catalog_bounds(position, rotation, dimensions) -> Bounds:
    """Legacy support-origin catalog cuboid, retained as a diagnostic."""
    dimensions = np.asarray(dimensions, dtype=float)
    center = np.asarray(position) + np.asarray(rotation) @ np.array([0., 0., dimensions[2] / 2])
    half = np.abs(rotation) @ (dimensions / 2)
    return Bounds(center - half, center + half, "rotated_catalog_bbox")


def containment_checks(part: Bounds, interior: Bounds, *, tolerance_m=.003,
                       floor_tolerance_m=.006) -> dict[str, bool]:
    return {
        "inside_xy": bool(np.all(part.lower[:2] >= interior.lower[:2] - tolerance_m)
                          and np.all(part.upper[:2] <= interior.upper[:2] + tolerance_m)),
        "below_rim": bool(part.upper[2] <= interior.upper[2] + tolerance_m),
        "above_floor": bool(part.lower[2] >= interior.lower[2] - floor_tolerance_m),
    }


class CollisionGeometryBounds:
    """Cache compiled collision hulls; evaluate only their current world pose.

Supported primitives have exact support functions. Missing hull graphs use
all mesh vertices, which conservatively enclose every convex subset. Other
geoms use their bounding sphere; unbounded/unknown shapes fail containment.
Only lightweight test doubles without geometry arrays use the prior catalog
enclosure, and this compatibility fallback is explicitly reported.
"""
    def __init__(self, model):
        self.model = model
        self._mesh_vertices: dict[int, tuple[np.ndarray, str, tuple[str, ...]]] = {}

    def _mesh_hull(self, mesh_id):
        if mesh_id in self._mesh_vertices:
            return self._mesh_vertices[mesh_id]
        begin = int(self.model.mesh_vertadr[mesh_id])
        count = int(self.model.mesh_vertnum[mesh_id])
        vertices = np.asarray(self.model.mesh_vert[begin:begin + count], dtype=float)
        if count == 0 or vertices.shape != (count, 3) or not np.isfinite(vertices).all():
            raise ValueError("missing_or_nonfinite_mesh_vertices")
        source, reasons = "mujoco_collision_convex_hull", ()
        graph_adr = getattr(self.model, "mesh_graphadr", None)
        graph = getattr(self.model, "mesh_graph", None)
        address = -1 if graph_adr is None else int(graph_adr[mesh_id])
        if graph is not None and address >= 0:
            # numvert, numface, vert_edgeadr[numvert], vert_globalid[numvert].
            hull_count = int(graph[address])
            indices = np.asarray(graph[address + 2 + hull_count:
                                       address + 2 + 2 * hull_count], dtype=int)
            if (hull_count < 4 or len(indices) != hull_count
                    or np.any(indices < 0) or np.any(indices >= count)):
                source = "all_mesh_vertices_conservative_enclosure"
                reasons = ("invalid_compiled_convex_graph",)
            else:
                vertices = vertices[indices]
        else:
            source = "all_mesh_vertices_conservative_enclosure"
            reasons = ("missing_compiled_convex_graph",)
        vertices = vertices.copy()
        vertices.setflags(write=False)
        result = (vertices, source, reasons)
        self._mesh_vertices[mesh_id] = result
        return result

    def geom_bounds(self, data, geom_id: int) -> Bounds:
        import mujoco
        geom_id = int(geom_id)
        center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        kind = int(self.model.geom_type[geom_id])
        size = np.asarray(self.model.geom_size[geom_id], dtype=float)
        types = mujoco.mjtGeom
        if kind == int(types.mjGEOM_MESH):
            mesh_id = int(self.model.geom_dataid[geom_id])
            if mesh_id < 0:
                raise ValueError("missing_mesh_asset")
            vertices, source, reasons = self._mesh_hull(mesh_id)
            world = vertices @ rotation.T + center
            return Bounds(world.min(axis=0), world.max(axis=0), source, reasons)
        if kind == int(types.mjGEOM_BOX):
            half = np.abs(rotation) @ size
            source = "box_support_function"
        elif kind == int(types.mjGEOM_SPHERE):
            half = np.full(3, size[0])
            source = "sphere_support_function"
        elif kind == int(types.mjGEOM_ELLIPSOID):
            half = np.sqrt((rotation ** 2) @ (size ** 2))
            source = "ellipsoid_support_function"
        elif kind == int(types.mjGEOM_CAPSULE):
            half = np.abs(rotation[:, 2]) * size[1] + size[0]
            source = "capsule_support_function"
        elif kind == int(types.mjGEOM_CYLINDER):
            half = np.sqrt(rotation[:, 0] ** 2 + rotation[:, 1] ** 2) * size[0]
            half += np.abs(rotation[:, 2]) * size[1]
            source = "cylinder_support_function"
        else:
            radius = float(self.model.geom_rbound[geom_id])
            if not np.isfinite(radius) or radius <= 0 or kind == int(types.mjGEOM_PLANE):
                raise ValueError(f"unbounded_or_unsupported_geom_type_{kind}")
            return Bounds(center - radius, center + radius,
                          "geom_bounding_sphere_conservative_enclosure",
                          (f"unsupported_geom_type_{kind}",))
        if not np.isfinite(np.r_[center, half]).all() or np.any(half < 0):
            raise ValueError("invalid_primitive_bounds")
        return Bounds(center - half, center + half, source)

    def object_bounds(self, data, geom_ids, legacy: Bounds) -> Bounds:
        if not hasattr(self.model, "geom_type") or not hasattr(data, "geom_xpos"):
            return Bounds(legacy.lower, legacy.upper, "catalog_bbox_compatibility_fallback",
                          ("missing_collision_geometry_arrays",))
        bounds, reasons, sources = [], [], set()
        if not geom_ids:
            reasons.append("missing_runtime_geom_ids")
        for geom_id in geom_ids:
            try:
                bound = self.geom_bounds(data, geom_id)
                bounds.append(bound)
                sources.add(bound.source)
                reasons.extend(f"geom_{geom_id}:{reason}" for reason in bound.fallback_reasons)
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                reasons.append(f"geom_{geom_id}:{error}")
                # A known mesh/primitive may still have a finite conservative
                # sphere when its vertex/shape payload is unavailable.
                try:
                    import mujoco
                    radius = float(self.model.geom_rbound[geom_id])
                    center = np.asarray(data.geom_xpos[geom_id], dtype=float)
                    if (int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                            or radius <= 0 or not np.isfinite(np.r_[radius, center]).all()):
                        raise ValueError("invalid_bounding_sphere")
                    bounds.append(Bounds(center - radius, center + radius,
                                         "geom_bounding_sphere_conservative_enclosure"))
                    sources.add("geom_bounding_sphere_conservative_enclosure")
                except (AttributeError, IndexError, TypeError, ValueError):
                    return Bounds(np.full(3, -np.inf), np.full(3, np.inf),
                                  "unavailable_geometry_reject_containment", tuple(reasons))
        if not bounds:
            return Bounds(np.full(3, -np.inf), np.full(3, np.inf),
                          "unavailable_geometry_reject_containment", tuple(reasons))
        return Bounds(np.min([value.lower for value in bounds], axis=0),
                      np.max([value.upper for value in bounds], axis=0),
                      "+".join(sorted(sources)), tuple(reasons))

    def bin_interior(self, data, box, geom_ids) -> Bounds:
        names = {self.model.geom(int(index)).name: int(index) for index in geom_ids}
        prefix = f"bin_{int(box['id'])}_"
        floor = names.get(prefix + "bottom", names.get(prefix + "floor"))
        wall_ids = [names.get(prefix + name) for name in ("left", "right", "front", "back")]
        reasons = ()
        if floor is not None and all(index is not None for index in wall_ids):
            try:
                import mujoco
                ids = [floor] + wall_ids
                for index in ids:
                    rotation = np.asarray(data.geom_xmat[index]).reshape(3, 3)
                    if (int(self.model.geom_type[index]) != int(mujoco.mjtGeom.mjGEOM_BOX)
                            or not np.allclose(np.abs(rotation), np.eye(3), atol=1e-8)):
                        raise ValueError("bin_walls_are_not_world_axis_aligned_boxes")
                base, left, right, front, back = [self.geom_bounds(data, index) for index in ids]
                lower = np.array([max(left.upper[0], base.lower[0]),
                                  max(front.upper[1], base.lower[1]), base.upper[2]])
                upper = np.array([min(right.lower[0], base.upper[0]),
                                  min(back.lower[1], base.upper[1]),
                                  min(wall.upper[2] for wall in (left, right, front, back))])
                if np.any(lower >= upper):
                    raise ValueError("invalid_physical_bin_interior")
                return Bounds(lower, upper, "collision_walls_and_bottom_inner_faces")
            except (AttributeError, IndexError, TypeError, ValueError) as error:
                reasons = (str(error),)
        else:
            reasons = ("missing_bottom_or_four_collision_walls",)
        if hasattr(self.model, "geom_type") and hasattr(data, "geom_xpos"):
            # Without verified wall inner faces, metadata cannot prove an
            # enclosure. Reject scoring instead of accepting an assumed box.
            return Bounds(np.full(3, np.inf), np.full(3, -np.inf),
                          "unavailable_bin_geometry_reject_containment", reasons)
        # Existing axis-aligned scene metadata is narrower than its walls.
        # Keep this explicitly labelled legacy enclosure for incomplete mocks;
        # do not infer a larger box from a debug site or wall bounding envelope.
        center, size = np.asarray(box["center"]), np.asarray(box["size"])
        return Bounds(center - size / 2, center + size / 2,
                      "configured_bin_interior_fallback", reasons)
