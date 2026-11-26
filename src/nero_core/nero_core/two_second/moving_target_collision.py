"""Shared sampled moving-point-cloud collision predicate, in robot base coordinates."""
import numpy as np


def first_moving_target_hit(model, points, tree, delta, *, contact, margin_m=.004):
    """Return the first hit in original geometry order; never alter the cloud.

    The immutable tree indexes exposure-time points. During/after contact the
    production policy allows the finger geometries, but still checks every
    other robot geometry. The caller may supply a workpiece margin; the legacy default is 4 mm.
    """
    points=np.asarray(points)
    delta=np.asarray(delta)
    groups = getattr(model, 'moving_target_query_groups', None)
    if groups is None:
        # Minimal/simulation models may supply geometry dynamically.
        geometries = [g for g in model.meshes if not (contact and g in model.fingers)]
        radii = [model.collision_radii[g] + 1e-9 for g in geometries]
    else:
        geometries, radii = groups[bool(contact)]
    if not len(geometries):
        return None
    centers = model.data.geom_xpos[geometries] - delta
    neighbors = tree.query_ball_point(centers, radii, return_sorted=False)
    for geom, ids in zip(geometries, neighbors):
        if ids:
            inside = model.inside_geom(points[ids] + delta, geom, margin_m)
            if inside.any():
                return int(geom), np.asarray(ids, dtype=np.int64)[inside]
    return None
