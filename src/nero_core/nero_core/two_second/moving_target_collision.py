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
    for geom in model.meshes:
        if contact and geom in model.fingers:
            continue
        center=model.data.geom_xpos[geom]
        ids=tree.query_ball_point(center-delta,model.collision_radii[geom]+1e-9)
        if ids:
            inside=model.inside_geom(points[ids]+delta,geom,margin_m)
            if inside.any():
                return geom,np.asarray(ids,dtype=np.int64)[inside]
    return None
