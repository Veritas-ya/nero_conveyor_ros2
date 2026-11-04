"""Visible-point oriented box, shared by simulation and measured hardware input."""
import numpy as np


class VisibleOBBError(ValueError):
    """This exposure's visible points cannot define the requested OBB."""


def visible_belt_aligned_obb(points_world):
    """Minimum-area XY rectangle of visible points, extruded along world Z.

    The caller supplies belt coordinates. This is an OBB of the observed
    surface, not an estimate of the hidden back side of an object.
    """
    from scipy.spatial import ConvexHull, QhullError
    points = np.asarray(points_world, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 21 or not np.isfinite(points).all():
        raise VisibleOBBError('need at least 21 finite visible world points')
    xy = points[:, :2]
    try:
        hull = xy[ConvexHull(xy).vertices]
    except QhullError as exc:
        raise VisibleOBBError('visible XY points cannot define an oriented box') from exc
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    edges = edges[lengths > 1e-12]
    if not len(edges):
        raise VisibleOBBError('visible XY points cannot define an oriented box')
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best = None
    for angle in angles:
        co, si = np.cos(angle), np.sin(angle)
        rotation = np.array([[co, -si], [si, co]])
        local = xy @ rotation
        lower, upper = local.min(axis=0), local.max(axis=0)
        area = np.prod(upper - lower)
        if best is None or area < best[0]:
            best = (area, rotation, lower, upper)
    _, rotation, lower, upper = best
    center_xy = rotation @ ((lower + upper) / 2)
    z_lower, z_upper = points[:, 2].min(), points[:, 2].max()
    center = np.r_[center_xy, (z_lower + z_upper) / 2]
    basis = np.eye(3); basis[:2, :2] = rotation
    return {'center_world_m': center.tolist(), 'axes_world_columns': basis.tolist(),
            'extent_m': [*(upper - lower).tolist(), float(z_upper-z_lower)],
            'visible_point_count': len(points)}
