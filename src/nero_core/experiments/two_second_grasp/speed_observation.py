"""Experimental simulation-only visual point-cloud position for speed."""
import numpy as np
from experiments.graspgenx_sim.camera import cloud_for_track


def percentile_cloud_center(frame, track_id):
    """Center of the visible cloud's 5th-to-95th percentile XY extent.

    This is an observed visible-cloud position, not the physical object center.
    It is used for speed fitting only, never as an object pose or grasp point.
    """
    points = cloud_for_track(frame, track_id)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return None
    low, high = np.percentile(points, [5., 95.], axis=0)
    return (low+high)/2.
