"""Tracked exposure admission follows the measured detector footprint."""
import numpy as np

from .reference import EntryNotReady


def detector_freeze_window(config, surface=None, *, belt_region=None):
    """Project the measured detection prism, clipped to detector base Y.

    This only bounds which already-qualified observations can freeze. It
    neither changes the arm workspace nor certifies a future grasp path.
    """
    if belt_region is not None:
        # The private planner uses the frozen exposure's geometry, never a
        # newer camera surface or the active execution model's mutable scene.
        transform = np.asarray(belt_region['plane_to_base'], float)
        if 'hull_xy_m' in belt_region:
            hull = np.asarray(belt_region['hull_xy_m'], float).reshape(-1, 2)
            source = 'frozen_measured_hull'
        else:
            # Older archives retained native quantile bounds but not the hull.
            # This is an admission envelope only, not a collision certificate.
            bounds = np.asarray(belt_region['bounds_xy_m'], float)
            x0, x1 = bounds[0]
            y0, y1 = bounds[1]
            hull = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            source = 'legacy_frozen_native_bounds'
    elif surface is not None:
        hull = np.asarray(surface.hull_xy, float).reshape(-1, 2)
        transform = np.asarray(surface.plane_to_base, float)
        source = 'live_measured_hull'
    else:
        raise EntryNotReady('measured detection footprint is not ready', 'detection_footprint_missing')
    y_bounds = config.raw['perception_detector_workspace_y_m']
    coordinates = []
    bottom = np.column_stack((hull, np.zeros(len(hull)))) @ transform[:3,:3].T + transform[:3,3]
    top = bottom + config.raw['maximum_target_height_m']*transform[:3,2]
    # Clipping a tilted prism also creates vertices on its vertical edges.
    for a,b in zip(bottom,top):
        for limit in y_bounds:
            if (a[1]-limit)*(b[1]-limit)<0:
                point=a+(b-a)*(limit-a[1])/(b[1]-a[1])
                coordinates.append(float(point@config.direction))
    for height in (0., config.raw['maximum_target_height_m']):
        polygon = np.column_stack((hull, np.full(len(hull), height)))
        polygon = polygon @ transform[:3, :3].T + transform[:3, 3]
        for limit, greater in zip(y_bounds, (True, False)):
            clipped = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                da, db = a[1]-limit, b[1]-limit
                ina, inb = (da >= 0, db >= 0) if greater else (da <= 0, db <= 0)
                if ina:
                    clipped.append(a)
                if ina != inb:
                    clipped.append(a+(b-a)*da/(da-db))
            polygon = np.asarray(clipped).reshape(-1, 3)
        coordinates.extend(polygon @ config.direction)
    if not coordinates:
        raise EntryNotReady('detector range does not intersect measured belt', 'detection_footprint_empty')
    upstream = float(min(coordinates))
    distance = float(config.follow_bounds[0]-upstream)
    if distance <= 0:
        raise EntryNotReady('no upstream detector area', 'detection_footprint_empty')
    limit = config.raw.get('tracked_freeze_upstream_limit_m')
    allowed = distance if limit is None else min(distance, limit)
    return dict(distance_m=allowed, required_m=distance, capped=allowed < distance,
        policy='measured_detector_upstream', upstream_coordinate_m=float(config.follow_bounds[0]-allowed),
        detector_upstream_coordinate_m=upstream,
        detector_workspace_y_m=list(y_bounds), footprint_source=source)
