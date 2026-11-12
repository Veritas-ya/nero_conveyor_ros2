"""Isolate unobservable RGB candidates without changing native RGB-D or limits.

The boundary-only helper retains its original contract. The candidate helper
also rejects interior components locally, so an independent valid workpiece
can reach the unchanged detector/admission/tracker. Rejected interior blobs
are unknown, not measured background; the caller must not publish an empty
frame when any such uncertain component remains.
"""
import cv2
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.detection import color_masks
from vendor.nero7_speed_reference.nero7_perception.geometry import roi_mask


def isolate_boundary_depth_gaps(color, depth, params, *, camera=None,
                                color_to_base=None, surface=None):
    return _isolate_depth_gaps(color, depth, params, camera=camera,
        color_to_base=color_to_base, surface=surface, include_interior=False)


def isolate_candidate_depth_gaps(color, depth, params, *, camera=None,
                                 color_to_base=None, surface=None):
    return _isolate_depth_gaps(color, depth, params, camera=camera,
        color_to_base=color_to_base, surface=surface, include_interior=True)


def _isolate_depth_gaps(color, depth, params, *, camera, color_to_base,
                        surface, include_interior):
    _, gray = color_masks(color, params)
    mask = gray & roi_mask(gray.shape, params['roi_px'])
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    output = color.copy()
    valid = np.isfinite(depth) & (depth > 0)
    skipped = []
    u0, v0, u1, v1 = params['roi_px']
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < params['min_area_px']:
            continue
        x, y, w, h = map(int, stats[label, :4])
        boundary = x <= u0+2 or y <= v0+2 or x+w >= u1-2 or y+h >= v1-2
        component = (labels[y:y+h, x:x+w] == label)
        region = component.astype(np.uint8)
        border = params['depth_erode_px']
        if border:
            region = cv2.erode(region, np.ones((2*border+1, 2*border+1), np.uint8),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
        region = region.astype(bool)
        inside = int(region.sum())
        usable = int((region & valid[y:y+h, x:x+w]).sum())
        if usable >= params['depth_min_valid_pixels'] and usable >= params['depth_min_valid_ratio']*inside:
            continue
        reason = 'roi_boundary' if boundary else 'insufficient_interior_depth'
        maximum_distance = None
        if (camera is not None and color_to_base is not None and surface is not None
                and usable >= params['depth_min_valid_pixels']):
            # Missing depth alone never proves that an interior blob is
            # background. Require enough real samples, all at least 5 mm
            # outside the measured belt footprint. A single plausible belt
            # sample retains the original upstream ambiguity/depth checks.
            rr, cc = np.nonzero(region & valid[y:y+h, x:x+w])
            optical = camera.rays(cc+x, rr+y)*depth[rr+y, cc+x, None]
            transform = np.asarray(color_to_base)
            base = optical@transform[:3, :3].T+transform[:3, 3]
            plane = np.asarray(surface.plane_to_base)
            local = (base-plane[:3, 3])@plane[:3, :3]
            if np.isfinite(local).all():
                distances = [cv2.pointPolygonTest(surface.hull_xy,
                    tuple(map(float, point)), True) for point in local[:, :2]]
                maximum_distance = float(max(distances))
                if maximum_distance < -.005:
                    reason = 'measured_outside_belt'
        if not include_interior and not boundary and reason!='measured_outside_belt':
            continue
        output[y:y+h, x:x+w][component] = 0
        skipped.append({'bbox_px':[x,y,w,h], 'valid_pixels':usable,
            'interior_pixels':inside, 'reason':reason,
            'maximum_belt_signed_distance_m':maximum_distance})
    return output, skipped
