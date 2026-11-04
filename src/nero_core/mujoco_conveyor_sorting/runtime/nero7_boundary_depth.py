"""Isolate clipped RGB ROI-edge blobs with insufficient registered depth.

Keep the upstream detector and its depth thresholds unchanged. Only components
cut by the ROI edge and failing the upstream depth test are removed; callers
must keep the frame ambiguous when no other accepted target remains.
"""
import cv2
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.detection import color_masks
from vendor.nero7_speed_reference.nero7_perception.geometry import roi_mask


def isolate_boundary_depth_gaps(color, depth, params):
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
        if not boundary:
            continue
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
        output[y:y+h, x:x+w][component] = 0
        skipped.append({'bbox_px':[x,y,w,h], 'valid_pixels':usable, 'interior_pixels':inside})
    return output, skipped
