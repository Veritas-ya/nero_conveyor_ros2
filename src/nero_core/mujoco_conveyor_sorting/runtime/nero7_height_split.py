"""Separate a belt-connected RGB candidate using the same frame's depth.

Only candidates that the original height-ratio check would reject are split.
The returned color copy is used solely to construct the tracking mask. Native
RGB-D, target geometry, collision inputs, and unknown depths are not filled.
"""
import cv2
import numpy as np

from vendor.nero7_speed_reference.nero7_perception.detection import color_masks
from vendor.nero7_speed_reference.nero7_perception.geometry import roi_mask


def split_mixed_belt_candidates(color, depth, height, params):
    """Isolate measured above-belt pixels from merged belt/target components."""
    if depth.shape != height.shape or color.shape[:2] != depth.shape:
        raise ValueError('same-frame color, registered depth and height must share a grid')
    _, gray = color_masks(color, params)
    mask = gray & roi_mask(gray.shape, params['roi_px'])
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    result = color
    splits = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if area < params['min_area_px']:
            continue
        component = labels[y:y+h, x:x+w] == label
        border = int(params['depth_erode_px'])
        interior = component.astype(np.uint8)
        if border:
            interior = cv2.erode(interior, np.ones((2*border+1, 2*border+1), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0)
        interior = interior.astype(bool)
        z = depth[y:y+h, x:x+w]
        hgt = height[y:y+h, x:x+w]
        valid = np.isfinite(z) & (z > 0) & np.isfinite(hgt)
        above = valid & (hgt >= params['block_height_min']) & (hgt <= params['block_height_max'])
        belt = valid & (hgt < params['block_height_min'])
        inside = int(interior.sum())
        above_inside = int((interior & above).sum())
        required = max(params['depth_min_valid_pixels'], params['depth_min_valid_ratio']*inside)
        # No split for clean candidates or components that do not contain
        # enough measured above-belt evidence for the original detector.
        if (above_inside >= required or above_inside < params['depth_min_valid_pixels']
                or not np.any(interior & belt)):
            continue
        candidate = (component & above).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, part_stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        if not any(part_stats[i, cv2.CC_STAT_AREA] >= params['min_area_px'] for i in range(1, n)):
            continue
        if result is color:
            result = color.copy()
        discard = component & ~above
        result[y:y+h, x:x+w][discard] = 0
        splits.append({'bbox_px': [x, y, w, h], 'original_interior_pixels': inside,
            'above_belt_interior_pixels': above_inside,
            'measured_belt_pixels': int((component & belt).sum()),
            'unobserved_depth_pixels': int((component & ~valid).sum())})
    return result, splits
