"""User-selected temporary empty-region assumption at the captured gripper pose."""
import hashlib
import json

import numpy as np
from scipy.spatial import HalfspaceIntersection, cKDTree

PADDING_M = .05
GRIPPER_LINKS = frozenset({'gripper_flange', 'gripper_base', 'gripper_link1', 'gripper_link2'})


def ignore_captured_gripper_scene(points, bodies, state, *, enabled=False):
    """Filter environment only, once per captured scene, never at planned poses.

    This is an explicit experimental assumption, not a depth-validity test.
    The caller retains the separate target cloud and the original robot model.
    """
    points = np.asarray(points, dtype=float)
    removed = np.zeros(len(points), dtype=bool)
    selected = []
    if enabled:
        selected = [body for body in bodies if body.name in GRIPPER_LINKS]
        tree = cKDTree(points)
        for body in selected:
            equations = body.equations.copy()
            equations[:, 3] -= PADDING_M
            # Bound the offset hull itself: offset corners can extend farther
            # than nominal radius + padding. Original self-filter limits stay
            # independent of this explicitly chosen experimental region.
            vertices = HalfspaceIntersection(equations, body.vertices.mean(axis=0)).intersections
            radius = float(np.linalg.norm(vertices, axis=1).max())
            indices = np.asarray(tree.query_ball_point(body.position, radius+1e-9), dtype=int)
            for start in range(0, len(indices), 512):
                ids = indices[start:start+512]
                local = (points[ids]-body.position) @ body.rotation
                inside = np.all(local @ equations[:, :3].T+equations[:, 3] <= 1e-10, axis=1)
                removed[ids[inside]] = True
    reference = {'joints_rad': np.asarray(state['joints'], dtype=float).tolist(),
                 'width_m': float(state['width_m'])}
    statistics = {
        'enabled': bool(enabled), 'reason': 'user_assumed_empty_region' if enabled else 'disabled',
        'padding_m': PADDING_M, 'links': sorted(body.name for body in selected),
        'reference_state': reference, 'reference_frame': 'robot_base_at_paired_capture',
        'region_follows_commanded_robot': False,
        'region_definition': 'union of captured gripper convex hull halfspaces offset by padding_m',
        'input_points': len(points), 'removed_points': int(removed.sum()),
        'retained_points': int((~removed).sum()),
        'input_sha256': hashlib.sha256(points.tobytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        'depth_error_proven': False, 'target_cloud_modified': False,
    }
    return points[~removed], removed, statistics
