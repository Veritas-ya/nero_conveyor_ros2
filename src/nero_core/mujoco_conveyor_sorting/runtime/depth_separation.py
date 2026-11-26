"""Native point selection for OBB fitting, including isolated seed noise.

Measured points only; coordinates are not changed or completed. Broad real
flanges within belt-plane uncertainty may be removed; opt in explicitly.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ALGORITHM_VERSION = 'native_depth_separation_v2'

# Conservative OBB-only limits. Count ALL supported points outside the
# largest metric component before making any decision about a small island.
MAX_ISLAND_POINTS = 8
MAX_PERIPHERAL_FRACTION = .01
MIN_ISLAND_GAP_M = .020
ISLAND_GAP_IN_LINK_RADII = 4


def _isolated_seed_noise(tree, groups, supported, gap):
    """Use the original metric graph; a connected thin feature stays intact."""
    noise = np.zeros(len(groups), dtype=bool)
    sizes = np.bincount(groups[supported])
    main = int(np.argmax(sizes))
    peripheral = supported & (groups != main)
    count = np.count_nonzero(peripheral)
    if not count or count > MAX_PERIPHERAL_FRACTION * sizes.sum():
        return noise
    for group in np.unique(groups[peripheral]):
        if sizes[group] > MAX_ISLAND_POINTS:
            continue
        island = groups == group
        # Distance is between nearest actual samples, not cluster centres.
        # Also protect a tiny fragment next to another supported component.
        neighbours = tree.query_ball_point(tree.data[island], gap + 1e-12)
        nearby = np.concatenate(neighbours).astype(np.intp, copy=False)
        if not np.any(supported[nearby] & ~island[nearby]):
            noise[island] = True
    return noise


def separate(points, pixels, seed, *, spacing, plane_rms, plane_band):
    p = np.asarray(points)
    band = max(plane_band, 3*plane_rms)
    raised = p[:, 2] > band
    ij = pixels-np.min(pixels, axis=0)
    mask = np.zeros(tuple(ij.max(axis=0)+1), np.uint8)
    mask[tuple(ij[raised].T)] = 1
    _, labels = cv2.connectedComponents(mask, connectivity=8)
    ids = labels[tuple(ij.T)]
    seeded_ids = np.unique(ids[np.asarray(seed, bool) & raised])
    seeded_ids = seeded_ids[seeded_ids != 0]
    body = raised & np.isin(ids, seeded_ids)
    if np.count_nonzero(body) < 21:
        raise ValueError('insufficient raised native target support')
    support_radius = max(.003, 2*spacing)
    low_distance = cKDTree(p[body, :2]).query(p[:, :2])[0]
    # Numerical tolerance only (one picometre): translating the origin must
    # not flip a point exactly on the metric boundary due to roundoff.
    epsilon = 1e-12
    low = ~raised & (low_distance <= support_radius+epsilon)
    keep = body | low
    retained = np.flatnonzero(keep)
    radius = max(.004, 3.5*spacing)
    tree = cKDTree(p[retained])
    pairs = tree.query_pairs(radius+epsilon, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(retained), len(retained)))
    _, groups = connected_components(graph, directed=False)
    supported_groups = np.unique(groups[np.asarray(seed, bool)[retained]])
    unsupported = ~np.isin(groups, supported_groups)
    gap = max(MIN_ISLAND_GAP_M, ISLAND_GAP_IN_LINK_RADII * radius)
    islands = _isolated_seed_noise(tree, groups, ~unsupported, gap)
    keep[retained[unsupported | islands]] = False
    return keep, dict(algorithm_version=ALGORITHM_VERSION,
        plane_band_m=band, low_support_radius_m=support_radius,
        raised_seed_components=len(seeded_ids), retained_raised=int((keep & raised).sum()),
        retained_low=int((keep & ~raised).sum()), removed=int((~keep).sum()),
        original_seed_count=int(np.sum(seed)), retained_seed_count=int(np.sum(keep & seed)),
        disconnected_completion_removed=int(unsupported.sum()), metric_link_radius_m=radius,
        isolated_seed_island_points_removed=int(islands.sum()),
        isolated_seed_island_count=int(np.unique(groups[islands]).size),
        minimum_island_gap_m=gap, maximum_island_points=MAX_ISLAND_POINTS,
        maximum_peripheral_fraction=MAX_PERIPHERAL_FRACTION,
        coordinates_modified=False, hidden_points_added=0,
        limitation='A genuine broad flange inside the belt uncertainty band can be removed; not universally validated.')
