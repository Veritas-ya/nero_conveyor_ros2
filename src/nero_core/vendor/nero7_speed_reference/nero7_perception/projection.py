"""Pure projection function extracted unchanged from upstream alignment.py."""
import numpy as np
from .geometry import check_transform


def project_cloud(points, transform, camera):
    """Nearest optical Z per rounded color pixel, including distortion."""
    transform = check_transform(transform)
    points = np.asarray(points, float).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    points = points @ transform[:3, :3].T + transform[:3, 3]
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > .05)]
    depth = np.full((camera.height, camera.width), np.inf, np.float32)
    if len(points):
        projection = camera.project(points)
        good = (np.isfinite(projection).all(axis=1) & (projection[:, 0] >= -.5)
                & (projection[:, 0] <= camera.width-.5) & (projection[:, 1] >= -.5)
                & (projection[:, 1] <= camera.height-.5))
        points, projection = points[good], projection[good]
        pixels = np.rint(projection).astype(np.int64)
        u, v = pixels.T
        inside = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
        np.minimum.at(depth.ravel(), v[inside] * camera.width + u[inside], points[inside, 2])
    depth[~np.isfinite(depth)] = np.nan
    return depth
