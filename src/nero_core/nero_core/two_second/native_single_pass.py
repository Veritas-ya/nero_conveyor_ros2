"""Extract each triggered native region once and share its geometry downstream."""
from copy import copy
from dataclasses import dataclass, replace

import cv2
import numpy as np

from mujoco_conveyor_sorting.runtime.belt_surface import _geometry_fingerprint
from mujoco_conveyor_sorting.runtime.frame_products import frame_products, green_rgb_mask
from mujoco_conveyor_sorting.runtime.local_target_geometry import NATIVE_CONTEXT_PIXELS
from mujoco_conveyor_sorting.runtime.single_pick import detect_target
from .belt_collision_scene import BELT_SUPPORT_BAND_M
from .fixed_freeze import DepthCandidate, NativeDepthLocator
from .measured_rgbd import fit_plane, green_belt_points


_SCENE_ROUNDOFF_M = 1e-8
_SCENE_HALF_HEIGHT_M = BELT_SUPPORT_BAND_M + _SCENE_ROUNDOFF_M


@dataclass(frozen=True)
class PreparedComponent:
    frame: object
    surface: object
    state: object
    model: object
    config: object
    d2c: np.ndarray
    geometry: dict
    scene: object


def _base_points(frame, rows, cols, depth):
    k = frame.depth_intrinsics
    z = np.asarray(depth, dtype=np.float64)
    camera = np.stack(((cols - k.cx) * z / k.fx, (rows - k.cy) * z / k.fy, z), axis=-1)
    transform = frame.camera_to_world_cv
    return camera @ transform[:3, :3].T + transform[:3, 3]


def _belt_scene_window(frame, plane, bounds):
    """Conservative image window around the measured belt's 5 mm slab."""
    # The actual membership test below still uses native depth. Project all
    # eight corners, including the rounding allowance used by that test.
    corners = np.array([[x, y, h]
                        for x in (bounds[0, 0]-_SCENE_ROUNDOFF_M, bounds[0, 1]+_SCENE_ROUNDOFF_M)
                        for y in (bounds[1, 0]-_SCENE_ROUNDOFF_M, bounds[1, 1]+_SCENE_ROUNDOFF_M)
                        for h in (-_SCENE_HALF_HEIGHT_M, _SCENE_HALF_HEIGHT_M)])
    camera = frame.camera_to_world_cv
    try:
        base = np.linalg.solve(plane[:3, :3].T, corners.T).T + plane[:3, 3]
        optical = np.linalg.solve(camera[:3, :3], (base-camera[:3, 3]).T).T
    except np.linalg.LinAlgError:
        return np.s_[:, :]
    if not np.isfinite(optical).all() or np.any(optical[:, 2] <= 1e-6):
        return np.s_[:, :]
    k = frame.depth_intrinsics
    uv = optical[:, :2]/optical[:, 2:] * [k.fx, k.fy] + [k.cx, k.cy]
    if not np.isfinite(uv).all():
        return np.s_[:, :]
    # With positive optical depth the projection of the box lies in the
    # convex hull of these projections. Round outwards by one more pixel.
    size = [k.width, k.height]
    x0, y0 = np.clip(np.floor(uv.min(axis=0))-1, 0, size).astype(int)
    x1, y1 = np.clip(np.ceil(uv.max(axis=0))+2, 0, size).astype(int)
    return np.s_[y0:y1, x0:x1]


def _depth_window(frame, d2c, bounds):
    """A read-only view of the native depth, with the original full colour image."""
    y0, y1, x0, x1 = bounds
    k = frame.depth_intrinsics
    depth = frame.depth_m[y0:y1, x0:x1]
    # The SDK exposure has already validated and frozen its arrays. Keep RGB,
    # IR, timestamps and calibration by reference, with crop-local intrinsics.
    crop = copy(frame)
    object.__setattr__(crop, 'depth_m', depth)
    object.__setattr__(crop, 'depth_intrinsics', replace(
        k, width=x1-x0, height=y1-y0, cx=k.cx-x0, cy=k.cy-y0))
    object.__setattr__(crop, '_depth_pixel_origin', (y0, x0))
    object.__setattr__(crop, '_uncropped_depth_intrinsics', k)
    cache = frame_products(crop, d2c)
    cache.get('green_rgb_mask', lambda: green_rgb_mask(frame, d2c))
    rows, cols = np.ogrid[y0:y1, x0:x1]
    cache.get('native_deprojection', lambda: _base_points(frame, rows, cols, depth))
    return crop


class SinglePassLocator:
    def __init__(self, config, surface, frame, d2c, *, capture_band_m):
        layout = NativeDepthLocator(config, surface, frame, capture_band_m=capture_band_m)
        self.config = config
        self.surface = surface
        self.search = layout.surface
        self.d2c = np.array(d2c, copy=True)
        self.d2c.setflags(write=False)
        self.crop = layout.crop
        self.maximum_height = config.raw['maximum_target_height_m']
        plane = surface.plane_to_base
        k = frame.depth_intrinsics
        normal = frame.camera_to_world_cv[:3, :3].T @ plane[:3, 2]
        self.height_rate = ((np.arange(k.width) - k.cx) * normal[0] / k.fx
                            + (np.arange(k.height)[:, None] - k.cy) * normal[1] / k.fy
                            + normal[2])
        self.origin_height = float((layout.origin - plane[:3, 3]) @ plane[:3, 2])

        # Intersect each calibrated ray with the search prism and capture band.
        # Only depth intervals are cached here. Each exposure supplies its depth.
        rays, origin = layout.rays_base, layout.origin
        near = np.zeros(rays.shape[:2])
        far = np.full_like(near, np.inf)

        def clip(axis, minimum):
            rate = rays @ axis
            offset = minimum - float(origin @ axis)
            parallel = np.abs(rate) <= 1e-14
            t = np.divide(offset, rate, out=np.zeros_like(rate), where=~parallel)
            np.maximum(near, t, out=near, where=rate > 1e-14)
            np.minimum(far, t, out=far, where=rate < -1e-14)
            if offset > 0:
                near[parallel] = np.inf

        clip(config.direction, config.follow_bounds[0] - capture_band_m[1] - .001)
        clip(-config.direction, -(config.follow_bounds[0] - capture_band_m[0]) - .001)
        hull = self.search.hull_xy.reshape(-1, 2).astype(float)
        center = hull.mean(axis=0)
        for a, b in zip(hull, np.roll(hull, -1, axis=0)):
            side = np.array([a[1]-b[1], b[0]-a[0]])
            length = np.linalg.norm(side)
            if length == 0:
                continue
            side /= length
            if (center-a) @ side < 0:
                side = -side
            axis = plane[:3, :2] @ side
            clip(axis, plane[:3, 3] @ axis + a @ side - .001)
        self.seed_near, self.seed_far = near, far

    def _foreground(self, depth, rate):
        height = depth * rate + self.origin_height
        # Superset of the full detector's exact metric height predicate.
        return (depth > 0) & (height >= .012-1e-9) & (height <= self.maximum_height+1e-9)

    def _regions(self, frame):
        y0, y1, x0, x1 = self.crop
        depth = frame.depth_m[y0:y1, x0:x1]
        fg = self._foreground(depth, self.height_rate[y0:y1, x0:x1])
        trigger = fg & (depth >= self.seed_near) & (depth <= self.seed_far)
        if not trigger.any():
            return [], False
        _, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
        chosen = np.unique(labels[trigger])
        height, width = fg.shape

        def touches_crop(stat):
            x, y, w, h = stat[:4]
            return x == 0 or y == 0 or x+w == width or y+h == height

        expanded = any(touches_crop(stats[i]) for i in chosen)
        if expanded:
            # Expand coarse membership before any 3-D extraction. Even one
            # boundary pixel can be part of a large object outside the window.
            rows, cols = np.nonzero(trigger)
            fg = self._foreground(frame.depth_m, self.height_rate)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
            chosen = np.unique(labels[rows+y0, cols+x0])
            y0 = x0 = 0
        regions = []
        for i in chosen:
            x, y, w, h, area = stats[i]
            if i and area >= 70:
                regions.append((int(y+y0), int(x+x0), labels[y:y+h, x:x+w] == i))
        return regions, expanded

    def _scene(self, frame):
        z = frame.depth_m
        plane = self.surface.plane_to_base
        bounds = self.surface.scene_bounds_xy_m
        if bounds is None:
            bounds = self.surface.bounds_xy_m
        window = _belt_scene_window(frame, plane, bounds)
        depth = z[window]
        height = depth * self.height_rate[window] + self.origin_height
        rows, cols = np.nonzero((depth > 0) & (np.abs(height) <= _SCENE_HALF_HEIGHT_M+1e-9))
        rows += window[0].start or 0
        cols += window[1].start or 0
        points = _base_points(frame, rows, cols, z[rows, cols])
        local = (points - plane[:3, 3]) @ plane[:3, :3]
        x, y, h = local.T
        keep = ((np.abs(h) <= _SCENE_HALF_HEIGHT_M)
                & (x >= bounds[0, 0]-_SCENE_ROUNDOFF_M) & (x <= bounds[0, 1]+_SCENE_ROUNDOFF_M)
                & (y >= bounds[1, 0]-_SCENE_ROUNDOFF_M) & (y <= bounds[1, 1]+_SCENE_ROUNDOFF_M))
        scene = points[keep]
        ids = rows[keep] * z.shape[1] + cols[keep]
        scene.setflags(write=False)
        ids.setflags(write=False)
        return scene, ids, int(np.count_nonzero(z > 0))

    def _extract(self, frame, y, x, membership, robot_model, robot_state):
        # Support radius depends on median depth in the original measurement
        # context. Only the disjoint raw region can contribute object pixels.
        h, w = membership.shape
        native_h, native_w = frame.depth_m.shape
        halo = NATIVE_CONTEXT_PIXELS
        y0, x0 = max(0, y-halo), max(0, x-halo)
        bounds = y0, min(native_h, y+h+halo), x0, min(native_w, x+w+halo)
        crop = _depth_window(frame, self.d2c, bounds)
        region = np.zeros(crop.depth_m.shape, bool)
        region[y-y0:y-y0+h, x-x0:x-x0+w] = membership
        surface = replace(self.search, geometry_fingerprint=_geometry_fingerprint(crop, self.d2c))
        raw = self.config.raw
        detections, rejected = detect_target(
            crop, self.d2c, measured_surface=surface, fit_plane=fit_plane,
            green_belt_points=green_belt_points, robot_model=robot_model, robot_state=robot_state,
            native_region_mask=region, target_shape='simple', scene_scope='measured_belt_surface_only',
            rgb_cloud_completion=False, belt_direction_base=self.config.direction,
            minimum_complete_height_m=raw['native_single_min_height_m'],
            maximum_target_height_m=self.maximum_height,
            maximum_target_length_m=raw.get('maximum_target_length_m', .16),
            belt_edge_margin_m=raw.get('perception_belt_edge_margin_m', .025),
            belt_near_camera_margin_m=raw.get('perception_belt_near_camera_margin_m', 0.))
        return detections, rejected, bounds

    def locate(self, frame, *, robot_model=None, robot_state=None):
        if not self.surface.matches_frame(frame, self.d2c):
            raise ValueError('单帧候选与带面标定或相机时钟不一致')
        if (robot_model is None) != (robot_state is None):
            raise ValueError('自身过滤需要同曝光配对的机器人状态')
        regions, expanded = self._regions(frame)
        found, errors = [], []
        cached_scene = None

        def scene():
            nonlocal cached_scene
            if cached_scene is None:
                cached_scene = self._scene(frame)
            return cached_scene

        height, width = frame.depth_m.shape
        for y, x, membership in regions:
            detections, rejected, bounds = self._extract(frame, y, x, membership, robot_model, robot_state)
            errors.extend(rejected)
            for det in detections:
                rows, cols = np.nonzero(det['mask'])
                rows += bounds[0]
                cols += bounds[2]
                if np.any((rows == 0) | (cols == 0) | (rows == height-1) | (cols == width-1)):
                    errors.append({'reason': 'incomplete_at_sensor_edge'})
                    continue
                ids = rows * width + cols
                mask = np.zeros(frame.depth_m.shape, bool)
                mask.flat[ids] = True
                points = det['object_points']
                det.update(mask=mask, object_seed_pixel_indices=ids, local_processing={
                    'policy': 'native_foreground_single_extraction', 'crop_yxyx': list(bounds),
                    'rough_window_expanded': expanded, 'rgb_completion': False})
                for key in ('points', 'points_plane', '_frame_products'):
                    det.pop(key)
                det['excluded_depth_pixels'] = int(det.pop('_rejected_target_depth_mask').sum())
                along = points @ self.config.direction
                prepared = PreparedComponent(frame, self.surface, robot_state, robot_model,
                                             self.config, self.d2c, det, scene)
                found.append(DepthCandidate(np.median(points, axis=0), len(ids), ids,
                                            (float(along.min()), float(along.max())), prepared))
        return found, dict(rough_regions=len(regions), full_extractions=len(regions),
                           expanded=expanded, candidate_count=len(found), component_rejections=errors[:8])
