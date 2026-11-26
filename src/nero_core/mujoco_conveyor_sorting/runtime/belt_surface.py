"""Measured belt geometry used to select the local Nero7 camera ROI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np


class BeltSurfaceNotReady(ValueError):
    """The current image cannot yet supply a measured belt plane."""

    def __init__(self, message: str, *, reason: str, details: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.details = {} if details is None else dict(details)


def _camera_facing_axes(surface, hull, camera_origin_base, belt_direction_base):
    rotation = surface.plane_to_base[:3, :3]
    direction = np.asarray(belt_direction_base, float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError('单侧带边检测需要有效带向')
    axis = (direction @ rotation)[:2]
    if np.linalg.norm(axis) < 1e-9:
        raise ValueError('单侧带边检测需要有效带向')
    axis /= np.linalg.norm(axis)
    side = np.array([-axis[1], axis[0]])
    camera = ((np.asarray(camera_origin_base)-surface.plane_to_base[:3, 3]) @ rotation)[:2]
    facing = float((camera-hull.mean(axis=0)) @ side)
    if abs(facing) < 1e-6:
        raise ValueError('相机投影位于带面中线上，无法区分近侧带边')
    side *= np.sign(facing)
    return axis, side


def near_camera_belt_clearance(surface, center_xy, camera_origin_base, belt_direction_base):
    """Signed lateral clearance at one centre (no cross-section: outside)."""
    hull = np.asarray(surface.hull_xy, float).reshape(-1, 2)
    axis, side = _camera_facing_axes(surface, hull, camera_origin_base, belt_direction_base)
    relative = hull-np.asarray(center_xy)
    along, across = relative @ axis, relative @ side
    crossings = []
    for i in range(len(hull)):
        j = (i+1) % len(hull)
        if abs(along[i]) < 1e-10:
            crossings.append(across[i])
        if along[i]*along[j] < 0:
            fraction = -along[i]/(along[j]-along[i])
            crossings.append(across[i]+fraction*(across[j]-across[i]))
    return float(max(crossings)) if crossings else float('-inf')


def near_camera_footprint_clearances(surface, base_points, camera_origin_base, belt_direction_base):
    """Project measured points onto the belt. Compare each with the near side
    at its own longitudinal station; beyond the two ends the strip is undefined.
    Use the same visible detection outline as the single-exposure detector.
    """
    outline = getattr(surface, 'detection_hull_xy', None)
    hull = np.asarray(surface.hull_xy if outline is None else outline, float).reshape(-1, 2)
    axis, side = _camera_facing_axes(surface, hull, camera_origin_base, belt_direction_base)
    transform = surface.plane_to_base
    xy = (base_points-transform[:3, 3]) @ transform[:3, :2]
    station = xy @ axis
    along, across = hull @ axis, hull @ side
    edge = np.full(len(xy), -np.inf)
    for i, end in enumerate(along):
        start = along[i-1]
        if abs(end-start) < 1e-12:
            at = np.abs(station-end) < 1e-10
            edge[at] = np.maximum(edge[at], max(across[i-1], across[i]))
        else:
            at = (station >= min(start, end)) & (station <= max(start, end))
            t = (station[at]-start)/(end-start)
            edge[at] = np.maximum(edge[at], across[i-1]+t*(across[i]-across[i-1]))
    edge[~np.isfinite(edge)] = np.nan
    return edge - xy @ side


def _geometry_fingerprint(frame: Any, d2c: Any) -> tuple:
    """Identify a camera geometry and clock session, independently of exposure."""
    def array_key(value):
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64)
        return array.shape, array.tobytes()

    def intrinsics_key(value):
        if value is None:
            return None
        return (int(value.width), int(value.height), array_key(value.matrix),
                array_key(value.distortion), getattr(value, 'distortion_model', None))

    rgb = getattr(frame, 'rgb', None)
    header = getattr(frame, 'header', None)
    return (getattr(frame, 'serial_number', ''), getattr(frame, 'calibration_id', ''),
            getattr(header, 'clock_epoch', None), tuple(frame.depth_m.shape),
            None if rgb is None else tuple(rgb.shape),
            intrinsics_key(frame.depth_intrinsics),
            intrinsics_key(getattr(frame, 'color_intrinsics', None)),
            array_key(frame.camera_to_world_cv), array_key(d2c))


def _scene_bounds_from_green_outline(frame, d2c, green_rgb, plane_to_base, native_bounds, *, include_hull=False):
    """Estimate the scene crop from visible green, without filling any depth.

    Depth density is not uniform at the stereo image edge. Its quantiles are
    still used by the original detector, but cannot define the full scene
    extent. Reuse the already-selected/eroded green component and measured
    plane to bound its outer contour, even where native depth is missing.
    """
    diagnostics = {'source': 'native_depth_quantiles', 'applied': False,
                   'scope': 'scene_roi_only', 'depth_points_added': 0,
                   'native_depth_modified': False,
                   'native_quantile_bounds_xy_m': native_bounds.tolist()}
    intrinsics = getattr(frame, 'color_intrinsics', None)
    rgb = getattr(frame, 'rgb', None)
    if (green_rgb is None or intrinsics is None or rgb is None
            or np.shape(green_rgb) != np.shape(rgb)[:2]):
        diagnostics['reason'] = 'green_outline_or_color_geometry_unavailable'
        return None, diagnostics
    contours, _ = cv2.findContours(np.uint8(np.asarray(green_rgb) > 0),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        diagnostics['reason'] = 'green_outline_empty'
        return None, diagnostics
    # green_belt_points already chose the largest component and eroded it.
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    try:
        rays_xy = cv2.undistortPoints(contour.astype(np.float64).reshape(-1, 1, 2),
            intrinsics.matrix, np.asarray(intrinsics.distortion, dtype=float)).reshape(-1, 2)
        rays = np.column_stack((rays_xy, np.ones(len(rays_xy))))
        color_to_base = np.asarray(frame.camera_to_world_cv) @ np.linalg.inv(np.asarray(d2c))
        origin = color_to_base[:3, 3]
        directions = rays @ color_to_base[:3, :3].T
        normal = plane_to_base[:3, 2]
        denominator = directions @ normal
        if (not np.isfinite(denominator).all() or np.any(np.abs(denominator) <= 1e-12)):
            raise ValueError('green_outline_parallel_to_plane')
        distance = ((plane_to_base[:3, 3]-origin) @ normal)/denominator
        if not np.isfinite(distance).all() or np.any(distance <= 0):
            raise ValueError('green_outline_not_in_front_of_camera')
        world = origin+distance[:, None]*directions
        flat = np.linalg.solve(plane_to_base[:3, :3], (world-plane_to_base[:3, 3]).T).T
        projected = np.column_stack((flat[:, :2].min(axis=0), flat[:, :2].max(axis=0)))
        if not np.isfinite(projected).all() or np.any(projected[:, 0] >= projected[:, 1]):
            raise ValueError('green_outline_degenerate')
    except (ValueError, np.linalg.LinAlgError, cv2.error) as error:
        diagnostics['reason'] = str(error)
        return None, diagnostics
    # Never shrink the original region; keep its usual expansion in the
    # downstream config. These intersections are bounds, not observed points.
    bounds = np.column_stack((np.minimum(native_bounds[:, 0], projected[:, 0]),
                              np.maximum(native_bounds[:, 1], projected[:, 1])))
    diagnostics.update(applied=True, source='green_rgb_outline_on_measured_plane',
        reason='applied', projected_contour_pixels=len(contour),
        projected_bounds_xy_m=projected.tolist(), scene_bounds_xy_m=bounds.tolist(),
        capture_sequence=getattr(getattr(frame, 'header', None), 'sequence', None),
        capture_clock_epoch=getattr(getattr(frame, 'header', None), 'clock_epoch', None))
    if include_hull:
        diagnostics['projected_hull_xy_m'] = cv2.convexHull(flat[:, :2].astype(np.float32)).tolist()
    return bounds, diagnostics


@dataclass(slots=True)
class BeltSurface:
    """Static belt plane, footprint and bounds in local plane coordinates."""

    plane_to_base: np.ndarray
    bounds_xy_m: np.ndarray
    hull_xy: np.ndarray
    plane: dict
    point_count: int = 0
    geometry_fingerprint: tuple | None = None
    scene_bounds_xy_m: np.ndarray | None = None
    boundary_diagnostics: dict = field(default_factory=dict)
    detection_hull_xy: np.ndarray | None = None
    detection_boundary_diagnostics: dict = field(default_factory=dict)

    def region_metadata(self) -> dict:
        """Keep native detection bounds separate from inferred scene bounds."""
        metadata = {'plane_to_base': self.plane_to_base.copy(),
                    'bounds_xy_m': self.bounds_xy_m.copy(),
                    'hull_xy_m': self.hull_xy.copy(),
                    'source': 'measured_green_belt_plane_native_depth',
                    'bounds_quantiles': [.02, .98], 'plane_points': self.point_count,
                    'scene_boundary_estimation': dict(self.boundary_diagnostics)}
        if self.scene_bounds_xy_m is not None:
            metadata['scene_bounds_xy_m'] = self.scene_bounds_xy_m.copy()
        return metadata

    def matches_frame(self, frame: Any, d2c: Any) -> bool:
        """Reuse only within the same camera session and unchanged calibration."""
        return (self.geometry_fingerprint is not None
                and self.geometry_fingerprint == _geometry_fingerprint(frame, d2c))

    @property
    def normal_base(self) -> np.ndarray:
        return self.plane_to_base[:3, 2].copy()

    @classmethod
    def from_frame(
        cls, frame: Any, d2c: Any, *, fit_plane: Callable,
        green_belt_points: Callable,
    ) -> 'BeltSurface':
        """Fit the visible green belt once for the existing local image ROI."""
        points, mask, green_rgb = green_belt_points(frame, d2c)
        points = np.asarray(points, dtype=float)
        measured = points[np.asarray(mask, dtype=bool)]
        measured = measured[np.isfinite(measured).all(axis=1)]
        from .frame_products import frame_products
        products = frame_products(frame, d2c)
        # Complete finite green extraction BEFORE the 5 mm plane-only subset.
        # The motion PCA needs exactly this full measured set and ordering.
        products.get(('measured_green_points', green_belt_points), lambda: measured)
        abc, plane = fit_plane(measured)
        abc = np.asarray(abc, dtype=float)
        normal = np.array(plane.get('normal_base', [-abc[0], -abc[1], 1.]),
                          dtype=float, copy=True)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError('plane normal must contain three finite values')
        normal /= np.linalg.norm(normal)
        ex = np.array([1., 0., 0.])
        ex -= normal * np.dot(ex, normal)
        if np.linalg.norm(ex) < 1e-9:
            ex = np.array([0., 1., 0.])
            ex -= normal * np.dot(ex, normal)
        ex /= np.linalg.norm(ex)
        plane_to_base = np.eye(4)
        plane_to_base[:3, :3] = np.column_stack((ex, np.cross(normal, ex), normal))
        plane_to_base[:3, 3] = [0., 0., abc[2]]
        flat = (measured - plane_to_base[:3, 3]) @ plane_to_base[:3, :3]
        # Keep the existing 5 mm support selection so off-plane green pixels
        # cannot enlarge the local belt ROI onto the table.
        belt = flat[np.abs(flat[:, 2]) < .005]
        hull = cv2.convexHull(belt[:, :2].astype(np.float32))
        lower, upper = np.quantile(belt[:, :2], [.02, .98], axis=0)
        bounds = np.column_stack((lower, upper))
        scene_bounds, diagnostics = _scene_bounds_from_green_outline(
            frame, d2c, green_rgb, plane_to_base, bounds)
        surface = cls(plane_to_base, bounds, hull, dict(plane), len(belt),
                      _geometry_fingerprint(frame, d2c), scene_bounds, diagnostics)
        # Erosion is useful for fitting the plane, but must not define where
        # an object may be detected. Keep this optical boundary separate from
        # the native-depth hull and all collision/scene geometry.
        if green_rgb is not None:
            return visible_detection_boundary(surface, frame, d2c)
        return surface


def visible_detection_boundary(surface, frame, d2c):
    """Private detection footprint from uneroded visible green on the real plane.

    No depth samples are added. Reject inconsistent optical projections and
    retain the original native detection boundary in that case.
    """
    from dataclasses import replace
    from .frame_products import green_rgb_mask
    green = green_rgb_mask(frame, d2c)
    _, report = _scene_bounds_from_green_outline(frame, d2c, green,
        surface.plane_to_base, surface.bounds_xy_m, include_hull=True)
    report = {**report, 'scope': 'detection_only', 'mask_erosion_pixels': 0}
    if not report.get('applied'):
        return replace(surface, detection_hull_xy=None, detection_boundary_diagnostics=report)
    hull = np.asarray(report.pop('projected_hull_xy_m'), dtype=np.float32)
    native = np.asarray(surface.hull_xy, dtype=np.float32)
    native_area, area = cv2.contourArea(native), cv2.contourArea(hull)
    overlap = cv2.intersectConvexConvex(native, hull)[0]
    ratio = area/native_area if native_area > 0 else float('inf')
    coverage = overlap/native_area if native_area > 0 else 0.
    report.update(area_ratio_to_native=ratio, native_area_coverage=coverage)
    if not .8 <= ratio <= 1.5 or coverage < .9:
        report.update(applied=False, reason='visible_outline_inconsistent_with_native_belt')
        hull = None
    return replace(surface, detection_hull_xy=hull, detection_boundary_diagnostics=report)
