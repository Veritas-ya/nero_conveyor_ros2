"""Explicit measured-belt-surface environment, using native observed points.

This scope intentionally excludes non-belt scenery. It does not disable the
robot's scene checker or change self/workpiece checks. No plane points are
synthesised, no images are reprojected, and indices refer to this exposure.
"""
import numpy as np
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.calibration.geometry import rigid_transform
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.scene_region_filter import voxel_downsample


OBSERVED_SCENE = 'observed_scene'
BELT_SURFACE_ONLY = 'measured_belt_surface_only'
SCOPES = (OBSERVED_SCENE, BELT_SURFACE_ONLY)
# Existing BeltSurface.from_frame support membership, not a new grip threshold.
BELT_SUPPORT_BAND_M = .005


def possible_belt_neighbors(points, origin, normal, distance_m):
    """Conservative projection broad phase for an installed belt-only scene."""
    height = (points-origin) @ normal
    reach = BELT_SUPPORT_BAND_M + distance_m*np.linalg.norm(normal) + 1e-12
    return np.abs(height) <= reach


def install_belt_surface(model, target):
    observation = target.detection
    region = observation.get('belt_region')
    if not isinstance(region, dict):
        raise PickRejected('仅带面碰撞模式缺少实测传送带边界')
    try:
        plane = rigid_transform(region['plane_to_base'], 'measured belt plane')
        bounds = np.asarray(region.get('scene_bounds_xy_m', region['bounds_xy_m']), dtype=float)
        points = np.asarray(observation['scene_points'], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise PickRejected('仅带面碰撞模式的实测输入无效') from error
    if (bounds.shape != (2,2) or not np.isfinite(bounds).all()
            or np.any(bounds[:,0] >= bounds[:,1])
            or points.ndim != 2 or points.shape[1] != 3):
        raise PickRejected('仅带面碰撞模式的点云或边界无效')
    frame = observation.get('frame')
    if frame is not None and (frame.header.sequence != target.sequence
            or frame.header.clock_epoch != target.epoch):
        raise PickRejected('带面点云与本次目标帧/时钟代次不匹配')
    finite = np.isfinite(points).all(axis=1)
    # Read only the already-deprojected points of the selected exposure.
    local = (points-plane[:3,3]) @ plane[:3,:3]
    inside = ((local[:,:2] >= bounds[:,0]) & (local[:,:2] <= bounds[:,1])).all(axis=1)
    selected = np.flatnonzero(finite & inside & (np.abs(local[:,2]) < BELT_SUPPORT_BAND_M))
    if not len(selected):
        raise PickRejected('本帧没有有效实测带面碰撞点；不会用空场景代替')
    sampled = voxel_downsample(points[selected], voxel_size_m=model.belt_scene_filter_config.voxel_size_m,
                               max_error_m=.004)
    retained = selected[sampled.retained_indices]
    scene = np.array(points[retained], copy=True)
    source_pixels = observation.get('scene_pixel_indices')
    native_pixels = None
    if source_pixels is not None:
        source_pixels = np.asarray(source_pixels)
        if source_pixels.shape != (len(points),) or source_pixels.dtype.kind not in 'iu':
            raise PickRejected('带面碰撞点缺少一致的原生像素索引')
        native_pixels = source_pixels[retained].copy()
        native_pixels.setflags(write=False)
    tree = cKDTree(scene)
    scene.setflags(write=False)
    retained.setflags(write=False)
    report = {'scope':BELT_SURFACE_ONLY, 'source':'native_scene_points_of_selected_exposure',
        'target_identity':list(target.identity),'frame_sequence':int(target.sequence),
        'clock_epoch':int(target.epoch),'capture_monotonic_s':float(target.capture_s),
        'calibration_id':None if frame is None else frame.calibration_id,
        'input_scene_points':int(observation.get('scene_input_count_before_prefilter', len(points))),
        'scene_prefilter_scope':observation.get('scene_prefilter_scope', OBSERVED_SCENE),
        'prefiltered_scene_points':len(points),'native_belt_points':len(selected),
        'excluded_non_belt_points':int(observation.get('scene_input_count_before_prefilter', len(points)))-len(selected),'collision_points':len(scene),
        'support_band_m':BELT_SUPPORT_BAND_M,'bounds_xy_m':bounds.tolist(),
        'plane_to_base':plane.tolist(),'downsample':sampled.statistics,
        'synthetic_points':0,'rails_modelled':False,
        'coverage':'visible native belt surface only; non-belt scenery excluded by configured scope'}
    # Publish a complete, immutable scene/index pair only after validation.
    model.scene, model.tree = scene, tree
    model.target = np.empty((0,3))  # moving workpiece checked by MeasuredPlanner
    model.scene_voxel_result = sampled
    model.scene_source_indices = {'belt_surface_retained':retained}
    model.scene_native_pixel_indices = ({} if native_pixels is None else
                                      {'belt_surface_retained':native_pixels})
    model._observed_scene_index = None
    return report
