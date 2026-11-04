"""Bridge the existing measured-grasp algorithms without opening devices.

The returned candidate is a *prefiltered* capture-time grasp, not a certified
trajectory. The caller must use the original full planner on that candidate.
"""
from dataclasses import dataclass
from time import monotonic
from typing import Any

from mujoco_conveyor_sorting.runtime.nero_graspgenx import (
    PreparedGraspInference, select_measured_network_grasp,
)


@dataclass(frozen=True)
class PreparedSelection:
    observation: dict
    scene: Any
    timings: dict


def prepare_and_select(observation, model, backend, *, motion_region, settings,
                       snapshot_provider, seed=0, config=None,
                       scene_options=None, clock=monotonic):
    """Overlap original inference and native-scene installation, then select.

    The caller owns and starts the persistent backend and supplies an already
    refined observation and measured motion snapshot provider. Scene options
    are passed to the original RobotModel.install_scene without changing its
    collision or filtering rules. Never call concurrently for one model.
    """
    if observation.get('target_identity') is None:
        raise ValueError('a confirmed target identity is required')
    if settings is None or snapshot_provider is None:
        raise ValueError('live settings and measured motion provider are required')
    frame = observation['frame']
    if frame.camera_to_world_cv is None:
        raise ValueError('calibrated frame is required')
    state = observation['scene_state']
    options = dict(scene_options or {})
    # Match the original live entry; replay callers may explicitly override.
    options.setdefault('gripper_nearby_scene_ignore', True)
    background = getattr(model, 'scene_background_reference', None)
    if background is not None:
        options.setdefault('scene_background_reference', background)
    if observation.get('scene_pixel_indices') is not None:
        options.setdefault('scene_pixel_indices', observation['scene_pixel_indices'])
    started = clock()
    prepared = PreparedGraspInference(observation, backend, seed=seed, config=config)
    try:
        scene_started = clock()
        scene = model.install_scene(
            observation['scene_points'], observation['object_points'], state,
            camera_origin=frame.camera_to_world_cv[:3, 3],
            belt_region=observation.get('belt_region'), depth_frame=frame,
            target_mask=observation.get('mask'), **options)
        scene_finished = clock()
        selected = select_measured_network_grasp(
            observation, model, state, backend, motion_region=motion_region,
            seed=seed, config=config, prepared_inference=prepared,
            snapshot_provider=snapshot_provider, settings=settings)
        selection_finished = clock()
        timings = {'preparation_start_monotonic_s': started,
                   'scene_start_monotonic_s': scene_started,
                   'scene_finish_monotonic_s': scene_finished,
                   'selection_finish_monotonic_s': selection_finished,
                   'scene_wall_s': scene_finished - scene_started,
                   'selection_after_scene_wall_s': selection_finished - scene_finished,
                   'overlapped_delivery_wall_s': selection_finished - started,
                   'inference': dict(prepared.timings)}
        return PreparedSelection(selected, scene, timings)
    finally:
        prepared.close()
