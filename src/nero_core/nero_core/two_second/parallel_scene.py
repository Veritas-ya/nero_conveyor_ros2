"""Prepare one exposure's belt scene alongside IK, without sharing robot state."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
import time

import numpy as np
from scipy.spatial import cKDTree

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .belt_collision_scene import install_belt_surface


SCENE_FIELDS = ('scene', 'tree', 'target', 'scene_voxel_result', 'scene_source_indices',
                'scene_native_pixel_indices', '_observed_scene_index')


def _readonly(value):
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def snapshot_target(target):
    source = target.detection
    observation = {key:source[key] for key in
        ('frame', 'scene_input_count_before_prefilter', 'scene_prefilter_scope') if key in source}
    observation['scene_points'] = _readonly(source['scene_points'])
    if source.get('scene_pixel_indices') is not None:
        observation['scene_pixel_indices'] = _readonly(source['scene_pixel_indices'])
    observation['belt_region'] = deepcopy(source['belt_region'])
    return SimpleNamespace(detection=observation, identity=target.identity,
        sequence=target.sequence, epoch=target.epoch, capture_s=target.capture_s,
        points=_readonly(target.points))


def _prepare(snapshot, filter_config):
    started = time.perf_counter()
    shadow = SimpleNamespace(belt_scene_filter_config=filter_config)
    report = install_belt_surface(shadow, snapshot)
    tree = cKDTree(snapshot.points)
    return shadow, report, tree, time.perf_counter()-started


class ParallelBeltScene:
    """Lazy, bounded worker. Discarded candidates never publish its result."""
    def __init__(self, planner, target):
        self.planner, self.target = planner, target
        self.pool = self.future = None
        self.timings = {}

    def __enter__(self):
        return self

    def start(self):
        if self.future is not None:
            return
        started = time.perf_counter()
        self.version = self.planner.version
        self.scene_revision = self.planner.scene_revision
        snapshot = snapshot_target(self.target)
        self.timings['scene_prepare_snapshot_s'] = time.perf_counter()-started
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pick-belt-scene')
        self.future = self.pool.submit(_prepare, snapshot, self.planner.model.belt_scene_filter_config)

    def _current(self, target):
        if (target is not self.target or self.future is None
                or self.planner.version != self.version
                or self.planner.scene_revision != self.scene_revision):
            raise PickRejected('并行场景结果与当前目标或规划版本不一致')

    def install(self, target, state, *, cancelled=lambda: False):
        self._current(target)
        began = time.perf_counter()
        shadow, report, tree, worker_s = self.future.result()
        # A cancellation/version change can arrive while waiting for the job.
        self._current(target)
        if cancelled():
            raise PickRejected('并行场景准备完成时本次规划已取消或超时')
        self.timings.update(scene_prepare_wait_s=time.perf_counter()-began,
                            scene_prepare_worker_s=worker_s)
        self.planner.install(target, state, prepared_scene=(shadow, report, tree))

    def __exit__(self, *exc):
        if self.pool is not None:
            # No robot/scene mutation occurs in the worker. Join before allowing
            # the next request, including exceptions and cancellation.
            self.future.cancel()
            self.pool.shutdown(wait=True, cancel_futures=True)
