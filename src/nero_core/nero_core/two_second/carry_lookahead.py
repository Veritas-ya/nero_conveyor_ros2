"""Speculative carry planning in a private process; no device API in workers."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import replace
import multiprocessing as mp
import os
import time

import numpy as np
from scipy.spatial import cKDTree

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .prepared_cycle import nominal_state, prepare_cycle
from .runtime_sweep import scene_snapshot, restore_scene


_planner = _cancel = None


def _initialize(config, cancel):
    global _planner, _cancel
    from .planning import MeasuredPlanner
    from .taught import prepare_taught_timing
    _planner, _cancel = MeasuredPlanner(config), cancel
    for kind, segment in config.raw['taught_segments'].items():
        prepare_taught_timing(config, segment, _planner.model, payload=kind == 'transfer')


def _ping():
    return {'pid': os.getpid(), 'hardware_connections': 0}


def predict_cycle(planner, snapshot, reference, *, cancelled):
    """All start/payload data here are predictions, never measured feedback."""
    restore_scene(planner, snapshot)
    path = snapshot['path']
    if any(np.max(abs(path.curve(path.duration, d))) > 1e-7 for d in (1,2)):
        raise PickRejected('搬运预规划要求原抓取轨迹以零速度、零加速度结束')
    # Reference follow end is the attachment hypothesis. Motion continues to
    # the certified brake endpoint, which is the subsequent carry's q0.
    stamp = reference.start_s + reference.follow_end_s
    width = planner.config.settings.open_width_m
    planner.model.set_state(path.position(reference.follow_end_s), width)
    capture = planner.model.tcp().copy()
    planner.capture_tcp = capture
    planner.payload_local = (planner.target_points_at(stamp)-capture[:3, 3]) @ capture[:3, :3]
    planner.payload_tree = cKDTree(planner.payload_local)
    state = nominal_state(planner.model, path.q1, width)
    planner.capture_tcp = planner.model.tcp().copy()  # predicted brake endpoint support anchor
    bundle = prepare_cycle(planner, planner.config, state, path.q1, cancelled=cancelled)
    return bundle


def _job(snapshot, reference):
    began = time.monotonic()
    result = dict(token=snapshot['token'], identity=snapshot['identity'],
        scene_revision=snapshot['scene_revision'], source_version=snapshot['path'].version,
        predicted_start_joints=snapshot['path'].q1.copy(), pid=os.getpid())
    try:
        result['bundle'] = predict_cycle(_planner, snapshot, reference, cancelled=_cancel.is_set)
        result['ok'] = not _cancel.is_set()
    except Exception as error:
        result.update(ok=False, error=str(error), details=getattr(error, 'details', {}))
    result.update(planning_wall_s=time.monotonic()-began, completed_monotonic_s=time.monotonic())
    return result


class CarryLookahead:
    """One warmed process and one cancellable job, shared across cycles."""
    def __init__(self, config):
        context = mp.get_context('spawn')
        self.cancel = context.Event()
        # Runtime callbacks/caches belong to the parent; send only config and
        # taught samples, with caches rebuilt once in this private model.
        private = replace(config, raw=deepcopy({k:v for k,v in config.raw.items()
                                               if not k.startswith('_')}))
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=context,
            initializer=_initialize, initargs=(private, self.cancel))
        self.future = None
        try:
            self.worker = self.pool.submit(_ping).result(timeout=30.)
        except BaseException:
            self.close()
            raise

    def submit(self, planner, path, reference):
        if self.future is not None and not self.future.done():
            raise PickRejected('上一搬运预规划仍在退出，本轮使用实际状态规划')
        self.cancel.clear()
        snapshot = scene_snapshot(planner, path)
        self.future = self.pool.submit(_job, snapshot, reference)
        return snapshot['token'], self.future

    def discard(self):
        self.cancel.set()
        if self.future is not None:
            self.future.cancel()

    def close(self):
        self.discard()
        self.pool.shutdown(wait=True, cancel_futures=True)


def revalidate_cycle(planner, state, anchor, result, *, token, source_version, cancelled,
                     defer_return_validation=False):
    """Reuse only curve coefficients; certify actual width/payload and scene."""
    bundle = result.get('bundle')
    if (not result.get('ok') or bundle is None or result.get('token') != token
            or result['identity'] != planner.identity or bundle.identity != planner.identity
            or result['source_version'] != source_version or source_version != planner.version
            or result['scene_revision'] != planner.scene_revision
            or bundle.scene_revision != planner.scene_revision):
        raise PickRejected('搬运预规划结果失效或未交付', details={'prediction_error':result.get('error')})
    if cancelled():
        raise PickRejected('搬运预规划复核已取消')
    anchor = np.asarray(anchor)
    if (np.max(abs(bundle.carry.q0-anchor)) > 1e-9
            or np.max(abs(np.asarray(state['joints'])-anchor)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD):
        raise PickRejected('实际减速终点与搬运预规划起点不符')
    if planner.payload_local is None:
        raise PickRejected('搬运预规划缺少实际夹持载物估计')
    planner.model.check_feedback(state)
    actual_tcp = planner.model.tcp().copy()
    saved = {key:getattr(planner,key,None) for key in
        ('payload_local','payload_tree','capture_tcp','released_points','released_tree')}
    began = time.monotonic()
    try:
        width = float(state['width_m'])
        planner.check_path(bundle.carry, width, payload=True, cancelled=cancelled)
        at_drop = nominal_state(planner.model, bundle.carry.q1, width)
        planner.validate_release_opening(at_drop)
        if not defer_return_validation:
            planner.release_measured(at_drop)
            planner.check_path(bundle.returning, planner.config.settings.open_width_m,
                payload=False, cancelled=cancelled)
        if cancelled():
            raise PickRejected('搬运预规划复核已取消')
        planner.model.set_state(anchor, width)
        error = float(np.linalg.norm(planner.model.tcp()[:3,3]-actual_tcp[:3,3]))
        return bundle, {'validation_wall_s':time.monotonic()-began,
            'predicted_actual_tcp_error_m':error,
            'max_start_joint_error_rad':float(np.max(abs(np.asarray(state['joints'])-anchor))),
            'actual_width_m':width, 'payload_source':'actual_grip_time_visual_attachment_estimate',
            'return_validation':('deferred_until_actual_release' if defer_return_validation else 'completed'),
            'path_search_reused':True, 'physical_grasp_proven':False}
    finally:
        for key,value in saved.items():
            setattr(planner,key,value)
        planner.model.set_state(state['joints'],state['width_m'])
