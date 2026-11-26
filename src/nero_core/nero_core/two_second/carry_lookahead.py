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
from .prepared_cycle import nominal_state, prepare_cycle, prepare_empty_return
from .runtime_sweep import scene_snapshot, restore_scene


_planner = _cancel = _return_generation = None


def _initialize(config, cancel, return_generation):
    global _planner, _cancel, _return_generation
    _return_generation = return_generation
    from .planning import MeasuredPlanner
    from .taught import prepare_taught_timing
    _planner, _cancel = MeasuredPlanner(config), cancel
    for kind, segment in config.raw['taught_segments'].items():
        prepare_taught_timing(config, segment, _planner.model, payload=kind == 'transfer')


def _ping():
    return {'pid': os.getpid(), 'hardware_connections': 0}


def predict_cycle(planner, snapshot, reference, *, cancelled, carry_only=False):
    """All start/payload data here are predictions, never measured feedback."""
    restore_scene(planner, snapshot)
    planner.bin_departure_tree = None  # This job begins after the next grasp.
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
    bundle = prepare_cycle(planner, planner.config, state, path.q1, cancelled=cancelled,
        carry_only=carry_only)
    return bundle


def _job(snapshot, reference, carry_only):
    began = time.monotonic()
    result = dict(token=snapshot['token'], identity=snapshot['identity'],
        scene_revision=snapshot['scene_revision'], source_version=snapshot['path'].version,
        predicted_start_joints=snapshot['path'].q1.copy(), pid=os.getpid())
    try:
        result['bundle'] = predict_cycle(_planner, snapshot, reference, cancelled=_cancel.is_set, carry_only=carry_only)
        result['ok'] = not _cancel.is_set()
    except Exception as error:
        result.update(ok=False, error=str(error), details=getattr(error, 'details', {}))
    result.update(planning_wall_s=time.monotonic()-began, completed_monotonic_s=time.monotonic())
    return result


def _return_job(snapshot, width, recovery, generation):
    def cancelled():
        return generation != _return_generation.value
    if cancelled():
        raise PickRejected('回程预规划已取消')
    restore_scene(_planner, snapshot)
    _planner.payload_local = snapshot['payload_local']
    _planner.payload_tree = cKDTree(_planner.payload_local)
    carry = snapshot['path']
    drop = nominal_state(_planner.model, carry.q1, width)
    _planner.release_measured(drop)
    path = prepare_empty_return(_planner, _planner.config, carry.q1, cancelled=cancelled, recovery=recovery)
    return path


def _validate_job(request):
    from .carry_validation import validate_in_private_model
    return validate_in_private_model(_planner,request,cancelled=_cancel.is_set)


class CarryLookahead:
    """One warmed process and one cancellable job, shared across cycles."""
    def __init__(self, config):
        context = mp.get_context('spawn')
        self.cancel = context.Event()
        self.return_generation = context.Value('L', 0)
        # Runtime callbacks/caches belong to the parent; send only config and
        # taught samples, with caches rebuilt once in this private model.
        private = replace(config, raw=deepcopy({k:v for k,v in config.raw.items()
                                               if not k.startswith('_')}))
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=context,
            initializer=_initialize, initargs=(private, self.cancel, self.return_generation))
        self.future = None
        try:
            self.worker = self.pool.submit(_ping).result(timeout=30.)
        except BaseException:
            self.close()
            raise

    def submit(self, planner, path, reference, *, carry_only=False):
        if self.future is not None and not self.future.done():
            raise PickRejected('上一搬运预规划仍在退出，本轮使用实际状态规划')
        self.cancel.clear()
        snapshot = scene_snapshot(planner, path)
        self.future = self.pool.submit(_job, snapshot, reference, carry_only)
        return snapshot['token'], self.future

    def validate(self, planner, state, anchor, result, *, token, source_version):
        """Use the warmed private model; the parent keeps sending rest targets."""
        from .carry_validation import validation_request
        if self.cancel.is_set() or (self.future is not None and not self.future.done()):
            raise PickRejected('搬运后台复核已取消或上一任务尚未结束')
        request = validation_request(planner,state,anchor,result,token=token,
                                     source_version=source_version)
        self.future = self.pool.submit(_validate_job,request)
        return request,self.future

    def plan_return(self, planner, carry, width, *, recovery):
        snapshot = scene_snapshot(planner, carry)
        snapshot['payload_local'] = planner.payload_local.copy()
        with self.return_generation.get_lock():
            self.return_generation.value += 1
            generation = self.return_generation.value
        return self.pool.submit(_return_job, snapshot, width, recovery, generation)

    def discard(self):
        with self.return_generation.get_lock():
            self.return_generation.value += 1
        self.cancel.set()
        if self.future is not None:
            self.future.cancel()

    def close(self):
        self.discard()
        self.pool.shutdown(wait=True, cancel_futures=True)


def validate_inputs(planner, state, anchor, result, *, token, source_version, cancelled):
    """Identity, actual attachment and feedback checks shared by both workers."""
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
    return bundle,planner.model.tcp().copy()


def revalidate_cycle(planner, state, anchor, result, *, token, source_version, cancelled,
                     defer_return_validation=False):
    """Reuse only curve coefficients; certify actual width/payload and scene."""
    bundle,actual_tcp = validate_inputs(planner,state,anchor,result,token=token,
        source_version=source_version,cancelled=cancelled)
    anchor = np.asarray(anchor)
    saved = {key:getattr(planner,key,None) for key in
        ('payload_local','payload_tree','capture_tcp','released_points','released_tree')}
    began = time.monotonic()
    try:
        width = float(state['width_m'])
        planner.check_path(bundle.carry, width, payload=True, cancelled=cancelled)
        at_drop = nominal_state(planner.model, bundle.carry.q1, width)
        planner.validate_release_opening(at_drop)
        if not defer_return_validation and bundle.returning is not None:
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
            'return_validation':('deferred_until_actual_release' if defer_return_validation or bundle.returning is None else 'completed'),
            'path_search_reused':True, 'physical_grasp_proven':False}
    finally:
        for key,value in saved.items():
            setattr(planner,key,value)
        planner.model.set_state(state['joints'],state['width_m'])
