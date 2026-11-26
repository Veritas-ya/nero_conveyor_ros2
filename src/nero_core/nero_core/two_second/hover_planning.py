"""Prepare a runnable prefix/stop pair, then plan the descent on a private process."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import product
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import PPoly
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .hover_config import KEY
from .hover_math import (HoverPrefix, prepare, descent, fallback, MovingJoin,
                         cartesian_state, require_limits)
from .planning import MeasuredPlanner, InterceptPlan
from .runtime_sweep import restore_scene
from .synchronized_reference import SynchronizedIntercept
from .trajectory import TimedPath
from .velocity_source import is_fixed


@dataclass
class PreparedHover:
    old: object
    prefix: TimedPath
    stopping: TimedPath
    reference: HoverPrefix
    switch_s: float
    scene_revision: int
    grasp: np.ndarray
    evidence: dict


def concatenate(prefix, suffix):
    if prefix.identity != suffix.identity or prefix.version != suffix.version:
        raise PickRejected('分段轨迹身份或版本不一致')
    for d in (0, 1, 2):
        if np.max(abs(prefix.curve(prefix.duration, d)-suffix.curve(0., d))) > 1e-7:
            raise PickRejected('分段接续位置/速度/加速度不连续')
    curve = PPoly(np.concatenate((prefix.curve.c, suffix.curve.c), axis=1),
                  np.r_[prefix.curve.x, suffix.curve.x[1:]+prefix.duration], extrapolate=False)
    return TimedPath(curve, prefix.identity, prefix.version, 'hover_descent_staged',
                     prefix.duration+suffix.duration, prefix.planning_s+suffix.planning_s)


def shell_reference(bundle, planner, state, duration):
    qva = bundle.prefix.sample(bundle.switch_s)
    tcp, twist, acc = cartesian_state(planner.model, qva, state['width_m'])
    r = bundle.old.reference
    return MovingJoin(capture_s=r.capture_s, start_s=r.start_s+bundle.switch_s,
        duration_s=duration, capture_tcp_world=r.capture_tcp_world, velocity_world=r.velocity_world,
        start_tcp_world=tcp[:3, 3], start_quaternion_wxyz=Rotation.from_matrix(tcp[:3, :3]).as_quat()[[3,0,1,2]],
        target_quaternion_wxyz=r.target_quaternion_wxyz, follow_s=r.follow_s, brake_s=r.brake_s,
        prealign_s=duration, linear_velocity=twist[:3], linear_acceleration=acc[:3],
        angular_velocity=twist[3:], angular_acceleration=acc[3:])


def prepare_hover(planner, target, state, *, cancelled=lambda: False):
    cfg = planner.config; spec = cfg.raw[KEY]; begin = planner.clock()
    speed = float(np.linalg.norm(target.velocity))
    limit = spec['maximum_speed_m_s']
    if is_fixed(cfg) and cfg.raw['fixed_belt_speed_m_s'] <= limit:
        # Projecting an admitted scalar onto the calibrated unit direction
        # can round its reconstructed vector norm up by one float.
        limit = np.nextafter(limit, np.inf)
    if not 0 < speed <= limit or target.velocity@cfg.direction <= 0:
        raise PickRejected('目标带速超出分段接续低速测试范围')
    if not 0 <= begin-target.capture_s <= cfg.raw['maximum_observation_age_s']:
        raise PickRejected('分段接续观测已过期')
    planner.model.check_feedback(state)
    start_tcp = planner.model.tcp().copy()
    planner.install(target, state)
    installed = planner.clock()
    # This is a new prospective schedule. Never backdate prefix execution.
    origin = installed+spec['preparation_reserve_s']+cfg.raw['command_advance_s']
    deadline = origin-cfg.raw['command_advance_s']-spec['send_guard_s']
    planner._hover_cancel = lambda: cancelled() or planner.clock() >= deadline
    failures = []
    grasps = sorted(enumerate(target.grasps[:2]), key=lambda item:
        Rotation.from_matrix(start_tcp[:3,:3].T@item[1][:3,:3]).magnitude())
    try:
        # Try the finger-symmetric pose before spending the budget on every
        # duration of one bad wrist branch. Coarse curves still receive the
        # original exact extrema, FK and collision checks; density can refine.
        candidates = product((10., 25., 50.), spec['hover_heights_m'], spec['ramp_times_s'],
                             spec['alignment_fractions'], grasps)
        for ik_hz, height, ramp, alignment, (grasp_index, grasp) in candidates:
            if planner._hover_cancel():
                raise PickRejected('前段及停车退路未在启动截止前完成', details={
                    'failures':failures, 'origin_s':origin, 'deadline_s':deadline,
                    'wall_s':planner.clock()-begin, 'scene_prepare_s':installed-begin})
            candidate_begin = planner.clock()
            context = dict(height_m=height, ramp_s=ramp, alignment_fraction=alignment,
                input_grasp_index=grasp_index, ik_hz=ik_hz, origin_s=origin)
            r = SynchronizedIntercept(target.capture_s, origin, spec['descent_times_s'][0],
                grasp[:3,3], target.velocity, start_tcp[:3,3],
                Rotation.from_matrix(start_tcp[:3,:3]).as_quat()[[3,0,1,2]],
                Rotation.from_matrix(grasp[:3,:3]).as_quat()[[3,0,1,2]],
                cfg.settings.full_plan_close_window_s, cfg.raw['brake_duration_s'],
                spec['descent_times_s'][0])
            old = SimpleNamespace(reference=r, target=SimpleNamespace(identity=target.identity,
                velocity=target.velocity.copy(), capture_s=target.capture_s))
            try:
                ref = HoverPrefix(old, cfg.direction, ramp, spec['sync_hold_s'], height,
                    alignment_fraction=alignment, bin_join_min_m=(cfg.follow_bounds[0]
                    if getattr(planner, 'bin_departure_tree', None) is not None else None))
                context.update(wait_s=ref.wait_s, sync_s=ref.sync_s,
                               rotation_end_s=ref.rotation_end_s)
                prefix, pcheck = prepare(planner, old, state, ref, ik_hz=ik_hz)
                prefix = replace(prefix, version=planner.version)
                stop, scheck = fallback(planner, old, state, prefix, ref.full_duration_s)
                stop = replace(stop, version=planner.version)
                full_stop = concatenate(prefix, stop)
                require_limits(full_stop.curve, cfg, planner.model)
                evidence = dict(prefix=pcheck, fallback=scheck, failed_candidates=failures,
                    scene_prepare_s=installed-begin, wall_s=planner.clock()-begin,
                    origin_s=origin, sync_s=ref.sync_s, switch_s=ref.full_duration_s,
                    height_m=height, ramp_s=ramp, speed_m_s=speed,
                    alignment_fraction=alignment, input_grasp_index=grasp_index,
                    rotation_end_s=ref.rotation_end_s, wait_s=ref.wait_s,
                    dispatch_deadline_s=origin-cfg.raw['command_advance_s'])
                if planner._hover_cancel():
                    raise PickRejected('前段及停车退路完成时已错过启动截止')
                return PreparedHover(old, prefix, stop, ref, ref.full_duration_s,
                                     planner.scene_revision, grasp.copy(), evidence)
            except (PickRejected, ValueError) as error:
                failures.append(dict(**context, wall_s=planner.clock()-candidate_begin,
                    error=str(error), details=getattr(error,'details',{})))
        raise PickRejected('没有通过检查的同步前段及停车退路', details={'failures':failures})
    finally:
        planner._hover_cancel = lambda: False


_worker = _cancel = None


def _initialize(config, cancel):
    global _worker, _cancel
    _worker, _cancel = MeasuredPlanner(config), cancel


def _ping():
    return dict(pid=os.getpid(), hardware_connections=0)


def plan_suffix(planner, bundle, state, *, cancelled=lambda: False, deadline=float('inf')):
    planner._hover_cancel = lambda: cancelled() or planner.clock() >= deadline
    attempts = []
    try:
        for duration in planner.config.raw[KEY]['descent_times_s']:
            if planner._hover_cancel():
                break
            row, path = descent(planner, bundle.old, state, bundle.prefix, bundle.switch_s, duration)
            attempts.append(row)
            if path is not None and not planner._hover_cancel():
                path = replace(path, version=bundle.prefix.version,
                    diagnostics={'phase_timing':row['phase'], 'hover_descent':row})
                full = concatenate(bundle.prefix, path)
                require_limits(full.curve, planner.config, planner.model)
                reference = shell_reference(bundle, planner, state, duration)
                return dict(ok=True, path=path, reference=reference, attempts=attempts,
                    identity=bundle.prefix.identity, version=bundle.prefix.version,
                    scene_revision=bundle.scene_revision, completed_s=planner.clock())
        return dict(ok=False, attempts=attempts, error='后段规划失败或已到交付截止', completed_s=planner.clock())
    finally:
        planner._hover_cancel = lambda: False


def _job(snapshot, bundle, state, deadline):
    restore_scene(_worker, snapshot)
    try:
        return plan_suffix(_worker, bundle, state, cancelled=_cancel.is_set, deadline=deadline)
    except Exception as error:
        return dict(ok=False,error=str(error),details=getattr(error,'details',{}),completed_s=time.monotonic())


class HoverService:
    """Warm before camera/CAN; worker owns no device and no live planner model."""
    def __init__(self, config):
        context = mp.get_context('spawn')
        self.cancel = context.Event()
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=context,
            initializer=_initialize, initargs=(config,self.cancel))
        try:
            self.worker = self.pool.submit(_ping).result(timeout=20.)
        except BaseException:
            self.close()
            raise

    def submit(self, snapshot, bundle, state, deadline):
        self.cancel.clear()
        return self.pool.submit(_job, snapshot, bundle, state, deadline)

    def close(self):
        self.cancel.set()
        self.pool.shutdown(wait=True, cancel_futures=True)
