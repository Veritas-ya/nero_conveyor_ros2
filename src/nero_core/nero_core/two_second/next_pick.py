"""Plan the next frozen exposure from standby and the drop endpoint off-device."""
from concurrent.futures import Future, ProcessPoolExecutor
from collections import deque
from copy import copy
from dataclasses import dataclass, replace
from math import factorial
import multiprocessing as mp
import os
from threading import RLock
from types import SimpleNamespace
import time

import numpy as np
from scipy.interpolate import BPoly, PPoly
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .lookahead_selection import latest_dispatch
from .prepared_cycle import nominal_state
from .belt_collision_scene import BELT_SURFACE_ONLY
from ..pick_origin_policy import PREFER_BIN, PREFER_WAIT, WAIT_ONLY


# On-device bin recertification took up to 0.64 s before dispatch preparation.
INITIAL_BIN_HANDOFF_CHECK_S = .70


def worker_target(target, config):
    """Retain the measured inputs to planner.install without serializing perception caches."""
    detection = target.detection
    observation = {key: detection[key] for key in ('scene_points', 'scene_pixel_indices',
        'belt_region', 'plane', 'scene_input_count_before_prefilter', 'scene_prefilter_scope',
        'depth_to_color_cv', 'camera_origin_base_m') if key in detection}
    frame = detection.get('frame')
    if config.raw.get('environment_collision_scope') == BELT_SURFACE_ONLY:
        # This scope checks exposure identity and calibration, then builds the
        # same voxel index from the native belt points and source pixel indices.
        if frame is not None:
            observation['frame'] = SimpleNamespace(header=frame.header, calibration_id=frame.calibration_id)
    else:
        if 'mask' in detection:
            observation['mask'] = detection['mask']
        if frame is not None:
            # Full-scene filtering needs the depth frame and target mask. The
            # lazy image-products cache carries weakrefs and a process-local lock.
            frame = copy(frame)
            vars(frame).pop('_perception_products', None)
            observation['frame'] = frame
    fields = dict(vars(target), detection=observation)
    return SimpleNamespace(**fields)


@dataclass(frozen=True)
class Candidate:
    previous_owner: tuple
    identity: tuple
    origin: str
    plan: object
    submitted_s: float
    ready_s: float

    def deadline(self, config):
        return self.plan.reference.start_s - config.raw['command_advance_s']

    def check_time(self):
        return sum(self.plan.timings_s.get(k, 0.) for k in
                   ('scene_prepare', 'polynomial_limits', 'sampled_geometry'))


def _initialize(config, generation):
    global _planner, _generation
    from .planning import MeasuredPlanner
    _planner, _generation = MeasuredPlanner(config), generation


def _ping():
    return os.getpid()


def _plan(generation, owner, target, origin, window, earliest, submitted):
    p = _planner
    def cancelled():
        return _generation[origin].value != generation
    try:
        if cancelled():
            raise PickRejected('预规划所属周期已结束')
        q, payload_local, planning_lead, post_lead, wait_joints = window
        p.intercept_not_before_s = earliest
        p.measured_planning_lead_s, p.measured_post_planning_s = planning_lead, post_lead
        if hasattr(p, 'planning_time_history'):
            del p.planning_time_history
        p.bin_departure_tree = None
        width = p.config.settings.open_width_m
        if origin == 'bin':
            state = nominal_state(p.model, q, width)
            tcp = p.model.tcp()
            p.bin_departure_tree = cKDTree(payload_local @ tcp[:3,:3].T + tcp[:3,3])
        else:
            state = nominal_state(p.model, wait_joints, width)
        plan = p.intercept(target, state, cancelled=cancelled)
        # Only the certified curve crosses processes. The execution owner
        # installs the original exposure and verifies the actual scene.
        return Candidate(owner, target.identity, origin, replace(plan, target=None), submitted, p.clock()), None
    except PickRejected as error:
        return None, str(error)


class NextPickPlans:
    def __init__(self, config, *, clock=time.monotonic, emit=None):
        self.config, self.clock = config, clock
        self.emit = emit
        ctx = mp.get_context('spawn')
        self.generations = {origin:ctx.Value('L', 0) for origin in ('wait', 'bin')}
        self.pool = ProcessPoolExecutor(2, mp_context=ctx, initializer=_initialize,
            initargs=(replace(config, raw={k:v for k,v in config.raw.items() if not k.startswith('_')}),
                      self.generations))
        self.lock = RLock()
        self.owner = self.target = self.worker_input = None
        self.forecast_future = self.brake_end = self.window = None
        self.previous_motion = None
        self.brake_to_carry = self.margin + config.raw['command_advance_s']
        self.jobs = {}
        self.requested_dispatch = {}
        self.wait_joints = config.standby.copy()
        self.wait_check_times = deque(maxlen=config.raw.get('planning_time_window_size', 8))
        self.bin_check_times = deque(maxlen=self.wait_check_times.maxlen)
        self.release_lag = self.margin + .15  # Opening confirmation plus endpoint/feedback overhead.
        try:
            warm = [self.pool.submit(_ping) for _ in range(2)]
            for job in warm:
                job.result(timeout=30.)
        except BaseException:
            self.close()
            raise

    @property
    def margin(self):
        return self.config.raw['lookahead_return_margin_s']

    @property
    def policy(self):
        return self.config.raw.get('next_pick_policy', PREFER_WAIT)

    def clear(self):
        with self.lock:
            self._cancel_jobs()
            self.owner = self.target = self.worker_input = None
            self.forecast_future = self.brake_end = self.window = None

    def _cancel_jobs(self):
        for origin in tuple(self.jobs):
            self._cancel_job(origin)

    def _cancel_job(self, origin):
        job = self.jobs.pop(origin, None)
        if job is not None:
            self.generations[origin].value += 1
            self.requested_dispatch.pop(origin, None)
            job.cancel()

    def close(self):
        self.clear()
        self.pool.shutdown(wait=True, cancel_futures=True)

    def _set_window(self, drop_joints, payload_local, leads, carry_end, return_duration, source):
        self.window = (drop_joints.copy(), payload_local, *leads, self.wait_joints.copy())
        self.carry_end = carry_end
        self.return_lead = (None if return_duration is None else
            self.config.raw['command_advance_s'] + return_duration + self.margin)
        self.schedule_source = source

    @staticmethod
    def _planning_leads(planner):
        return (float(getattr(planner, 'measured_planning_lead_s', 0.)),
                float(getattr(planner, 'measured_post_planning_s', 0.)))

    def anticipate(self, planner, path, reference, *, token, future):
        """Prepare the permitted origins from the first intercept's forecast."""
        target = planner.target
        if target.identity != path.identity or target.capture_s != reference.capture_s:
            raise PickRejected('下一抓时间估计缺少本轮冻结曝光')
        # All inputs here are immutable reference data. FK on the live planner
        # would race the first pick's collision checks and feedback sampling.
        xyz, quat = reference.sample(reference.follow_end_s)
        rotation = Rotation.from_quat(np.asarray(quat)[[1, 2, 3, 0]]).as_matrix()
        stamp = reference.start_s + reference.follow_end_s
        local = (target.points + target.velocity * (stamp - target.capture_s) - xyz) @ rotation
        brake_end = reference.start_s + path.duration
        leads = self._planning_leads(planner)
        revision = planner.scene_revision
        def estimate(joints, carry_s, return_s, source):
            self._set_window(joints, local, leads, brake_end + self.brake_to_carry + carry_s,
                             return_s, source)
        with self.lock:
            self.clear()
            self.owner, self.brake_end, self.forecast_future = path.identity, brake_end, future
            if self.previous_motion is not None:
                estimate(*self.previous_motion, 'previous_carry')

        def carry_ready(job):
            if job.cancelled() or job.exception() is not None:
                return
            result = job.result()
            with self.lock:
                if self.forecast_future is not job or self.owner != path.identity:
                    return
                bundle = result.get('bundle')
                if (not result.get('ok') or bundle is None or result['token'] != token
                        or result['identity'] != path.identity or result['source_version'] != path.version
                        or result['scene_revision'] != revision or bundle.identity != path.identity):
                    return
                if 'bin' in self.jobs and not np.allclose(self.window[0], bundle.carry.q1, rtol=0., atol=1e-8):
                    self._cancel_job('bin')
                duration = bundle.returning.duration if bundle.returning is not None else (
                    self.previous_motion[2] if self.previous_motion is not None else None)
                estimate(bundle.carry.q1, bundle.carry.duration, duration, 'predicted_carry')
                if self.target is not None:
                    self._schedule()
        if future is not None:
            future.add_done_callback(carry_ready)

    def begin(self, planner, carry, returning, carry_origin):
        with self.lock:
            same_forecast = self.owner == carry.identity and self.brake_end is not None
            if same_forecast:
                self.brake_to_carry = max(self.config.raw['command_advance_s'], carry_origin - self.brake_end)
                # A different drop anchor or a material change to the estimated
                # attachment warrants a fresh bin route using the measured cloud.
                if 'bin' in self.jobs:
                    q, predicted = self.window[:2]
                    actual = planner.payload_local
                    if (not np.allclose(q, carry.q1, rtol=0., atol=1e-8) or predicted.shape != actual.shape
                            or not len(actual) or np.max(np.linalg.norm(predicted-actual, axis=1))
                               > self.config.raw.get('workpiece_collision_margin_m', .004)):
                        self._cancel_job('bin')
            else:
                self.clear()
            self.owner = carry.identity
            self.brake_end = self.forecast_future = None
            duration = returning.duration if returning is not None else (
                self.previous_motion[2] if self.previous_motion is not None else None)
            self.previous_motion = (carry.q1.copy(), carry.duration, duration)
            self._set_window(carry.q1, planner.payload_local.copy(), self._planning_leads(planner),
                             carry_origin + carry.duration, duration, 'confirmed_carry')
            if self.target is not None:
                self._schedule()

    def return_prepared(self, owner, duration):
        with self.lock:
            if self.owner != owner or self.window is None or self.previous_motion is None:
                return
            self.previous_motion = (*self.previous_motion[:2], duration)
            self.return_lead = self.config.raw['command_advance_s'] + duration + self.margin
            if self.target is not None:
                self._schedule()

    def observe_wait(self, joints):
        # Called only after the owner's fresh, empty-gripper wait-start checks.
        # The next return may settle differently and is always checked again.
        with self.lock:
            self.wait_joints = np.asarray(joints, dtype=float).copy()

    def observe_wait_check(self, elapsed):
        if np.isfinite(elapsed) and elapsed >= 0:
            with self.lock:
                self.wait_check_times.append(elapsed)

    def wait_dispatch_lead(self, candidate=None):
        # Returning ends before the next owner installs the scene, anchors and
        # certifies the curve, then prepares the controller for dispatch.
        with self.lock:
            check = max(self.wait_check_times, default=0.)
            if candidate is not None:
                check = max(check, candidate.check_time())
            return self.return_lead + max(self.margin, check) + self.window[3]

    def observe_bin_check(self, elapsed, *, timed_out=False):
        if np.isfinite(elapsed) and elapsed >= 0:
            with self.lock:
                # An interrupted check measures only a lower bound.
                self.bin_check_times.append(elapsed + (self.margin if timed_out else 0.))

    def bin_dispatch_lead(self, candidate=None):
        with self.lock:
            check = max(self.bin_check_times, default=0.)
            if candidate is not None:
                # Worker timings exclude handing the exposure to its new owner.
                check = max(check, self.margin + candidate.check_time())
            return max(INITIAL_BIN_HANDOFF_CHECK_S, check) + self.window[3]

    def _bin_option(self, direct, status, now):
        lead = self.bin_dispatch_lead(direct)
        deadline = direct.deadline(self.config) if direct is not None else None
        slack = None if deadline is None else deadline-now-lead
        if slack is not None and slack <= 0.:
            direct, status = None, 'insufficient_bin_handoff_time'
        return direct, dict(bin_plan_deadline_s=deadline, bin_handoff_budget_s=lead,
                            bin_handoff_slack_s=slack, bin_status=status)

    def _submit(self, origin, earliest, *, release):
        submitted = self.clock()
        try:
            job = self.pool.submit(_plan, self.generations[origin].value, self.owner,
                self.worker_input, origin, self.window, earliest, submitted)
        except RuntimeError as error:
            job = Future()
            job.set_result((None, str(error)))
        self.jobs[origin] = job
        self.requested_dispatch[origin] = earliest
        if self.emit is not None:
            frozen_s = self.target.detection.get('freeze_delivery', {}).get('published_monotonic_s')
            fields = dict(previous_identity=self.owner, identity=self.target.identity, origin=origin,
                next_pick_policy=self.policy,
                schedule_source=self.schedule_source, capture_s=self.target.capture_s, submitted_s=submitted,
                frozen_s=frozen_s, estimated_release_s=release, earliest_dispatch_s=earliest,
                predicted_carry_end_s=self.carry_end)
            self.emit('next_pick_preplan_submitted', **fields)
            def finished(done):
                if done.cancelled():
                    return
                candidate, error = self._result(done)
                self.emit('next_pick_preplan_finished', **fields, error=error,
                    superseded=self.jobs.get(origin) is not done, completed_s=self.clock(),
                    ready_s=None if candidate is None else candidate.ready_s)
            job.add_done_callback(finished)

    def observe_release(self, owner):
        with self.lock:
            if self.owner == owner:
                self.release_lag = max(.15, self.clock()-self.carry_end)

    def offer(self, owner, target):
        """Retain the frozen exposure even if the first carry estimate is still pending."""
        with self.lock:
            if owner is None or owner != self.owner or owner[0] != target.epoch or target is self.target:
                return
            latest = latest_dispatch(target, self.config)
            if self.target is not None and (latest, target.capture_s) >= (
                    latest_dispatch(self.target, self.config), self.target.capture_s):
                return
            self._cancel_jobs()
            self.target = target
            self.worker_input = worker_target(target, self.config)
            if self.window is not None:
                self._schedule()

    def _schedule(self):
        release = max(self.clock() + max(self.margin, self.window[2]), self.carry_end + self.release_lag)
        latest = latest_dispatch(self.target, self.config)
        origins = ('wait',) if self.policy == WAIT_ONLY else ('wait', 'bin')
        for origin in origins:
            if origin == 'wait' and self.return_lead is None:
                continue
            job = self.jobs.get(origin)
            candidate, _ = self._result(job)
            lead = self.wait_dispatch_lead if origin == 'wait' else self.bin_dispatch_lead
            if candidate is not None and min(candidate.deadline(self.config), latest) > release + lead(candidate):
                continue
            earliest = release + lead()
            if job is not None:
                # A running solve may finish later than its requested dispatch
                # floor. Keep it if even that floor fits the updated timing.
                if not job.done() and min(self.requested_dispatch[origin], latest) >= earliest:
                    continue
                self._cancel_job(origin)
            if origin == 'bin':
                self._submit(origin, earliest + self.margin, release=release)
            elif earliest < latest:
                self._submit(origin, earliest, release=release)

    @staticmethod
    def _result(job):
        if job is None or not job.done():
            return None, 'not_ready'
        try:
            return job.result()
        except Exception as error:
            return None, str(error)

    def choose(self, owner, entry, *, active, sleep):
        """Select the handoff at release using this run's start-pose policy."""
        release_checked = now = self.clock()
        target = entry.target
        latest = latest_dispatch(target, self.config)
        with self.lock:
            if self.owner != owner:
                raise PickRejected('下一抓预规划的搬运周期已失效')
            if self.brake_end is not None or self.window is None:
                raise PickRejected('下一抓尚未确认本轮搬运轨迹')
            if target is not self.target:
                # Claiming the frozen queue is authoritative, even if a former
                # speculative target was removed between camera publications.
                self.target = None
                self.offer(owner, target)
            decision_end = min(now+self.config.raw['planning_budget_s'], latest-self.wait_dispatch_lead())
            wait_job, bin_job = self.jobs.get('wait'), self.jobs.get('bin')
            # The actual release may be earlier than predicted. Try standby
            # once if a newly opened window has never been planned.
            if wait_job is None and now < decision_end:
                self._submit('wait', now + self.wait_dispatch_lead() + min(self.margin, (decision_end-now)/2),
                             release=release_checked)
                wait_job = self.jobs['wait']
        while True:
            active()
            wait, wait_error = self._result(wait_job)
            direct, bin_error = self._result(bin_job)
            now = self.clock()
            available = now+self.wait_dispatch_lead(wait)
            wait_ready = wait is not None and min(wait.deadline(self.config), latest) > available
            if self.policy == WAIT_ONLY:
                bin_timing = dict(bin_status='disabled_by_policy')
                origin, candidate, reason = 'wait', wait if wait_ready else None, 'wait_only_return'
                break
            bin_option, bin_timing = self._bin_option(direct, bin_error, now)
            if self.policy == PREFER_BIN:
                if bin_option is not None:
                    origin, candidate, reason = 'bin', bin_option, 'certified_bin_plan_in_time'
                else:
                    origin, candidate, reason = 'wait', wait if wait_ready else None, 'bin_unavailable_return_to_wait'
                break
            if wait_ready:
                origin, candidate, reason = 'wait', wait, 'certified_wait_plan_in_time'
                break
            if direct is not None:
                # Leave polling/feedback slack at the boundary so checking
                # standby cannot consume the bin candidate's handoff budget.
                decision_end = min(decision_end, now+bin_timing['bin_handoff_slack_s']-self.margin)
            if wait_job is None or wait_error != 'not_ready' or now >= decision_end:
                origin = 'bin'
                candidate = bin_option
                if available >= latest:
                    reason = 'return_outside_window'
                elif wait is not None:
                    reason = 'wait_plan_expires_before_dispatch'
                else:
                    reason = 'no_deliverable_wait_plan'
                break
            sleep(min(.01, decision_end-now))
        return origin, candidate, dict(origin=origin, reason=reason, next_pick_policy=self.policy, identity=target.identity,
            capture_s=target.capture_s, earliest_wait_dispatch_s=available, latest_wait_dispatch_s=latest,
            return_lead_s=self.return_lead, after_return_budget_s=available-now-self.return_lead,
            wait_plan_deadline_s=wait.deadline(self.config) if wait else None,
            wait_status=wait_error, release_checked_s=release_checked, **bin_timing)

    def wait_candidate(self, owner, target, selected, *, after_return=False, decision=None):
        """Use a same-exposure wait plan that still fits at the current physical stage."""
        with self.lock:
            if self.owner != owner:
                return None
            candidate = selected
            status = None
            if candidate is None and self.target is target:
                candidate, status = self._result(self.jobs.get('wait'))
            if candidate is not None and (candidate.origin != 'wait' or candidate.previous_owner != owner
                    or candidate.identity != target.identity or candidate.plan.reference.capture_s != target.capture_s):
                candidate, status = None, 'preplan_replaced'
            lead = self.wait_dispatch_lead(candidate)
            if after_return:
                lead -= self.return_lead
            earliest = self.clock() + lead
            deadline = candidate.deadline(self.config) if candidate is not None else None
            if deadline is not None and earliest >= min(deadline, latest_dispatch(target, self.config)):
                candidate, status = None, 'insufficient_wait_handoff_time'
            if decision is not None:
                decision.update(earliest_wait_dispatch_s=earliest, wait_plan_deadline_s=deadline, wait_status=status)
            return candidate

    def confirm_choice(self, owner, entry, origin, selected, decision):
        """Recheck the chosen deadline after obtaining fresh release feedback."""
        target = entry.target
        if self.policy == PREFER_BIN:
            direct = self.bin_candidate(owner, target, selected=selected, decision=decision)
            if direct is not None:
                decision.update(origin='bin', reason='certified_bin_plan_in_time')
                return 'bin', direct
            decision['reason'] = 'bin_unavailable_return_to_wait'
        if self.policy != PREFER_WAIT:
            decision['origin'] = 'wait'
            wait = self.wait_candidate(owner, target, selected if origin == 'wait' else None, decision=decision)
            return 'wait', wait
        if origin == 'wait':
            earliest = self.clock() + self.wait_dispatch_lead(selected)
            decision['earliest_wait_dispatch_s'] = earliest
            if earliest < min(selected.deadline(self.config), decision['latest_wait_dispatch_s']):
                return 'wait', selected
            decision.update(origin='bin', reason='wait_window_expired_before_return')
            selected = None
        return 'bin', self.bin_candidate(owner, target, selected=selected, decision=decision)

    def bin_candidate(self, owner, target, *, selected=None, decision=None):
        with self.lock:
            # The frozen entry has been claimed. A new camera offer may
            # replace speculative jobs, but cannot change the claimed input.
            if (self.owner == owner and selected is not None and selected.origin == 'bin'
                    and selected.previous_owner == owner and selected.identity == target.identity
                    and selected.plan.reference.capture_s == target.capture_s):
                direct, status = selected, None
            elif self.owner == owner and self.target is target:
                direct, status = self._result(self.jobs.get('bin'))
            else:
                if decision is not None:
                    decision.update(bin_status='preplan_replaced', bin_plan_deadline_s=None, bin_handoff_slack_s=None)
                return None
            candidate, timing = self._bin_option(direct, status, self.clock())
            if decision is not None:
                decision.update(timing)
            return candidate


def anchor_curve(path, delta, transition_end):
    """Fade measured joint offset to zero with C2 continuity before synchronization."""
    curve = path.curve
    knots = curve.x
    end = knots[knots <= transition_end][-1]
    if end <= 0. or curve.c.shape[0] != 6:
        raise PickRejected('预计算曲线缺少可复核的接近段')
    zero = np.zeros_like(delta)
    offset = BPoly.from_derivatives([0., end], [[delta, zero, zero], [zero, zero, zero]])
    coefficients = curve.c.copy()
    before = knots[:-1] < end
    for order in range(6):
        coefficients[5-order, before] += offset(knots[:-1][before], order) / factorial(order)
    return replace(path, curve=PPoly(coefficients, knots, extrapolate=False),
        diagnostics={**path.diagnostics, 'measured_start_adjustment_rad':delta.tolist()})


def adopt(planner, candidate, handoff, target, state, *, cancelled):
    """Reinstall the actual scene and certify the curve before any motion."""
    plan = candidate.plan
    if (candidate.previous_owner != handoff.previous_identity or candidate.origin != handoff.origin
            or handoff.entry.target is not target or target.identity != candidate.identity
            or target.identity != plan.path.identity or target.capture_s != plan.reference.capture_s):
        raise PickRejected('预计算曲线与本次交接或曝光不一致')
    from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
    delta = np.asarray(state['joints']) - plan.path.q0
    if not np.isfinite(delta).all() or np.max(abs(delta)) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
        raise PickRejected('实测起点与预计算曲线起点不一致')
    if candidate.origin == 'bin' and (handoff.released_tree is None or
            planner.bin_departure_tree is not handoff.released_tree):
        raise PickRejected('缺少本次实际放料场景')
    deadline = candidate.deadline(planner.config)
    def check_cancelled():
        if cancelled():
            return True
        if planner.clock() >= deadline:
            raise PickRejected('交接场景复核已错过发送窗口',
                               details={'reason':'preplan_send_deadline', 'send_deadline_s':deadline})
        return False
    if check_cancelled():
        raise PickRejected('交接场景复核已取消')
    planner.install(target, state)
    planner.model.check_feedback(state)
    tcp = planner.model.tcp().copy()
    ref = replace(plan.reference, start_tcp_world=tcp[:3,3],
        start_quaternion_wxyz=Rotation.from_matrix(tcp[:3,:3]).as_quat()[[3,0,1,2]])
    transition = plan.path.diagnostics.get('bin_joint_transit_s', ref.prealign_s)
    if getattr(ref,'relative_descent_s',0.):
        transition = min(transition,ref.relative_alignment_s)
    path = anchor_curve(replace(plan.path, version=planner.version), delta, transition)
    from .hover_math import require_limits
    bounds = require_limits(path.curve, planner.config, planner.model)
    for name, key in [('velocity','peak_velocity_rad_s'), ('acceleration','peak_acceleration_rad_s2')]:
        path.diagnostics[key] = np.maximum(abs(bounds[name]['min']), abs(bounds[name]['max'])).tolist()
    path.diagnostics['reference_check'] = planner.check_smooth_intercept(path, ref, cancelled=check_cancelled)
    if check_cancelled():
        raise PickRejected('交接场景复核已取消')
    planner.contact_reference = ref
    return replace(plan, path=path, target=target, reference=ref, scene_revision=planner.scene_revision)
