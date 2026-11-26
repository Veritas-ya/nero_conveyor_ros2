"""Native-depth backup for the tracked capture window, using the same camera exposure."""
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from itertools import count
from threading import Lock, Timer
from types import SimpleNamespace
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from .capture_feedback_worker import CaptureFeedbackWorker
from .diagnostics import VisionProgress
from .fixed_freeze import NativeDepthLocator, frozen_geometry
from .grasp import VisibleOBBError
from .reference import EntryNotReady
from .tracked_freeze import detector_freeze_window, native_exposure_window, pairing_deadline, upstream_at
from .velocity_source import prediction_velocity, speed_report


def use_single_extraction(config):
    r = config.raw
    return (native_exposure_window(config)
            and r.get('native_single_geometry_mode') == 'foreground_once'
            and r.get('environment_collision_scope') == 'measured_belt_surface_only'
            and not r.get('rgb_cloud_completion_enabled',False))


class SingleFrameBackup:
    def __init__(self, vision):
        self.vision = vision
        raw = vision.config.raw
        opening = raw['tracked_freeze_upstream_limit_m']
        self.exposure_gate = native_exposure_window(vision.config)
        self.cutoff = (raw['native_single_min_capture_upstream_m'] if self.exposure_gate
                       else raw['tracked_freeze_publish_before_upstream_m'])
        self.capture_band = self.cutoff, opening
        self.config = vision.config
        self.pool = ThreadPoolExecutor(1, thread_name_prefix='single-frame-freeze')
        self.model = self.future = self.surface = self.locator = None
        self.exposure_ids = count(1)
        self.last_rejection = None
        self.progress = VisionProgress()
        self.feedback_worker = CaptureFeedbackWorker(vision, self.pool, self.progress, lambda: time.monotonic())
        self.warming = None
        self.stereo = SimpleNamespace(refine=vision._refine_stereo)
        self.timer_lock = Lock()
        self.timers = {}

    def _warm(self):
        from .planning import calibrated_model
        self.model = calibrated_model(self.config)

    def start(self):
        self.future = self.warming = self.pool.submit(self._warm)

    def close(self):
        self.feedback_worker.close()
        self.pool.shutdown(wait=True, cancel_futures=True)
        with self.timer_lock:
            for timer in self.timers.values():
                timer.cancel()
            self.timers.clear()

    def offer(self, frame, d2c, surface, captured):
        with self.vision.lock:
            if self.vision.frozen_queue.full or self.vision.stop.is_set():
                return None
            generation = self.vision.geometry_generation
        decision = None if self.vision.native_band_only else Future()
        job = self._submit(frame, d2c, surface, captured, generation, decision)
        return decision if job is not None else None

    def _submit(self, frame, d2c, surface, captured, generation, decision):
        if self.warming is not None:
            if not self.warming.done():
                return None
            self.warming.result()
            self.warming = None
        if self.surface is not surface:
            try:
                distance = detector_freeze_window(self.config, surface)['distance_m']
                if use_single_extraction(self.config):
                    from .native_single_pass import SinglePassLocator
                    locator = SinglePassLocator(self.config,surface,frame,d2c,capture_band_m=self.capture_band)
                else:
                    locator = NativeDepthLocator(self.config, surface, frame, capture_band_m=self.capture_band)
            except (ValueError, EntryNotReady) as error:
                self.last_rejection = str(error)
                return None
            self.max_distance, self.locator, self.surface = distance, locator, surface
        # Bind camera rays and the measured envelope to this exposure. A later
        # surface change cannot replace either input of a retained job.
        locator, max_distance = self.locator, self.max_distance
        deadline = pairing_deadline(self.config, captured, surface, max_distance)
        d2c = np.array(d2c, copy=True)

        def calculate(state, timing):
            self._calculate(frame, d2c, surface, captured, generation, decision, state, timing, locator, max_distance)
        job = self.feedback_worker.submit(captured, frame.header.clock_epoch, generation, deadline, calculate)
        if job is not None:
            self.future = job
        return job

    def _matching_job(self, decision, target):
        match = decision.result()
        if match is None:
            return None
        captured, position, gate, job = match
        predicted = position + target.velocity * (target.capture_s - captured)
        if np.linalg.norm(predicted[:2] - target.center[:2]) <= gate:
            return job
        return None

    def _prefer_match(self, generation, target, decision, deadline):
        if decision.done():
            job = self._matching_job(decision, target)
            if job is None or job.done():
                self._publish(generation, target)
                return

        # A finished backup must not hold up localization of the next piece.
        # Wake before the line to allow for scheduling; commit rechecks real time.
        key = target.identity

        def release(_=None):
            with self.timer_lock:
                pending = self.timers.pop(key, None)
                if pending is None:
                    return
                pending.cancel()
            self._publish(generation, target)

        def matched(future):
            job = self._matching_job(future, target)
            if job is None:
                release()
            else:
                job.add_done_callback(release)

        timer = Timer(max(0., deadline-time.monotonic()-.020), release)
        timer.daemon = True
        with self.timer_lock:
            self.timers[key] = timer
        timer.start()
        decision.add_done_callback(matched)

    def _publish(self, generation, target):
        self.vision._finish_geometry()  # a finished, qualified match has priority
        with self.vision.lock:
            publication = self.vision._accept_geometry_locked(generation, None, target)
        self.vision._notify_frozen(target, publication)

    def process(self, frame, d2c, surface, captured, generation, decision):
        """Synchronous replay entry using the same admission and feedback wait as live acquisition."""
        job = self._submit(frame, d2c, surface, captured, generation, decision)
        if job is not None:
            job.result()

    def _calculate(self, frame, d2c, surface, captured, generation, decision, state, timing, locator, max_distance):
        self.progress.frame(sequence=frame.header.sequence, clock_epoch=frame.header.clock_epoch,
                            capture_monotonic_s=captured, **timing)
        state = deepcopy(state)
        located_at = time.monotonic()
        candidates, locator_report = locator.locate(frame, robot_model=self.model, robot_state=state)
        locate_wall_s = time.monotonic()-located_at
        if not candidates:
            if self.exposure_gate:
                self.progress.block('single_frame_location','no_eligible_component',
                    size_rejections=locator_report.get('component_extraction',{}).get('size_rejection_count',0),
                    component_rejections=locator_report.get('component_rejections',[]))
            return
        cfg = self.config
        velocity = prediction_velocity(cfg, None, surface.normal_base)
        along = float(velocity @ cfg.direction)
        candidates.sort(key=lambda candidate: -float(candidate.position @ cfg.direction))
        for candidate in candidates:
            start = time.monotonic()
            distance = upstream_at(cfg, candidate.position, velocity, captured, captured)
            # A rough seed may straddle 75 cm. Only its subsequently measured
            # complete cloud centre can authorize the original exposure.
            near = (cfg.follow_bounds[0] - candidate.along_bounds_m[1]
                    if candidate.along_bounds_m is not None else distance)
            if distance < self.cutoff or near > max_distance:
                if self.exposure_gate:
                    self.progress.block('single_frame_location','seed_outside_exposure_band',upstream_m=distance)
                continue
            deadline = captured + cfg.raw['maximum_observation_age_s']
            if not self.exposure_gate:
                deadline = min(deadline, captured + (distance-self.cutoff)/along)
            y0, y1 = cfg.raw['perception_detector_workspace_y_m']
            if start > deadline or not y0 <= candidate.position[1] <= y1:
                continue
            with self.vision.lock:
                if self.vision.stop.is_set() or generation != self.vision.geometry_generation:
                    return
                if self.vision._repeats_locked(candidate.position, captured, frame_sequence=frame.header.sequence,
                                               seed_indices=candidate.seed_pixel_indices):
                    continue
            try:
                from .vision import MeasuredTarget
                detection, center, velocity, grasps, overlap = frozen_geometry(cfg, frame, d2c, surface,
                    candidate, captured, robot_model=self.model, robot_state=state,
                    fast_stereo=self.stereo if self.vision.fast_stereo is not None else None)
            except (PickRejected, VisibleOBBError, EntryNotReady) as error:
                self.last_rejection = str(error)
                if self.exposure_gate:
                    details = getattr(error, 'details', {})
                    reason = details.get('reason') or getattr(error,'reason',None) or 'geometry_rejected'
                    evidence = {k:details[k] for k in ('margin_m','minimum_clearance_m','points_in_strip',
                                'total_points','frame_sequence') if k in details}
                    quality = (details.get('fast_stereo') or {}).get('quality_check')
                    if quality:
                        evidence['fast_quality'] = {k:quality.get(k) for k in (
                            'reasons','retained_points','candidate_retained_fraction',
                            'native_coverage_fraction','center_shift_m','limits')}
                        if (quality.get('enforced_as_delivery_gate', True) and not quality['accepted']
                                and getattr(self.vision, 'event_sink', None) is not None):
                            self.vision.event_sink('fast_quality_rejected', frame_sequence=frame.header.sequence,
                                capture_monotonic_s=captured, upstream_m=distance, **evidence)
                    self.progress.block('single_frame_geometry',reason,
                                        upstream_m=distance,error=str(error),**evidence)
                continue
            measured_distance = upstream_at(cfg, center, velocity, captured, captured)
            if not self.cutoff <= measured_distance <= max_distance or not y0 <= center[1] <= y1:
                if self.exposure_gate:
                    self.progress.block('single_frame_geometry','measured_centre_outside_band_or_y',
                                        upstream_m=measured_distance,base_y_m=float(center[1]))
                continue
            verify_seed = candidate.prepared is None or self.vision.fast_stereo is not None
            if verify_seed and any(other is not candidate and 2*np.count_nonzero(detection['mask'].ravel()[other.seed_pixel_indices])
                                   >= len(other.seed_pixel_indices) for other in candidates):
                self.last_rejection = 'single_frame_component_ambiguous'
                continue
            deadline = min(deadline, captured + (measured_distance-self.cutoff)/along)
            observation = SimpleNamespace(sample_count=1, observation_source='single_exposure_native_depth',
                                          identity_source='native_depth_exposure_component')
            report = speed_report(cfg, observation, velocity)
            direct = self.vision.native_band_only
            report.update(acquisition=('single_exposure_native_depth' if direct else 'single_exposure_native_depth_fallback'),
                          geometry_pipeline=('foreground_single_extraction' if candidate.prepared is not None else 'seed_then_detector'),
                          seed_overlap_ratio=overlap, internal_tracking_kalman_still_runs=not direct,
                          tracking_used_for_this_freeze=False)
            finished = time.monotonic()
            geometry_start = located_at if candidate.prepared is not None else start
            detection['freeze_delivery'] = dict(capture_monotonic_s=captured, geometry_started_monotonic_s=geometry_start,
                geometry_done_monotonic_s=finished, geometry_wall_s=finished-geometry_start,
                frame_location_wall_s=locate_wall_s, candidate_finish_wall_s=finished-start,
                geometry_pipeline=report['geometry_pipeline'], **timing)
            cloud = detection.get('fast_stereo',{}).get('cloud_delivery')
            if cloud is not None:
                detection['freeze_delivery']['fast_cloud_id'] = cloud['cloud_id']
            # Real tracker generations are nonnegative. Each distinct native
            # component gets a unique identity, including two in the same frame.
            identity = (frame.header.clock_epoch, -1, next(self.exposure_ids))
            target = MeasuredTarget(identity, captured, frame.header.received_time, frame.header.source_time,
                frame.header.clock_epoch, frame.header.sequence, center, velocity, grasps,
                detection['object_points'], detection, state, report, entry_position=center)
            if direct:
                self._publish(generation, target)
            else:
                self._prefer_match(generation, target, decision, deadline)
