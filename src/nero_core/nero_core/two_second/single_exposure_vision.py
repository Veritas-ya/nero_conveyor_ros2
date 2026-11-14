"""Measured hardware vision: replace target acquisition, reuse the pick owner.

The locator/geometry are shared with camera_fixed_freeze_test. Robot self
filtering uses actual capture-paired feedback. This module sends no commands;
the existing cycle executor owns planning, dispatch, taught paths and recovery.
"""
from collections import deque
from copy import deepcopy
from threading import Thread
from types import SimpleNamespace
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface,BeltSurfaceNotReady
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected,TargetNotReady
from .fixed_freeze import ExposureBuffer,NativeDepthLocator,frozen_geometry
from .grasp import VisibleOBBError
from .measured_rgbd import depth_capture_timing,fit_plane,green_belt_points
from .reference import EntryNotReady
from .velocity_source import is_fixed,prediction_velocity,speed_report
from .vision import LiveVision,MeasuredTarget


class SingleExposureVision(LiveVision):
    def __init__(self,camera,calibration,config,model,read_feedback):
        if (not is_fixed(config) or config.raw.get('intercept_strategy')!='synchronized'
                or config.raw.get('freeze_position_upstream_m') is None):
            raise ValueError('单帧冻结真机入口需要固定带速、固定冻结位置和现有同步抓取策略')
        super().__init__(camera,calibration,config,model,read_feedback)
        self.buffer=ExposureBuffer(config.raw.get('single_exposure_buffer_frames',16))
        self.pending=deque(maxlen=3)
        self.reader_thread=Thread(target=self._receive,name='single-exposure-raw-camera')
        self.locator=None;self.locator_report={};self.d2c=None

    def start(self):
        # Factory D2C requires the actual first frame, just like the viewer.
        self.camera.read_latest(timeout_s=3.,wait_for_fresh=True)
        self.d2c=np.asarray(self.camera.get_depth_to_color_cv()).copy()
        self.geometry_future=self.geometry_pool.submit(self._warm_geometry_model)
        self.reader_thread.start();self.thread.start()
        return self

    def close(self):
        self.stop.set();self.buffer.close()
        super().close()
        if self.reader_thread.ident is not None:self.reader_thread.join(timeout=2.)
        if self.reader_thread.is_alive():raise RuntimeError('单帧冻结相机接收线程尚未退出')

    def consume(self,identity):
        super().consume(identity)
        with self.lock:self.pending.clear()

    def resume_targets(self):
        # Same production rule: only the existing confirmed-return owner calls
        # this, and only exposures newer than that call may start another pick.
        super().resume_targets()
        if not self.lookahead_enabled:
            with self.lock:self.pending.clear()

    def diagnostics(self):
        result=super().diagnostics()
        with self.lock:pending=len(self.pending);locator=dict(self.locator_report)
        result.update(freeze_source='single_exposure_native_depth',
            multiframe_matching_used=False,raw_frame_buffer=self.buffer.diagnostics(),
            single_frame_locator=locator,pending_exposures=pending,
            capture_state_source='actual_robot_feedback_with_original_200ms_pairing_limit')
        return result

    def _receive(self):
        sequence=None
        try:
            while not self.stop.is_set():
                frame=self.camera.read_next(timeout_s=.5,after_sequence=sequence,wait_for_fresh=True)
                sequence=frame.header.sequence
                previous=self._previous_camera_frame
                self.camera_consumption['frames']+=1
                if previous is not None and previous.header.clock_epoch==frame.header.clock_epoch:
                    self.camera_consumption['capture_sequence_skips']+=max(0,sequence-previous.header.sequence-1)
                    self.camera_consumption['max_device_gap_s']=max(self.camera_consumption['max_device_gap_s'],
                        frame.header.source_time-previous.header.source_time)
                self._previous_camera_frame=frame
                with self.lock:
                    if self.clock_epoch is not None and self.clock_epoch!=frame.header.clock_epoch:
                        self.geometry_generation+=1;self.latest=None;self.pending.clear()
                    # Update immediately even during execution. The unchanged
                    # executor observes this epoch and cancels an invalid plan.
                    self.clock_epoch=frame.header.clock_epoch
                self.buffer.put(frame)
        except BaseException as error:
            if not self.stop.is_set():
                with self.lock:self.error=error
        finally:self.buffer.close()

    def _submit_pending(self):
        if self.geometry_future is not None:return
        with self.lock:
            if (self.active_task is not None and not self.lookahead_enabled) or self.latest is not None:return
            while self.pending:
                item=self.pending.popleft()
                frame,_,_,_,capture,generation=item
                if (generation==self.geometry_generation and frame.header.clock_epoch==self.clock_epoch
                        and capture>=self.minimum_capture_s
                        and 0<=time.monotonic()-capture<=self.config.raw['maximum_observation_age_s']):
                    self.geometry_future=self.geometry_pool.submit(self._prepare_single,*item)
                    return

    def _prepare_single(self,frame,surface,candidate,state,captured,generation):
        progress=self.geometry_progress
        progress.frame(sequence=frame.header.sequence,clock_epoch=frame.header.clock_epoch,
            capture_monotonic_s=captured,acquisition='single_exposure_native_depth')
        with self.lock:
            if self.stop.is_set() or generation!=self.geometry_generation:return
        self._warm_geometry_model()
        progress.enter('detecting_target')
        try:
            with progress.measure('single_exposure_geometry'):
                detection,center,velocity,grasps,overlap=frozen_geometry(self.config,frame,self.d2c,
                    surface,candidate,captured,clock=time.monotonic,
                    robot_model=self.geometry_model,robot_state=state)
        except (TargetNotReady,VisibleOBBError,EntryNotReady) as error:
            progress.block('waiting_target_geometry',getattr(error,'reason',None) or
                getattr(error,'details',{}).get('reason','single_frame_geometry_rejected'),
                error=str(error),details=getattr(error,'details',{}))
            return
        except PickRejected as error:
            if 'seed_overlap_ratio' not in error.details:raise
            progress.block('waiting_target_geometry','same_exposure_component_mismatch',
                error=str(error),details=error.details)
            return
        with self.lock:
            if self.stop.is_set() or generation!=self.geometry_generation:return
        # This is an exposure identity, never a fabricated confirmed track.
        observation=SimpleNamespace(sample_count=1,observation_source='single_exposure_native_depth',
            identity_source='capture_epoch_generation_sequence')
        report=speed_report(self.config,observation,velocity)
        report.update(acquisition='single_exposure_native_depth',seed_overlap_ratio=overlap,
                      capture_state_source='measured_capture_paired_feedback')
        item=MeasuredTarget((frame.header.clock_epoch,generation,frame.header.sequence),captured,
            frame.header.received_time,frame.header.source_time,frame.header.clock_epoch,
            frame.header.sequence,center,velocity,grasps,np.asarray(detection['object_points']),
            detection,state,report,entry_position=center)
        progress.annotate_frame(point_count=len(item.points),seed_overlap_ratio=overlap,
            local_processing=detection.get('local_processing'),
            freeze_capture_upstream_m=float(self.config.follow_bounds[0]-center@self.config.direction))
        progress.mark_ready()
        return generation,item

    def _finish_geometry(self):
        future=self.geometry_future
        if future is None or not future.done():return
        self.geometry_future=None
        result=future.result()
        if result is None:return  # model warm-up or a rejected exposure
        generation,item=result
        if not 0<=time.monotonic()-item.capture_s<=self.config.raw['maximum_observation_age_s']:
            self.geometry_progress.block('waiting_fresh_observation','observation_stale_at_delivery',
                source_sequence=item.sequence)
            return
        with self.lock:
            if (self.stop.is_set() or generation!=self.geometry_generation or self.clock_epoch!=item.epoch
                    or (self.active_task is not None and not self.lookahead_enabled) or item.identity in self.consumed
                    or item.capture_s<self.minimum_capture_s):return
            self.latest=item;self.pending.clear()
        self.progress.mark_ready()

    def _paired_state(self,captured):
        self.progress.enter('pairing_feedback')
        with self.progress.measure('feedback_pairing'):state=self.capture_feedback(captured)
        if state is None:
            self.progress.block('waiting_feedback','feedback_missing');return None
        stamps=[mapped_stamp(state,kind) for kind in ('joint','pose')]
        if any(stamp is None or abs(stamp-captured)>.2 for stamp in stamps):
            self.progress.block('waiting_feedback',
                'feedback_time_missing' if any(s is None for s in stamps) else 'feedback_frame_skew',
                capture_monotonic_s=captured,feedback_monotonic_s=stamps,maximum_skew_s=.2)
            return None
        return state

    def _process_frame(self,raw):
        sink=getattr(self,'frame_sink',None)
        if sink is not None:sink(raw,self.d2c)
        self.progress.frame(sequence=raw.header.sequence,clock_epoch=raw.header.clock_epoch,
            source_device_s=raw.header.source_time,received_monotonic_s=raw.header.received_time)
        with self.lock:
            if raw.header.clock_epoch!=self.clock_epoch:return
            if (self.active_task is not None and not self.lookahead_enabled) or self.latest is not None:
                self.progress.enter('task_active_camera_only');return
        frame=apply_runtime_depth_calibration(raw,self.calibration,self.d2c)
        try:captured=depth_capture_timing(frame)['depth_sample_monotonic_estimate']
        except PickRejected as error:
            with self.lock:
                self.geometry_generation+=1;self.latest=None;self.pending.clear()
            self.progress.block('waiting_valid_timestamp','invalid_capture_time',
                error=str(error),timestamp_details=error.details)
            return
        with self.lock:
            if captured<self.minimum_capture_s:return
        if not 0<=time.monotonic()-captured<=self.config.raw['maximum_observation_age_s']:
            self.progress.block('waiting_fresh_observation','observation_stale_before_detection');return
        surface=getattr(self.model,'measured_belt_surface',None)
        if not isinstance(surface,BeltSurface) or not surface.matches_frame(frame,self.d2c):
            self.progress.enter('fitting_belt_surface')
            try:surface=BeltSurface.from_frame(frame,self.d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
            except BeltSurfaceNotReady as error:
                self.progress.block('fitting_belt_surface','belt_surface_not_ready',error=str(error));return
            self.model.measured_belt_surface=surface
            self.locator=NativeDepthLocator(self.config,surface,frame)
        if self.locator is None:self.locator=NativeDepthLocator(self.config,surface,frame)
        state=self._paired_state(captured)
        if state is None:return
        self.progress.enter('single_frame_depth_location')
        with self.progress.measure('single_frame_depth_location'):
            candidates,report=self.locator.locate(frame,robot_model=self.model,robot_state=state)
        with self.lock:self.locator_report={**report,'frame_sequence':frame.header.sequence}
        if len(candidates)!=1:
            self.progress.block('waiting_target_geometry','no_single_depth_component',
                candidate_count=len(candidates),locator=report)
            return
        candidate=candidates[0]
        if self.lookahead_enabled and self.repeats_recent_target(candidate.position, captured):
            self.progress.block('waiting_new_target', 'same_physical_workpiece')
            return
        velocity=prediction_velocity(self.config,None,surface.normal_base)
        try:self.entry_time(captured,candidate.position,velocity,time.monotonic())
        except EntryNotReady:return
        with self.lock:
            if (self.active_task is not None and not self.lookahead_enabled) or self.latest is not None:return
            self.pending.append((frame,surface,candidate,deepcopy(state),captured,self.geometry_generation))
        self._submit_pending()

    def _run(self):
        try:
            while not self.stop.is_set():
                self._finish_geometry();self._submit_pending()
                raw=self.buffer.take(timeout=.1)
                if raw is None:
                    with self.lock:error=self.error
                    if error is not None:raise error
                    continue
                self._process_frame(raw)
        except BaseException as error:
            if not self.stop.is_set():
                self.progress.block('vision_error','vision_exception',error_type=type(error).__name__,
                    error=str(error),details=getattr(error,'details',{}))
                with self.lock:self.error=error
        finally:
            self.stop.set();self.buffer.close()
            if self.reader_thread.ident is not None:self.reader_thread.join(timeout=2.)
            self.geometry_pool.shutdown(wait=True,cancel_futures=True)
            self.geometry_progress.finish();self.progress.finish()
