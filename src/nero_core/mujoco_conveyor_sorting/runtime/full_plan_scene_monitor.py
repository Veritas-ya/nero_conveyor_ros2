"""Read-only obstacle review from the existing camera's cached frames.

One private model, latest-frame processing, no camera ownership or robot IO.
The control thread only consumes an immutable status. Observed surfaces cannot
certify occluded space. New obstacles invalidate, never edit, an active plan.
"""
from collections import deque
from threading import Event,Lock,Thread
import time
from .feedback_clock import mapped_stamp

import numpy as np
from scipy.spatial import cKDTree

from ..perception import deproject_depth_to_world
from .nero_policy_context import _copy_geometry
from .scene_region_filter import filter_scene_region
from .scene_noise_filter import filter_scene_noise
from .single_pick import PickRejected


class FullPlanSceneMonitor:
    period = .2  # Existing 5 Hz review cadence, shared with the splice forecast.
    def __init__(self, context, *, read_frame, calibrate_frame, capture_monotonic,
                 belt_region, reference_frame=None, clock=time.monotonic):
        self.context=context
        self.geometry=_copy_geometry(context.geometry)
        self.reference=cKDTree(self.geometry.scene)
        self.read_frame,self.calibrate_frame=read_frame,calibrate_frame
        self.capture_monotonic,self.clock=capture_monotonic,clock
        self.region,self.plane=self.geometry.belt_scene_filter_config.resolve_region(belt_region)
        self.lock=Lock();self.stop_event=Event();self.samples=deque(maxlen=256)
        self.status={'enabled':True,'failed':False,'validated_capture_monotonic_s':None,
            'frames_checked':0,'reason':'waiting_for_paired_frame',
            'scope':'new_visible_points_in_original_ROI_against_remaining_certified_path',
            'surface_change_distance_m':.004,'pair_max_separation_s':.04}
        self.thread=None
        self.frame_binding=None if reference_frame is None else self._frame_binding(reference_frame)

    @staticmethod
    def _frame_binding(frame):
        return (frame.serial_number, frame.calibration_id, frame.header.clock_epoch,
            tuple(frame.depth_m.shape), np.asarray(frame.depth_intrinsics.matrix).tobytes(),
            np.asarray(frame.camera_to_world_cv).tobytes())

    def publish_state(self,state,policy_time):
        # Store raw received state and its real packet clock separately. This
        # offset is a host clock conversion, not a new encoder sample time.
        read=self.clock()
        stamp=mapped_stamp(state)
        if stamp is None or 'feedback_time' not in state:
            return  # no receive-boundary mapping: never invent a camera pairing
        q=np.array(state['joints'],dtype=float,copy=True)
        width=float(state['width_m'])
        with self.lock:self.samples.append((stamp,q,width,float(policy_time),read))

    def snapshot(self):
        with self.lock:return dict(self.status)

    def check_points(self,points,bundle,policy_time,*,payload_local=None):
        """Original 4 mm representation error separates existing/new points.

        Checking the whole remaining original sample grid is conservative: a
        currently visible obstacle may move away later, but is never assumed
        to do so. No planner is invoked on an unchanged scene.
        """
        points=np.asarray(points,dtype=float)
        if not len(points):return None
        distance,_=self.reference.query(points)
        novel=points[distance>.004]
        if not len(novel):return None
        geometry=self.geometry
        geometry.scene=novel.copy();geometry.tree=cKDTree(geometry.scene)
        geometry.target=np.empty((0,3))
        try:
            for phase,stamp,q in (*bundle.checked_samples, *getattr(bundle,'checked_command_samples',())):
                if stamp<policy_time:continue
                widths=(np.linspace(0.,.1,11) if phase=='closing' else
                        (bundle.request.full_grasp_width_m,) if phase=='lift' else (.1,))
                for width in widths:
                    geometry.check_state(q,float(width),allow_finger_contact=False)
                if phase=='lift' and payload_local is not None:
                    tcp=geometry.tcp()
                    payload=np.asarray(payload_local)@tcp[:3,:3].T+tcp[:3,3]
                    if len(payload) and np.any(geometry.tree.query(payload)[0]<.008):
                        return 'new_obstacle_in_nominal_payload_lift'
        except PickRejected as exc:
            with self.lock:self.status['obstacle_details']={'message': str(exc), 'details': exc.details,
                'future_phase': phase, 'future_sample_s': float(stamp), 'new_point_count': len(novel)}
            return 'new_obstacle_on_remaining_trajectory'
        return None

    def process_frame(self,raw):
        capture=float(self.capture_monotonic(raw))
        with self.lock:samples=tuple(self.samples)
        if not samples:return
        paired=min(samples,key=lambda s:abs(s[0]-capture))
        if abs(paired[0]-capture)>.04:
            with self.lock:self.status['reason']='no_joint_packet_within_40ms_of_depth_capture'
            return
        frame=self.calibrate_frame(raw)
        binding=self._frame_binding(frame)
        if self.frame_binding is not None and binding!=self.frame_binding:
            with self.lock:self.status.update(failed=True,failure_reason='scene_camera_binding_changed')
            return
        self.frame_binding=binding
        points=deproject_depth_to_world(frame.depth_m,frame.depth_intrinsics.matrix,
            frame.camera_to_world_cv,camera_convention='opencv').reshape(-1,3)
        points=points[np.isfinite(points).all(axis=1)&(np.linalg.norm(points,axis=1)<1.5)]
        points=filter_scene_region(points,self.plane,self.region).retained_points
        state={'joints':paired[1],'width_m':paired[2]}
        filtered=self.geometry.classify_robot_points(points,state,
            camera_origin=frame.camera_to_world_cv[:3,3])
        points=points[~filtered.removed_mask]
        points=filter_scene_noise(points,self.geometry.scene_noise_filter_config).retained_points
        bundle=self.context.policy._full_pick_bundle
        reason=None
        pending_reason=None
        pending=getattr(self.context.policy,'_full_pick_pending_bundle',None)
        checked_ids=[]
        if bundle is not None:
            # Remove only measured selected-object surface inside the EXISTING
            # 4 mm object margin; no broad ROI masks other obstacles away.
            motion=self.context._motion_snapshot
            payload=getattr(self.context, '_payload_local', None)
            if self.context.policy._full_pick_lift_admitted and payload is not None:
                self.geometry.set_state(paired[1], paired[2])
                pose=self.geometry.tcp().copy()
                target=payload@pose[:3,:3].T+pose[:3,3]
            elif motion is not None:
                pose=np.array(motion.observed_grasp,copy=True)
                velocity=self.context._motion_velocity()
                pose[:3,3]+=velocity*(capture-float(motion.observed_timestamp))
                target=self.context._template_local@pose[:3,:3].T+pose[:3,3]
            else:
                target=np.empty((0,3))
            if len(target):points=points[cKDTree(target).query(points)[0]>.004]
            origin=self.context._policy_clock_origin
            policy_time=capture-float(origin)
            reason=self.check_points(points,bundle,policy_time,
                payload_local=self.context._template_local if payload is None else payload)
            checked_ids.append(id(bundle))
            if pending is not None:
                pending_reason=self.check_points(points,pending,policy_time,
                    payload_local=self.context._template_local if payload is None else payload)
                if pending_reason is None:checked_ids.append(id(pending))
        with self.lock:
            self.status.update(validated_capture_monotonic_s=capture,
                frame_sequence=int(frame.header.sequence),frames_checked=self.status['frames_checked']+1,
                last_compute_end_monotonic_s=self.clock(),reason=reason or 'ok',
                checked_plan_identity=None if bundle is None else id(bundle),
                checked_plan_identities=checked_ids,pending_plan_identity=None if pending is None else id(pending),
                pending_failure_reason=pending_reason)
            if reason:self.status.update(failed=True,failure_reason=reason)

    def start(self):
        if self.thread is not None:raise RuntimeError('scene monitor already started')
        def run():
            last=None
            while not self.stop_event.is_set():
                begin=self.clock()
                try:
                    raw=self.read_frame(last)
                    if raw is not None and raw.header.sequence!=last:
                        self.process_frame(raw);last=raw.header.sequence
                except Exception as exc:
                    with self.lock:self.status['reason']=f'{type(exc).__name__}: {exc}'
                self.stop_event.wait(max(0.,self.period-(self.clock()-begin)))
        self.thread=Thread(target=run,name='full-pick-visible-obstacles',daemon=True)
        self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join(timeout=3.)
