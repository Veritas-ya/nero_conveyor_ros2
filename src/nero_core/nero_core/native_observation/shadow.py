"""Observe the existing RGB-D stream. This consumer has no sender or planning-queue handle."""
from collections import Counter
from dataclasses import replace
from queue import Empty, Full, Queue
from threading import Event, Thread
from pathlib import Path
import json
import time

import numpy as np

from mujoco_conveyor_sorting.runtime.feedback_pairing import capture_joint_pairing_valid
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface, BeltSurfaceNotReady
from nero_core.two_second.measured_rgbd import depth_capture_timing, fit_plane, green_belt_points
from nero_core.two_second.velocity_source import prediction_velocity, is_fixed
from .geometry import NativeGeometry, Parameters, Region
from .association import MotionAssociation


class NativeShadow:
    def __init__(self, config, calibration, output, feedback_reader, model, *, clock=time.monotonic):
        if not is_fixed(config):
            raise ValueError('native observation replay requires a configured conveyor speed')
        self.config, self.calibration = config, calibration
        self.parameters = Parameters(**config.raw.get('native_observation', {}))
        self.directory = Path(output)/'native_observations'
        self.directory.mkdir(parents=True,exist_ok=False)
        self.feedback_reader,self.model = feedback_reader,model
        self.clock=clock
        self.queue=Queue(maxsize=1)
        self.stop=Event()
        self.thread=Thread(target=self._run,name='read-only-native-observations')
        self.detector=None
        self.generation=0
        self.pending={}
        self.counts=Counter()
        self.dropped=0

    def offer(self, frame, d2c):
        packet=frame,d2c
        try:
            self.queue.put_nowait(packet)
        except Full:
            try:
                self.queue.get_nowait()
                self.dropped+=1
            except Empty:
                pass
            self.queue.put_nowait(packet)

    def log(self, event, **data):
        self.counts[event]+=1
        self.writer.write(json.dumps(dict(event=event,**data),ensure_ascii=False)+'\n')
        self.writer.flush()

    def process(self, raw_frame, d2c):
        # Private calibrated frame & private caches, no access to motion-owned models.
        f=apply_runtime_depth_calibration(replace(raw_frame),self.calibration,d2c)
        stamp=depth_capture_timing(f)['depth_sample_monotonic_estimate']
        if self.detector is None or not self.detector.surface.matches_frame(f,d2c):
            self.flush_pending(invalidate=True)
            surface=BeltSurface.from_frame(f,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
            self.detector=NativeGeometry(surface,f,self.config,self.parameters)
            self.association=MotionAssociation(self.parameters.association_gate_m)
            self.generation+=1
        velocity=prediction_velocity(self.config,None,self.detector.surface.normal_base)
        axis=(self.config.direction@self.detector.plane[:3,:3])[:2]
        horizon=(np.ptp(self.detector.hull@axis)+self.parameters.association_gate_m)/(velocity@self.config.direction)
        obs=self.detector.extract(f,d2c,4)
        tracks=self.association.update(f.header.clock_epoch,f.header.sequence,stamp,[o.center for o in obs],velocity,horizon)
        for o,t in zip(obs,tracks):
            if t is None:
                self.log('ambiguous_identity',sequence=f.header.sequence,capture_s=stamp)
                continue
            if t.frozen:continue
            identity=(f.header.clock_epoch,self.generation,t.id)
            key=f'{identity[0]}_{identity[1]}_{identity[2]}_{f.header.sequence}'
            np.savez_compressed(self.directory/(key+'.npz'),indices=o.indices,points=o.points,
                grasps=o.grasps,center=o.center,capture_s=stamp,sequence=f.header.sequence,identity=identity)
            self.log('raw_geometry_saved',identity=identity,sequence=f.header.sequence,capture_s=stamp,
                     point_count=len(o.points),center=o.center.tolist(),path=key+'.npz')
            if not 0<=self.clock()-stamp<=self.config.raw['maximum_observation_age_s']:
                self.log('expired_geometry',key=key)
                continue
            t.frozen=True
            if self.feedback_reader is not None:
                self.pending[key]=(stamp, o, f, d2c, t)
        self.flush_pending()

    def flush_pending(self, *, invalidate=False):
        for key,(stamp,ob,f,d2c,track) in list(self.pending.items()):
            if invalidate or not 0<=self.clock()-stamp<=self.config.raw['maximum_observation_age_s']:
                self.log('unpaired_geometry_retained',key=key,reason='reset_or_shutdown' if invalidate else 'age')
                track.frozen=False  # A later fresh exposure can use the same motion ID.
            else:
                state=self.feedback_reader(stamp)
                if state is None or not capture_joint_pairing_valid(state,stamp):
                    continue
                # Resolve the original pixels with the historical joints. A disconnected
                # remnant cannot inherit the complete candidate's ID by assumption.
                removed=self.model.classify_robot_points(ob.points,state,camera_origin=f.camera_to_world_cv[:3,3]).removed_mask
                paired=ob
                if np.any(removed):
                    import cv2
                    kept=~removed
                    mask=np.zeros(f.depth_m.shape,np.uint8)
                    mask.flat[ob.indices[kept]]=1
                    components,_=cv2.connectedComponents(mask,connectivity=8)
                    paired=None
                    if components==2 and kept.sum()>=self.parameters.min_points:
                        region=Region(ob.indices[kept],ob.points[kept],None)
                        paired=self.detector.measure(region,f,d2c)
                if paired is None:
                    self.log('paired_self_filter_rejected',key=key)
                    track.frozen=False
                else:
                    np.savez_compressed(self.directory/(key+'.paired.npz'),points=paired.points,
                        indices=paired.indices,grasps=paired.grasps,capture_s=stamp,
                        joints=state['joints'],brackets_s=state['capture_joint_pairing']['bracket_bounds_s'])
                    self.log('paired_geometry_ready',key=key,points=len(paired.points),
                        center=paired.center.tolist(),grasps=np.asarray(paired.grasps).tolist(),
                        planning_submitted=False)
            del self.pending[key]

    def _run(self):
        with (self.directory/'events.jsonl').open('x') as self.writer:
            while not self.stop.is_set():
                try:
                    try: packet=self.queue.get(timeout=.025)
                    except Empty:
                        self.flush_pending()
                        continue
                    self.process(*packet)
                except BeltSurfaceNotReady as error:
                    self.log('belt_not_ready',reason=str(error))
                except Exception as error:
                    self.log('observation_error',type=type(error).__name__,reason=str(error))
            self.flush_pending(invalidate=True)

    def start(self):
        self.thread.start()

    def close(self):
        self.stop.set()
        self.thread.join()
        (self.directory/'summary.json').write_text(json.dumps(dict(events=self.counts,
            overwritten_input_frames=self.dropped,planning_submissions=0,robot_commands=0),indent=2))
