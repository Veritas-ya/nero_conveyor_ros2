"""Asynchronous collision evidence and lossy single-camera preview transport."""
import json
from pathlib import Path
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
import time
import numpy as np
from .collision_capture import frame_metadata
from .diagnostics import json_value


class VisualArchive:
    def __init__(self, output, emit, *, live=False, direction=(1.,0.,0.), path_model_factory=None):
        self.directory=Path(output); self.emit=emit; self.live=live
        self.queue=Queue(maxsize=8); self.stop=Event(); self.lock=Lock()
        self.frame=None; self.last_frame_write=-float('inf')
        self.speed=None;self.direction=np.asarray(direction,dtype=float).copy()
        self.path_model_factory=path_model_factory;self.path_model=None
        self.saved=[]; self.errors=[]; self.dropped=0; self.write_wall_s=0.
        (self.directory/'preview').mkdir(exist_ok=True,parents=True)
        self.thread=Thread(target=self._run,name='collision-evidence-writer',daemon=True)
        self.thread.start()

    def submit(self, packet):
        self.emit('collision_detected',**packet['metadata'])
        try:self.queue.put_nowait(packet)
        except Full:
            with self.lock:self.dropped+=1

    def offer_target(self, target):
        try:self.queue.put_nowait({'kind':'target','target':target,'accepted_s':time.monotonic()})
        except Full:
            with self.lock:self.dropped+=1

    def offer_plan(self, plan, *, width_m):
        """Queue only; visualization must never delay the command sender."""
        try:self.queue.put_nowait({'kind':'plan','plan':plan,'width_m':float(width_m)})
        except Full:
            with self.lock:self.dropped+=1

    def _plan(self, plan, width_m):
        if self.path_model_factory is None:
            return
        if self.path_model is None:
            self.path_model=self.path_model_factory()
        ref=plan.reference;path=plan.path;target=plan.target
        # Run FK on a private model, never the active planner's mutable model.
        times=np.linspace(0.,path.duration,max(2,int(np.ceil(path.duration*20))+1))
        rotate_first_s=float(getattr(ref,'rotate_first_s',getattr(ref,'prealign_s',0.)))
        times=np.unique(np.r_[times,rotate_first_s,getattr(ref,'prealign_s',0.),ref.duration_s,
            min(path.duration,getattr(ref,'follow_end_s',ref.duration_s+.6))])
        times=times[(times>=0)&(times<=path.duration)]
        joints=np.array([path.position(t) for t in times]);xyz=[]
        for q in joints:
            self.path_model.set_state(q,width_m)
            xyz.append(self.path_model.tcp()[:3,3].copy())
        contact=np.asarray(plan.selected_grasp).copy();contact[:3,3]=ref.contact_xyz
        meta={'identity':list(target.identity),'frame_sequence':int(target.sequence),
            'capture_monotonic_s':float(target.capture_s),'path_version':int(path.version),
            'stage':path.stage,'path_duration_s':float(path.duration),
            'reference_start_monotonic_s':float(ref.start_s),
            'contact_elapsed_s':float(ref.duration_s),
            'rotate_first_s':rotate_first_s,
            'grasp_axis_policy':path.diagnostics.get('grasp_axis_policy','historical_unspecified'),
            'translation_start_elapsed_s':rotate_first_s,
            'join_s':path.diagnostics.get('join_s'),
            'selected_input_grasp_index':path.diagnostics.get('selected_input_grasp_index'),
            'follow_end_elapsed_s':float(min(path.duration,getattr(ref,'follow_end_s',ref.duration_s+.6))),
            'planning_wall_s':float(path.planning_s),
            'meaning':'planner_output_not_proof_of_dispatch_or_motion'}
        destination=self.directory/'preview/plan.npz'
        with destination.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream,metadata=np.asarray(json.dumps(json_value(meta),ensure_ascii=False)),
                times=times,joints=joints,tcp_xyz_base_m=np.asarray(xyz),
                contact_grasp_base=contact,selected_grasp_base=np.asarray(plan.selected_grasp),
                predicted_contact_points_base_m=(np.asarray(target.points)+np.asarray(target.velocity)*(ref.contact_s-target.capture_s)))
        destination.with_suffix('.tmp').replace(destination)

    def _target(self, target, accepted_s):
        from .preview import box_in_base
        from .speed_overlay import frozen_speed_snapshot
        frame=target.detection.get('frame')
        if frame is None:return
        d2c=target.detection['depth_to_color_cv']
        meta=frame_metadata(frame,d2c)
        meta.update(identity=list(target.identity),capture_s=target.capture_s,
            frozen_speed=frozen_speed_snapshot(target,self.direction,accepted_s),
            obb=box_in_base(target.detection['obb'],target.detection['belt_region']['plane_to_base']))
        destination=self.directory/'preview/target.npz'
        with destination.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream,metadata=np.asarray(json.dumps(json_value(meta),ensure_ascii=False)),
                frame_rgb=frame.rgb,frame_depth_m=frame.depth_m,
                points=target.points,grasps=np.asarray(target.grasps))
        destination.with_suffix('.tmp').replace(destination)

    def offer_frame(self, frame, d2c):
        if self.live:
            with self.lock:self.frame=(frame,d2c)

    def offer_speed(self, frame, d2c, peer, report, *, capture_s):
        if self.live:
            from .speed_overlay import live_speed_snapshot
            record=live_speed_snapshot(frame,peer,report,self.direction,capture_s)
            with self.lock:
                self.speed=record
                self.frame=(frame,d2c)

    def _live_frame(self, *, force=False):
        now=time.monotonic()
        if not self.live or not force and now-self.last_frame_write<.2:return False
        with self.lock:
            item=self.frame; self.frame=None; speed=self.speed
        if item is None:return False
        frame,d2c=item
        meta=frame_metadata(frame,d2c)
        meta['live_speed']=speed
        destination=self.directory/'preview/latest.npz'
        # One atomic binary file keeps image, intrinsics, calibration and
        # sequence together; no JSON image arrays and no second capture owner.
        with destination.with_suffix('.tmp').open('wb') as stream:
            np.savez(stream,frame_rgb=frame.rgb,frame_depth_m=frame.depth_m,
                metadata=np.asarray(json.dumps(json_value(meta),ensure_ascii=False)))
        destination.with_suffix('.tmp').replace(destination)
        self.last_frame_write=now
        return True

    def _run(self):
        from .collision_view import prepare_view,save_packet
        while not self.stop.is_set() or not self.queue.empty() or self.frame is not None:
            packet=None; did_work=False
            try:
                try:packet=self.queue.get(timeout=.05)
                except Empty:pass
                begin=time.monotonic()
                did_work=packet is not None
                if packet is not None:
                    if packet.get('kind')=='target':
                        self._target(packet['target'],packet['accepted_s'])
                    elif packet.get('kind')=='plan':
                        self._plan(packet['plan'],packet['width_m'])
                    else:
                        prepare_view(packet)
                        path=save_packet(self.directory/'collisions',packet)
                        self.saved.append(str(path))
                        self.emit('collision_saved',collision_id=packet['metadata']['collision_id'],path=str(path))
                did_work=self._live_frame(force=self.stop.is_set()) or did_work
            except Exception as error:
                did_work=True
                with self.lock:self.errors.append(f'{type(error).__name__}: {error}')
                self.emit('visualization_error',error_type=type(error).__name__,error=str(error))
            finally:
                # Queue waiting is excluded from artifact-writing time.
                if did_work:
                    self.write_wall_s+=time.monotonic()-begin
                if packet is not None:
                    self.queue.task_done()

    def close(self):
        self.stop.set(); self.thread.join(timeout=5.)
        with self.lock:
            return {'saved_collision_records':list(self.saved),'dropped':self.dropped,'errors':list(self.errors),
                'writer_alive':self.thread.is_alive(),'write_wall_s':self.write_wall_s,
                'live_preview_enabled':self.live,'camera_connections':0,'robot_commands':0}
