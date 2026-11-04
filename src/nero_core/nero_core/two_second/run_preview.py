"""Passive reader for a pick run: opens files only, never a camera or CAN."""
import json
from pathlib import Path
from threading import Event,Lock,Thread
import numpy as np
from .collision_view import read_packet,restore_frame


class RunPreviewWorker(Thread):
    def __init__(self, run, *, direction=(1.,0.,0.)):
        super().__init__(name='passive-run-viewer',daemon=True)
        self.run_directory=Path(run); self.stop=Event(); self.rearm=Event(); self.lock=Lock()
        self.latest={'status':'等待抓取程序共享画面','shared_run':True}
        self.can_connections=self.freeze_count=0
        self.camera_connections=0
        self.frame_key=self.collision_key=None
        self.target_key=None;self.plan_key=None;self.plan_record=None
        self.summary_key=None
        self.events_key=None;self.direction=np.asarray(direction,dtype=float)
        self.selected=None; self.selection_changed=False

    def select_collision(self, key):
        with self.lock:self.selected=key; self.selection_changed=True

    def poll(self):
        with self.lock:
            snap=dict(self.latest); selected=self.selected; changed=self.selection_changed
            self.selection_changed=False
        path=self.run_directory/'preview/latest.npz'
        key=path.stat().st_mtime_ns if path.exists() else None
        if key is not None and key!=self.frame_key:
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                frame=restore_frame(meta,{'frame_rgb':data['frame_rgb'],'frame_depth_m':data['frame_depth_m']})
            snap.update(frame=frame,d2c=np.asarray(meta['depth_to_color_cv']),replay=False,
                live_speed=meta.get('live_speed'))
            from .speed_overlay import paired_track_labels
            snap['track_labels']=paired_track_labels(snap['live_speed'],frame,self.direction)
            self.frame_key=key
        path=self.run_directory/'preview/target.npz'
        key=path.stat().st_mtime_ns if path.exists() else None
        if key is not None and key!=self.target_key:
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                frame=restore_frame(meta,{'frame_rgb':data['frame_rgb'],'frame_depth_m':data['frame_depth_m']})
                geometry={'points':data['points'],'grasps':tuple(data['grasps']),'obb':meta['obb'],
                    'frame':frame,'d2c':np.asarray(meta['depth_to_color_cv'])}
            snap.update(live_geometry=geometry,production_target=meta['identity'],
                production_frame_sequence=meta['header']['sequence'],
                frozen_speed=meta.get('frozen_speed'),
                geometry_status='抓取程序实际冻结输入；是否可执行以轨迹认证为准')
            if self.frame_key is None:snap.update(frame=frame,d2c=geometry['d2c'],replay=True)
            self.target_key=key
        path=self.run_directory/'preview/plan.npz'
        key=path.stat().st_mtime_ns if path.exists() else None
        if key is not None and key!=self.plan_key:
            with np.load(path,allow_pickle=False) as data:
                self.plan_record={'metadata':json.loads(str(data['metadata'])),
                    **{name:data[name].copy() for name in ('times','joints','tcp_xyz_base_m',
                        'contact_grasp_base','selected_grasp_base','predicted_contact_points_base_m')}}
            self.plan_key=key
        plan=self.plan_record
        if (plan and plan['metadata']['identity']==snap.get('production_target') and
                plan['metadata']['frame_sequence']==snap.get('production_frame_sequence')):
            snap['production_plan']=plan
        else:
            snap.pop('production_plan',None)
        events=self.run_directory/'events.jsonl'
        frozen=snap.get('frozen_speed')
        incomplete=not frozen or frozen.get('from_saved_events') and frozen.get('position_base_m') is None
        if snap.get('production_target') and incomplete and events.is_file():
            key=events.stat().st_mtime_ns
            if key!=self.events_key:
                from .speed_overlay import frozen_from_events
                snap['frozen_speed']=frozen_from_events(events,snap['production_target'],self.direction)
                if snap['frozen_speed']:
                    target_frame=snap['live_geometry']['frame']
                    snap['frozen_speed'].update(calibration_id=target_frame.calibration_id,
                        serial_number=target_frame.serial_number)
                self.events_key=key
        records=sorted((self.run_directory/'collisions').glob('collision_*.json'))
        keys=[p.stem for p in records]
        if records:
            chosen=next((p for p in records if p.stem==selected),records[-1])
            if chosen.stem!=self.collision_key or changed:
                packet=read_packet(chosen)
                snap.update(collision=packet,scene=packet['arrays']['scene_points_base_m'],
                    scene_source=('measured_belt_surface_only' if packet['metadata'].get(
                        'environment_collision_scope')=='measured_belt_surface_only' else
                        'original_RobotModel.install_scene'))
                self.collision_key=chosen.stem
                if self.frame_key is None:
                    snap.update(frame=packet['frame'],d2c=packet['d2c'],replay=True)
            snap['selected_collision']=chosen.stem
        collision=snap.get('collision')
        if (collision and snap.get('production_plan') and
                collision['metadata'].get('target_identity') != snap['production_target']):
            # Do not overlay a previous target's collision on the new plan.
            snap.pop('collision',None)
            collision=None
        suffix='' if collision is None else '；预测碰撞：'+collision['metadata']['geometry']
        source=(getattr(snap.get('frame'),'source','') if collision is None else
            (collision['metadata'].get('camera') or {}).get('source',''))
        prefix='【离线合成验证】' if 'synthetic' in source.lower() else ''
        path=self.run_directory/'summary.json'
        key=path.stat().st_mtime_ns if path.exists() else None
        if key is not None and key!=self.summary_key:
            summary=json.loads(path.read_text())
            snap['run_status']=summary.get('status');self.summary_key=key
        if snap.get('run_status'):
            prefix+='任务已结束，保留历史画面；'
            snap['replay']=True
        snap.update(shared_run=True,collision_choices=keys,
            status=prefix+'只读共享抓取画面，不重复打开相机/CAN'+suffix)
        with self.lock:self.latest=snap
        return snap

    def run(self):
        while not self.stop.is_set():
            try:self.poll()
            except (OSError,ValueError,KeyError) as error:
                with self.lock:self.latest={**self.latest,'status':f'等待完整共享记录：{error}'}
            self.stop.wait(.2)
