"""Dedicated camera-only fixed-position point-cloud and model-path experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import signal
from threading import Event,Lock,Thread
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

from .collision_capture import frame_metadata
from .config import CellConfig
from .diagnostics import json_value
from .fixed_freeze import ExposureBuffer,NativeDepthLocator,capture_admission,frozen_geometry
from .measured_rgbd import depth_capture_timing,fit_plane,green_belt_points
from .preview import box_in_base,CameraPreview
from .reference import EntryNotReady
from .trajectory_preview import TrajectoryPreview
from .velocity_source import prediction_velocity,with_fixed_belt_speed,policy
from .vision import obb_grasps
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface,BeltSurfaceNotReady
from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration,load_runtime_depth_calibration
from mujoco_conveyor_sorting.runtime.single_pick import detect_target,PickRejected


def interface_velocity_policy(config):
    cell_limit=config.raw.get('_camera_test_cell_speed_limit_m_s',
                              config.raw.get('maximum_measured_speed_m_s'))
    return {**policy(config),'tracking_source':'none_single_exposure_native_depth',
            'internal_tracking_kalman_still_runs':False,
            'identity_source':'manually_armed_exposure_event',
            'cell_speed_limit_cm_s':None if cell_limit is None else cell_limit*100.,
            'camera_test_speed_override':bool(cell_limit is not None and
                config.raw['fixed_belt_speed_m_s']>cell_limit)}


class FixedFreezeWorker(Thread):
    """Raw reader, light locator, geometry and planner use separate workers."""
    def __init__(self,config,calibration,output,*,buffer_frames=16,plan=True):
        super().__init__(name='fixed-position-test')
        self.config,self.calibration,self.output=config,calibration,Path(output)
        self.output.mkdir(parents=True,exist_ok=False)
        self.stop,self.rearm=Event(),Event()
        self.lock=Lock();self.latest={'geometry_legend':'青色：工件点云  绿色：OBB  橙色：候选夹爪  黄色：模型规划路径'}
        self.buffer=ExposureBuffer(buffer_frames)
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fixed-exposure-geometry')
        self.archive=ThreadPoolExecutor(max_workers=1,thread_name_prefix='fixed-exposure-save')
        self.paths=TrajectoryPreview(config,self.output/'paths') if plan else None
        self.camera_connections=0;self.rows=[];self.generation=0
        self.status='准备相机';self.capture_error=None;self.pending_archives=[]
        self.accepted=None;self.geometry_future=None;self.pending=[]
        self.current=None;self.frozen=False;self.trigger_capture=None
        self.minimum_capture_s=float('-inf')
        self.processed=0;self.last_diagnostics=0.

    def publish(self,**values):
        with self.lock:self.latest={**self.latest,**values}

    def log(self,event,**values):
        row=json_value(dict(event=event,monotonic_s=time.monotonic(),**values))
        with (self.output/'diagnostics.jsonl').open('a') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')

    def summary(self,closed=False):
        report=dict(mode='camera_only_fixed_position_single_exposure',
            source_config_path=str(getattr(self.config,'source','')),
            source_config_sha256=getattr(self.config,'source_sha256',None),
            model_planning_requested=self.paths is not None,
            fixed_speed_policy=interface_velocity_policy(self.config),freeze_upstream_cm=self.config.raw['freeze_position_upstream_m']*100,
            tolerance_cm=self.config.raw['freeze_position_tolerance_m']*100,
            identity_policy='one manual-armed exposure event; no multiframe tracking or speed qualification',
            raw_frame_buffer=self.buffer.diagnostics(),processed_frames=self.processed,
            camera_connections=self.camera_connections,can_connections=0,robot_commands_sent=0,
            completed_exposure_trials=len(self.rows),
            manual_tests_with_result=len({r['generation'] for r in self.rows}),
            counting_rule='Each saved row is one real exposure trial; retries within one manual test share generation and are not independent workpieces.',
            point_clouds_generated=sum(r['point_cloud_generated'] for r in self.rows),
            model_paths_generated=sum(r.get('model_path_generated',False) for r in self.rows),
            camera_closed=closed,archive_errors=getattr(self,'archive_errors',[]),attempts=self.rows,
            limitation='No robot feedback. Model paths assume configured standby and retain the original exposure timestamp. No claim of hardware success.')
        temporary=self.output/'summary.json.tmp'
        temporary.write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2)+'\n')
        temporary.replace(self.output/'summary.json')

    def reader(self,camera):
        sequence=None
        try:
            while not self.stop.is_set():
                frame=camera.read_next(timeout_s=.5,after_sequence=sequence,wait_for_fresh=True)
                sequence=frame.header.sequence
                self.buffer.put(frame)
        except Exception as error:
            if not self.stop.is_set():
                self.capture_error=f'{type(error).__name__}: {error}'
        finally:self.buffer.close()

    def prepare(self,frame,d2c,surface,candidate,capture,generation):
        start=time.monotonic()
        geometry=None
        row=dict(generation=generation,frame_sequence=frame.header.sequence,capture_monotonic_s=capture,
            point_cloud_generated=False,model_path_generated=False,can_connections=0,robot_commands_sent=0)
        arrays=dict(frame_rgb=frame.rgb,frame_depth_m=frame.depth_m,
            trigger_seed_pixel_indices=candidate.seed_pixel_indices,
            capture_monotonic_s=np.asarray(capture),
            metadata=np.asarray(json.dumps(json_value(frame_metadata(frame,d2c)),ensure_ascii=False)))
        try:
            detection,position,velocity,grasps,overlap=frozen_geometry(self.config,frame,d2c,surface,
                candidate,capture,clock=time.monotonic,detect_fn=detect_target,grasp_fn=obb_grasps)
            points=np.asarray(detection['object_points'])
            geometry=dict(frame=frame,rgb_frame=frame,d2c=d2c,points=points,grasps=grasps,
                obb=box_in_base(detection['obb'],surface.plane_to_base),source_sequence=frame.header.sequence,
                source_device_s=frame.header.source_time)
            ready=time.monotonic()
            row.update(point_cloud_generated=True,point_count=len(points),seed_overlap_ratio=overlap,
                capture_upstream_cm=float((self.config.follow_bounds[0]-position@self.config.direction)*100),
                locator_upstream_cm=float((self.config.follow_bounds[0]-candidate.position@self.config.direction)*100),
                capture_age_at_point_cloud_ready_s=ready-capture,geometry_wall_s=ready-start,
                depth_modified=False,multiframe_matching_used=False)
            arrays.update(object_points_base_m=points,grasps=np.asarray(grasps),
                scene_points=detection['scene_points'],scene_pixel_indices=detection['scene_pixel_indices'],
                detection_metadata=np.asarray(json.dumps(json_value({k:detection[k] for k in ('plane','belt_region')}))))
            if generation==self.generation:
                self.publish(live_geometry=geometry,status=f'已生成 {len(points)} 点工件点云，正在规划' if self.paths else '点云已锁存；空格开始下一次')
            result=None
            if self.paths:
                # A one-frame observation, not a fabricated confirmed visual track.
                observation=SimpleNamespace(observed_position_base=position,velocity_base_m_s=None,
                    track_id=frame.header.sequence,sample_count=1,span_s=0.,speed_valid=False,
                    observation_source='single_exposure_native_depth')
                peer=SimpleNamespace(clock_epoch=frame.header.clock_epoch,track_generation=generation)
                done=dict(detection=detection,geometry=geometry,capture=capture,peer=peer)
                result=self.paths._calculate(done,observation,.03,self.paths.generation)
                report=result['report']
                row.update(model_path_generated=True,remaining_send_lead_s=report['actual_remaining_send_lead_s'],
                    timing_feasible=report['actual_remaining_send_lead_s']>=0,
                    planning_wall_s=report['planning_wall_s'],path_report_file=result['file']+'.json')
                row['status']='model_path_ready' if row['timing_feasible'] else 'model_path_deadline_missed'
            else:row['status']='point_cloud_ready'
            return row,geometry,result,arrays
        except Exception as error:
            row.update(status='planning_failed' if row['point_cloud_generated'] else 'point_cloud_rejected',
                error_type=type(error).__name__,error=str(error),failure_details=getattr(error,'details',{}),
                admission_reason=getattr(error,'reason',None))
            return row,geometry,None,arrays

    def finish(self):
        if self.geometry_future is None or not self.geometry_future.done():return
        row,geometry,path,arrays=self.geometry_future.result();self.geometry_future=None
        index=len(self.rows)+1
        row.update(attempt=index,evidence_file=str(self.output/f'attempt_{index:03d}.npz'))
        self.pending_archives.append(self.archive.submit(np.savez,row['evidence_file'],**arrays))
        self.rows.append(row)
        with (self.output/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(json_value(row),ensure_ascii=False)+'\n')
        self.log('attempt_finished',**row)
        if row['generation']!=self.generation:
            self.log('result_discarded_after_rearm',frame_sequence=row['frame_sequence'])
            if self.pending and not self.stop.is_set():self.submit(self.pending.pop(0))
            self.summary()
            return
        if row['point_cloud_generated']:
            self.pending.clear();self.frozen=True
            text=(f'点云{row["point_count"]}点，位于上游{row["capture_upstream_cm"]:.2f}cm；'+
                (f'模型路径已生成，发送截止余量{row["remaining_send_lead_s"]*1000:.0f}ms' if row.get('model_path_generated') else
                 f'规划未生成：{row.get("error","")}' if self.paths else '未请求模型规划'))
            self.publish(status=text+'；空格开始下一次',live_geometry=geometry,path_preview=path)
            print('[结果]',text,flush=True)
        elif self.pending and not self.stop.is_set():
            self.submit(self.pending.pop(0))
        else:
            self.trigger_capture=None
            self.publish(status='本帧点云未通过：'+row['error']+'；继续等待冻结带内有效帧')
        self.summary()

    def submit(self,item):
        frame,d2c,surface,candidate,capture=item
        self.geometry_future=self.pool.submit(self.prepare,frame,d2c,surface,candidate,capture,self.generation)
        self.publish(status='已锁存原始曝光，正在提取完整工件点云')

    def run(self):
        camera=None;capture_thread=None;surface=locator=None;epoch=None
        try:
            if self.paths:
                from .planning import MeasuredPlanner,calibrated_model
                self.paths.planner=MeasuredPlanner(self.config,model=calibrated_model(self.config))
            from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
            values=dict(self.config.raw['camera'])
            values.update(enable_color=True,require_color=True,timestamp_source='device',
                serial_number=self.calibration['serial_number'],calibration_id=self.calibration['calibration_id'],
                camera_to_world_cv=self.calibration['depth_to_base_cv'])
            camera=OrbbecCameraBackend(values);camera.start();self.camera_connections=1
            camera.read_latest(timeout_s=3.,wait_for_fresh=True)
            d2c=camera.get_depth_to_color_cv()
            capture_thread=Thread(target=self.reader,args=(camera,),name='fixed-freeze-raw-capture')
            capture_thread.start()
            self.log('started',configuration=dict(freeze_upstream_m=self.config.raw['freeze_position_upstream_m'],
                tolerance_m=self.config.raw['freeze_position_tolerance_m']),velocity_policy=interface_velocity_policy(self.config))
            self.summary()
            while not self.stop.is_set():
                self.finish()
                if self.rearm.is_set():
                    self.rearm.clear();self.generation+=1;self.frozen=False;self.pending.clear();self.trigger_capture=None
                    self.minimum_capture_s=time.monotonic()
                    if self.paths:self.paths.reset()
                    self.publish(live_geometry=None,path_preview=None,status='已重新等待下一次过线')
                    self.log('manually_rearmed',generation=self.generation)
                raw=self.buffer.take()
                if raw is None:
                    if self.capture_error:raise RuntimeError(self.capture_error)
                    continue
                frame=apply_runtime_depth_calibration(raw,self.calibration,d2c)
                capture=depth_capture_timing(frame)['depth_sample_monotonic_estimate']
                if capture<self.minimum_capture_s:continue
                if epoch!=frame.header.clock_epoch:
                    epoch=frame.header.clock_epoch;surface=locator=None
                    self.generation+=1;self.frozen=False;self.pending.clear();self.trigger_capture=None
                    if self.paths:self.paths.reset()
                    self.publish(live_geometry=None,path_preview=None)
                self.processed+=1
                self.publish(frame=frame,d2c=d2c,buffer=self.buffer.diagnostics())
                if self.frozen:continue
                if surface is None or not surface.matches_frame(frame,d2c):
                    self.publish(status='拟合实测传送带，工件请先放在冻结线上游')
                    try:surface=BeltSurface.from_frame(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points)
                    except BeltSurfaceNotReady as error:
                        self.publish(status=str(error));continue
                    locator=NativeDepthLocator(self.config,surface,frame)
                velocity=prediction_velocity(self.config,None,surface.normal_base)
                began=time.monotonic();candidates,diagnostics=locator.locate(frame)
                diagnostics.update(locator_wall_s=time.monotonic()-began,sequence=frame.header.sequence,
                                   capture_monotonic_s=capture,capture_age_s=time.monotonic()-capture)
                self.publish(surface=surface,locator_diagnostics=diagnostics,
                    scene=CameraPreview.scene_cloud(frame),scene_source='raw_display_only')
                if len(candidates)!=1:
                    if self.geometry_future is None:
                        self.publish(status=f'冻结附近单帧深度候选{len(candidates)}个，等待一个完整独立工件')
                    diagnostics['reason']='no_candidate' if not candidates else 'multiple_candidates'
                else:
                    candidate=candidates[0]
                    diagnostics['candidate_upstream_cm']=float((self.config.follow_bounds[0]-candidate.position@self.config.direction)*100)
                    try:capture_admission(self.config,capture,candidate.position,velocity,time.monotonic())
                    except EntryNotReady as error:
                        diagnostics['reason']=error.reason
                        if self.geometry_future is None:self.publish(status=f'单帧工件位于上游{diagnostics["candidate_upstream_cm"]:.2f}cm；'+
                            ('等待进入冻结带' if error.reason=='too_far_upstream' else '已错过冻结带，请通过下一件或按空格重置'))
                    else:
                        item=(frame,np.array(d2c,copy=True),surface,candidate,capture)
                        if self.trigger_capture is None:self.trigger_capture=capture
                        if self.geometry_future is None:self.submit(item)
                        elif capture-self.trigger_capture<=self.config.raw['freeze_position_tolerance_m']/float(velocity@self.config.direction)+.04:
                            if len(self.pending)<3:self.pending.append(item)
                        diagnostics['reason']='in_band_exposure_retained'
                if time.monotonic()-self.last_diagnostics>=.1 or diagnostics.get('reason')=='in_band_exposure_retained':
                    self.log('locator',**diagnostics);self.last_diagnostics=time.monotonic()
        except Exception as error:
            self.publish(status='测试已停止：'+str(error),error=str(error))
            self.log('stopped_with_error',error_type=type(error).__name__,error=str(error))
            print('[停止]',error,flush=True)
        finally:
            self.stop.set();self.buffer.close()
            if self.paths:self.paths.reset()
            if capture_thread:capture_thread.join(timeout=2.)
            if camera:camera.stop()
            self.pool.shutdown(wait=True,cancel_futures=False)
            self.pending.clear()
            self.finish()
            if self.paths:self.paths.close()
            self.archive.shutdown(wait=True)
            errors=[]
            for future in self.pending_archives:
                try:future.result()
                except Exception as error:errors.append(str(error))
            if errors:self.log('archive_errors',errors=errors)
            self.archive_errors=errors
            self.summary(closed=True)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description='固定位置单帧工件点云测试：只开相机，不连接CAN，不执行机械臂。')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--belt-speed-cm-s',type=float,required=True)
    parser.add_argument('--freeze-upstream-cm',type=float,default=50.)
    parser.add_argument('--tolerance-cm',type=float,default=2.)
    parser.add_argument('--buffer-frames',type=int,default=16)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seconds',type=float,default=0.)
    parser.add_argument('--no-plan',action='store_true',help='只测原始曝光和点云，不计算模型路径')
    parser.add_argument('--check-config',action='store_true',help='仅校验接口，不打开相机或界面')
    args=parser.parse_args(argv)
    if not np.isfinite(args.belt_speed_cm_s) or args.belt_speed_cm_s<=0:
        parser.error('相机测试带速必须是正有限cm/s')
    if not np.isfinite([args.freeze_upstream_cm,args.tolerance_cm,args.seconds]).all() or not 0<args.tolerance_cm<args.freeze_upstream_cm or args.seconds<0:
        parser.error('冻结位置、容差和运行时间无效')
    try:
        config=CellConfig.load(args.config)
        speed=args.belt_speed_cm_s/100.
        cell_limit=config.raw['maximum_measured_speed_m_s']
        # This executable has no robot/CAN execution path. Admit the stated
        # test speed in this private config without changing the cell YAML or
        # weakening with_fixed_belt_speed for hardware callers. Keep all
        # kinematic limits and actual planning/deadline checks unchanged.
        config=replace(config,raw={**config.raw,
            '_camera_test_cell_speed_limit_m_s':cell_limit,
            'maximum_measured_speed_m_s':max(cell_limit,speed)})
        config=with_fixed_belt_speed(config,speed)
        if config.raw['intercept_strategy']!='synchronized':raise ValueError('该接口需要synchronized策略')
        config=replace(config,raw={**config.raw,'freeze_position_upstream_m':args.freeze_upstream_cm/100.,
            'freeze_position_tolerance_m':args.tolerance_cm/100.,
            'velocity_override_origin':'cli_fixed_freeze_test'})
        ExposureBuffer(args.buffer_frames)
        calibration=load_runtime_depth_calibration(config.raw['camera_calibration_file'])
        if not args.no_plan:
            from .taught import TaughtSegment
            config.raw['taught_segments']={kind:TaughtSegment.load(config.raw[f'taught_{kind}_file'],config,kind)
                                         for kind in ('transfer','return')}
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))
    if args.output is None:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S_%f')
        args.output=Path.cwd()/'runs'/f'fixed_freeze_test_{stamp}'
    if args.output.exists():parser.error('输出目录已存在，请换一个新目录，避免覆盖记录')
    return args,config,calibration


def main(argv=None):
    args,config,calibration=arguments(argv)
    if args.check_config:
        print(json.dumps(dict(status='configuration_valid',interface='single_frame_fixed_position',
            belt_speed_cm_s=args.belt_speed_cm_s,freeze_upstream_cm=args.freeze_upstream_cm,
            tolerance_cm=args.tolerance_cm,planning=not args.no_plan,can_connections=0,
            robot_commands_sent=0,camera_connections=0,
            velocity_policy=interface_velocity_policy(config)),ensure_ascii=False))
        return 0
    # Qt and camera SDK import only when the user actually runs the test.
    from PyQt5 import QtCore,QtGui,QtWidgets
    from .camera_view import CloudView
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    app=QtWidgets.QApplication([])
    worker=FixedFreezeWorker(config,calibration,args.output,buffer_frames=args.buffer_frames,plan=not args.no_plan)
    class Window(QtWidgets.QWidget):
        def __init__(self):
            super().__init__();self.setWindowTitle('固定位置点云测试｜仅相机｜CAN与机器人命令=0')
            self.resize(1220,900);layout=QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel(f'带速 {args.belt_speed_cm_s:g} cm/s；冻结上游 {args.freeze_upstream_cm:g}～{args.freeze_upstream_cm-args.tolerance_cm:g} cm；无多帧ID匹配'))
            images=QtWidgets.QHBoxLayout();self.live=QtWidgets.QLabel('等待相机');self.saved=QtWidgets.QLabel('等待冻结原始曝光')
            for title,label in (('实时相机',self.live),('冻结的原始曝光',self.saved)):
                column=QtWidgets.QVBoxLayout();column.addWidget(QtWidgets.QLabel(title))
                label.setMinimumSize(400,225);label.setAlignment(QtCore.Qt.AlignCenter);column.addWidget(label)
                images.addLayout(column)
            layout.addLayout(images);self.cloud=CloudView(config);layout.addWidget(self.cloud,1)
            self.status=QtWidgets.QLabel();self.status.setWordWrap(True);layout.addWidget(self.status)
            controls=QtWidgets.QHBoxLayout();again=QtWidgets.QPushButton('下一次测试（空格）');again.clicked.connect(worker.rearm.set)
            controls.addWidget(again);controls.addWidget(QtWidgets.QLabel(str(args.output)));layout.addLayout(controls)
            self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(100)
        def refresh(self):
            with worker.lock:snap=dict(worker.latest)
            self.cloud.set_snapshot(snap)
            for label,frame in ((self.live,snap.get('frame')),(self.saved,(snap.get('live_geometry') or {}).get('frame'))):
                if frame is not None and frame.rgb is not None:
                    rgb=np.ascontiguousarray(frame.rgb)
                    image=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
                    label.setPixmap(QtGui.QPixmap.fromImage(image).scaled(label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            buf=snap.get('buffer',{});diag=snap.get('locator_diagnostics',{})
            self.status.setText(snap.get('status','启动中')+f'\n原始帧 {buf.get("received",0)}；缓存溢出丢帧 {buf.get("dropped",0)}；定位 {diag.get("locator_wall_s",0)*1000:.1f}ms。空格开始下一次，Esc退出。')
        def keyPressEvent(self,event):
            if event.key()==QtCore.Qt.Key_Space:worker.rearm.set()
            elif event.key()==QtCore.Qt.Key_Escape:self.close()
        def closeEvent(self,event):
            worker.stop.set();worker.buffer.close()
            if worker.paths:worker.paths.reset()
            event.accept()
    window=Window();window.show();worker.start()
    signal.signal(signal.SIGINT,lambda *_:window.close());signal.signal(signal.SIGTERM,lambda *_:window.close())
    if args.seconds:QtCore.QTimer.singleShot(int(args.seconds*1000),window.close)
    print('固定位置点云测试：仅打开相机，CAN连接=0，机器人/夹爪命令=0。空格下一次，Esc退出。',flush=True)
    print('记录目录：'+str(args.output.resolve()),flush=True)
    code=app.exec_();worker.stop.set();worker.buffer.close();worker.join()
    return 1 if worker.capture_error or worker.latest.get('error') else code


if __name__=='__main__':raise SystemExit(main())
