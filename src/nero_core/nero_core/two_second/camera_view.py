#!/usr/bin/env python3
"""RGB-D/OBB viewer, or passive shared-run camera and collision predictions."""
import argparse
from datetime import datetime
import itertools
import json
import html
import os
from pathlib import Path
import signal
import threading
import time
from dataclasses import asdict
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from nero_core.two_second.config import CellConfig
from nero_core.two_second.preview import CameraPreview
from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration


def sample(points, count):
    points = np.asarray(points).reshape(-1, 3)
    return points[np.linspace(0, len(points)-1, min(count, len(points)), dtype=int)]


def box_lines(frozen):
    box = frozen['obb']
    center, axes, half = (np.asarray(box[k]) for k in ('center_world_m', 'axes_world_columns', 'extent_m'))
    half = half/2
    signs = list(itertools.product((-1, 1), repeat=3))
    corners = np.array([center+axes@(half*s) for s in signs])
    return [(corners[i], corners[j]) for i in range(8) for j in range(i+1, 8)
            if np.count_nonzero(np.array(signs[i]) != signs[j]) == 1]


def grasp_lines(pose, width):
    # Original live_cloud_view schematic: TCP origin and opening along TCP Y.
    local = np.array([[0, -width/2, 0], [0, -width/2, -.04],
                      [0, width/2, -.04], [0, width/2, 0], [0, 0, -.08], [0, 0, 0]])
    world = local@pose[:3, :3].T+pose[:3, 3]
    return [(world[a], world[b]) for a, b in ((0,1),(1,2),(2,3),(4,5))]


def display_grasp(geometry,candidate=0,production_plan=None):
    """RGB and cloud views show the same selected pose for the same exposure."""
    frame=geometry.get('frame')
    sequence=geometry.get('source_sequence',getattr(getattr(frame,'header',None),'sequence',None))
    if production_plan and sequence is not None and sequence==production_plan['metadata'].get('frame_sequence'):
        return np.asarray(production_plan['selected_grasp_base'])
    return geometry['grasps'][min(max(0,candidate),len(geometry['grasps'])-1)]


def collision_predicted_box(geometry,metadata):
    """Translate this exposure's OBB to the collision prediction's timestamp."""
    if not geometry:return None
    frame=geometry.get('frame')
    sequence=geometry.get('source_sequence',getattr(getattr(frame,'header',None),'sequence',None))
    if sequence is None or sequence!=metadata.get('frame_sequence'):return None
    stamp,capture=metadata.get('predicted_monotonic_s'),metadata.get('capture_monotonic_s')
    velocity=metadata.get('target_velocity_base_m_s')
    if stamp is None or capture is None or velocity is None:return None
    box=dict(geometry['obb'])
    box['center_world_m']=(np.asarray(box['center_world_m'])+np.asarray(velocity)*(stamp-capture)).tolist()
    return {'obb':box}


def collision_predicted_grasp(metadata):
    """Frozen candidate translated to this predicted sample, not robot feedback."""
    context = metadata.get('geometry_context') or {}
    pose = context.get('selected_grasp_tcp_base')
    stamp, capture = metadata.get('predicted_monotonic_s'), metadata.get('capture_monotonic_s')
    velocity = metadata.get('target_velocity_base_m_s')
    if pose is None or stamp is None or capture is None or velocity is None:
        return None
    pose = np.asarray(pose, dtype=float).copy()
    velocity = np.asarray(velocity, dtype=float)
    if pose.shape != (4,4) or velocity.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(velocity).all() or not np.isfinite([stamp,capture]).all():
        return None
    pose[:3,3] += velocity*(stamp-capture)
    return pose


class CloudView(QtWidgets.QWidget):
    def __init__(self, config):
        super().__init__()
        self.config, self.snapshot = config, None
        self.center = np.array([-.35, -.1, .1])
        self.scale, self.yaw, self.pitch = 650., -.7, .85
        self.last = None
        self.candidate = 0
        self.collision_id = None
        self.plan_token = None
        self.setMinimumSize(560, 440)

    def set_snapshot(self, snapshot):
        first = self.snapshot is None or 'surface' not in self.snapshot
        self.snapshot = snapshot
        plan=snapshot.get('production_plan')
        if plan:
            m=plan['metadata'];token=(tuple(m['identity']),m['frame_sequence'],m['path_version'])
            if token!=self.plan_token:
                self.plan_token=token
                geometry=snapshot.get('live_geometry') or {}
                points=np.vstack((plan['tcp_xyz_base_m'],np.asarray(geometry.get('points',[])).reshape(-1,3)))
                if len(points):
                    self.center=(points.min(axis=0)+points.max(axis=0))/2
                    self.scale=1.;size=np.ptp(self.project(points),axis=0)
                    self.scale=float(min(max(1,self.width()-120)/max(size[0],.05),max(1,self.height()-130)/max(size[1],.05)))
        collision=snapshot.get('collision')
        if collision and collision['metadata']['collision_id']!=self.collision_id:
            self.collision_id=collision['metadata']['collision_id']
            self.fit_collision_comparison()
        if first and 'surface' in snapshot:
            surface = snapshot['surface']
            self.center = surface.plane_to_base[:3, 3]+surface.plane_to_base[:3, :2]@surface.bounds_xy_m.mean(axis=1)
            self.fit_surface(surface)
        self.update()

    def focus_collision(self):
        collision=(self.snapshot or {}).get('collision')
        if collision:
            self.center=np.mean(collision['arrays']['hit_points_base_m'],axis=0)
            self.scale=1800.
            self.update()

    def fit_collision_comparison(self):
        """Keep both the frozen and predicted object in view by default."""
        collision=(self.snapshot or {}).get('collision')
        if not collision:return
        arrays=collision['arrays']
        points=np.vstack([np.asarray(arrays[name]).reshape(-1,3) for name in
            ('object_points_exposure_base_m','checked_points_base_m','hit_points_base_m')])
        if not len(points):return
        self.center=(points.min(axis=0)+points.max(axis=0))/2
        self.scale=1.
        size=np.ptp(self.project(points),axis=0)
        self.scale=float(min(max(1,self.width()-160)/max(size[0],.05),
            max(1,self.height()-180)/max(size[1],.05),1800.))
        self.update()

    def fit_surface(self, surface=None):
        surface=surface or (self.snapshot or {}).get('surface')
        if surface is None:return
        self.center=surface.plane_to_base[:3,3]+surface.plane_to_base[:3,:2]@surface.bounds_xy_m.mean(axis=1)
        local=np.array([[x,y,0] for x in surface.bounds_xy_m[0] for y in surface.bounds_xy_m[1]])
        corners=local@surface.plane_to_base[:3,:3].T+surface.plane_to_base[:3,3]
        self.scale=1.
        size=np.ptp(self.project(corners),axis=0)
        self.scale=float(min(max(1,self.width()-120)/max(size[0],.05),max(1,self.height()-130)/max(size[1],.05)))
        self.update()

    def project(self, points):
        co, si = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        rotation = np.array([[co, -si, 0], [si*sp, co*sp, cp], [-si*cp, -co*cp, sp]])
        p = (np.asarray(points)-self.center)@rotation.T
        return np.c_[self.width()/2+p[:,0]*self.scale, self.height()/2-p[:,1]*self.scale]

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), QtGui.QColor('#101923'))
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        def line(a, b, color, width=2):
            x, y = self.project([a,b]);p.setPen(QtGui.QPen(QtGui.QColor(color), width))
            p.drawLine(QtCore.QPointF(*x), QtCore.QPointF(*y))
        def dots(points, color, size):
            if not len(points):return
            p.setPen(QtGui.QPen(QtGui.QColor(color), size))
            for point in self.project(points):
                p.drawPoint(QtCore.QPointF(*point))
        def label_at(point,text,color):
            xy=self.project([point])[0]
            if not 0<=xy[0]<=self.width() or not 0<=xy[1]<=self.height():return
            p.setPen(QtGui.QColor(color))
            p.drawText(QtCore.QPointF(float(np.clip(xy[0]+8,8,max(8,self.width()-240))),
                float(np.clip(xy[1]-12,100,max(100,self.height()-36)))),text)
        snap = self.snapshot or {}
        path_preview=snap.get('path_preview')
        production_plan=snap.get('production_plan')
        if path_preview:
            dots(sample(path_preview['scene'], 4500), '#657482', 2)
        elif 'scene' in snap:dots(sample(snap['scene'], 4500), '#657482', 2)
        collision=snap.get('collision')
        if collision:
            a=collision['arrays']
            for segment in a.get('robot_edges_base_m',()):line(*segment,'#5e758b',1)
            dots(sample(a['object_points_exposure_base_m'],1500),'#a996df',2)
            dots(sample(a['checked_points_base_m'],2000),'#42dfed',3)
            for segment in a.get('collision_hull_edges_base_m',()):line(*segment,'#ff845f',2)
            for segment in a.get('margin_hull_edges_base_m',()):line(*segment,'#e8bb55',1)
            dots(sample(a['hit_points_base_m'],800),'#ff354e',7)
            target_pose = collision_predicted_grasp(collision['metadata'])
            if target_pose is not None:
                for start,end in grasp_lines(target_pose,collision['metadata']['gripper_width_m']):
                    line(start,end,'#ef75ff',3)
                label_at(target_pose[:3,3],'预测时刻目标TCP（紫）','#ef75ff')
            predicted=collision_predicted_box(snap.get('live_geometry'),collision['metadata'])
            if predicted:
                for a,b in box_lines(predicted):line(a,b,'#65f296',2)
        if 'surface' in snap:
            s = snap['surface']; lo, hi = s.bounds_xy_m.T
            corners = np.array([[lo[0],lo[1],0],[hi[0],lo[1],0],[hi[0],hi[1],0],[lo[0],hi[1],0]])
            corners = corners@s.plane_to_base[:3,:3].T+s.plane_to_base[:3,3]
            for i in range(4):line(corners[i],corners[(i+1)%4], '#458c78')
            # Show the exact configured s=p·direction freeze/entry/follow boundaries.
            direction = self.config.direction
            fixed_position = self.config.raw.get('freeze_position_upstream_m')
            distance = self.config.raw['freeze_upstream_distance_m']
            if fixed_position is not None:
                distance = fixed_position
            elif (self.config.raw.get('synchronized_adaptive_freeze') and
                  self.config.raw.get('planning_reserve_mode') == 'measured'):
                distance = self.config.raw['synchronized_max_freeze_upstream_distance_m']
            boundaries = [(self.config.follow_bounds[0]-distance,'#58cbd6',
                           None if fixed_position is None else f'冻结 {distance*100:g} cm'),
                          (self.config.follow_bounds[0],'#ffcb65',None),
                          (self.config.follow_bounds[1],'#ff7474',None)]
            if fixed_position is not None:
                cutoff = fixed_position-self.config.raw['freeze_position_tolerance_m']
                boundaries.append((self.config.follow_bounds[0]-cutoff,'#96793e',f'取帧截止 {cutoff*100:g} cm'))
            for along, color, label in boundaries:
                vertices=[]
                for a,b in zip(corners,np.roll(corners,-1,axis=0)):
                    da,db=a@direction-along,b@direction-along
                    if da*db<0:vertices.append(a+(b-a)*da/(da-db))
                if len(vertices)==2:
                    line(*vertices,color,2)
                    if label:label_at(np.mean(vertices,axis=0),label,color)
        live = snap.get('live_geometry')
        frozen = snap.get('frozen')
        if path_preview:
            live = frozen = None  # Never overlay a newer target on the saved path exposure.
        geometry = path_preview['geometry'] if path_preview else frozen or live
        if live and frozen:dots(sample(live['points'],1200),'#b8c7ff',3)
        for track in snap.get('track_labels',[]):
            dots([track['position']],'#ffb776' if track['reasons'] else '#65f296',7)
        if geometry:
            dots(sample(geometry['points'], 2500), '#42dfed', 3)
            for a,b in box_lines(geometry):line(a,b,'#65f296')
            pose=display_grasp(geometry,self.candidate,production_plan)
            for a,b in grasp_lines(pose,self.config.settings.open_width_m):line(a,b,'#ff9b54',3)
            dots([pose[:3,3]],'#e498ff',7)
            label_at(pose[:3,3],'冻结曝光OBB / 已选抓姿' if production_plan else '曝光OBB / 短轴候选','#ff9b54')
        if production_plan:
            path=production_plan['tcp_xyz_base_m'];times=production_plan['times'];meta=production_plan['metadata']
            for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
                t=(times[i]+times[i+1])/2
                color=('#59ef78' if t<meta['contact_elapsed_s'] else
                       '#42dfed' if t<meta['follow_end_elapsed_s'] else '#ffbf32')
                line(a,b,color,4)
            dots(sample(production_plan['predicted_contact_points_base_m'],1200),'#c77bff',2)
            contact=production_plan['contact_grasp_base']
            dots([contact[:3,3]],'#ffffff',8)
            for a,b in grasp_lines(contact,self.config.settings.open_width_m):line(a,b,'#ffffff',2)
        if path_preview:
            path=path_preview['tcp_xyz']
            for a,b in zip(path[:-1],path[1:]):line(a,b,'#ffea4a',3)
            dots([path[0]],'#76ffad',9);dots([path[-1]],'#ffea4a',9)
            if 'predicted_object_points' in path_preview:
                dots(sample(path_preview['predicted_object_points'],1200),'#df8aff',3)
                dots([path_preview['contact_xyz']],'#ffffff',9)
                for a,b in grasp_lines(path_preview['contact_grasp'],self.config.settings.open_width_m):
                    line(a,b,'#ffffff',2)
        if frozen:
            origin=frozen['track_position_base_m'];dots([origin],'#ffe07e',10)
            end=origin+frozen['velocity_base_m_s']*.5;line(origin,end,'#ffe07e',3)
            for axis,color in enumerate(('#ff7676','#7ddd9b','#77aaff')):
                line(pose[:3,3],pose[:3,3]+pose[:3,axis]*.075,color,2)
            candidate=next((c for c in frozen['candidates'] if c['window_fits']),frozen['candidates'][0])
            contact=candidate['contact_base_m'];dots([contact],'#ffffff',8)
            line(frozen['grasps'][0][:3,3],contact,'#91a5bb',1)
            line(contact,candidate['follow_end_base_m'],'#ffffff',2)
        speed=None if path_preview else snap.get('frozen_speed')
        if speed and speed.get('position_base_m') is not None and (
                not collision or speed['identity']==collision['metadata']['target_identity']):
            origin=np.asarray(speed['position_base_m']);dots([origin],'#ffe050',11)
            velocity=speed.get('velocity_base_m_s')
            if velocity is not None:line(origin,origin+np.asarray(velocity)*.5,'#ffe050',3)
            p.setPen(QtGui.QColor('#ffe050'))
            if speed.get('along_belt_m_s') is not None:
                p.drawText(15,75,f'黄色冻结测速点 ID{speed["identity"][2]} · {speed["along_belt_m_s"]*100:.2f} cm/s')
        p.setPen(QtGui.QColor('#c8d7e5'))
        p.drawText(15,25,'机器人基座坐标系 · 左键拖动旋转 / 滚轮缩放')
        if collision:
            m=collision['metadata'];t=m.get('trajectory_elapsed_s')
            when='' if t is None else f' · 轨迹 {t:.3f}s'
            p.drawText(15,50,'预测碰撞：'+m['geometry']+when)
            p.drawText(15,self.height()-18,'灰：预测机器人 · 橙黄凸包：碰撞几何 · 紫线：预测目标TCP · 橙夹爪：冻结抓姿')
        elif production_plan:p.drawText(15,self.height()-18,'绿：接近 · 青线：闭爪跟随 · 黄：减速 · 紫点：预测工件 · 白：接触抓姿')
        elif path_preview:p.drawText(15,self.height()-18,'黄：TCP轨迹 · 青：曝光工件 · 紫：预测接触工件 · 白：预测接触夹爪')
        else:p.drawText(15,self.height()-18,snap.get('geometry_legend','青色：工件点云  绿色：OBB  橙色：候选夹爪  黄色：冻结测速'))
        p.end()

    def mousePressEvent(self, event):self.last=event.pos()
    def mouseMoveEvent(self, event):
        if self.last is not None and event.buttons() & QtCore.Qt.LeftButton:
            delta=event.pos()-self.last;self.yaw+=delta.x()*.008;self.pitch+=delta.y()*.008
            self.last=event.pos();self.update()
    def mouseReleaseEvent(self, event):self.last=None
    def wheelEvent(self, event):
        self.scale=float(np.clip(self.scale*np.exp(event.angleDelta().y()/900),80,5000));self.update()


class CameraWorker(threading.Thread):
    def __init__(self, config, calibration, directory, *, replay_stem=None, read_robot_feedback=False, trajectory_preview=False):
        super().__init__(daemon=True)
        self.config,self.calibration,self.directory=config,calibration,directory
        self.stop=threading.Event();self.rearm=threading.Event()
        self.lock=threading.Lock();self.latest={'status':'正在连接相机'};self.freeze_count=0
        self.replay_stem=replay_stem
        self.last_log_s=0.
        self.read_robot_feedback=read_robot_feedback
        self.trajectory_preview=trajectory_preview
        self.can_connections=0
        self.camera_connections=0

    def publish(self, value):
        with self.lock:self.latest=value
        if time.monotonic()-self.last_log_s>=.5:
            self.last_log_s=time.monotonic()
            row={k:value.get(k) for k in ('status','stage','source_gap_s','processing_ms',
                'geometry_wall_s','geometry_status','qualified_count','track_labels',
                'sampling_warning','meter_diagnostics','depth_admission','replay','path_status')}
            row.update({k:value.get(k) for k in ('scene_source','scene_age_s','scene_filter_error',
                'scene_filter_report','robot_feedback_available','robot_feedback_error')})
            with (self.directory/'diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row,ensure_ascii=False,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a))+'\n')

    def run_replay(self):
        from mujoco_conveyor_sorting.runtime.types import RGBDFrame,CameraIntrinsics,SampleHeader
        stem=str(self.replay_stem);meta=json.loads(Path(stem+'.json').read_text())
        frame=RGBDFrame(header=SampleHeader(**meta['header']),depth_m=np.load(stem+'_depth_m.npy',allow_pickle=False),
            rgb=cv2.cvtColor(cv2.imread(stem+'_rgb.png'),cv2.COLOR_BGR2RGB),
            depth_intrinsics=CameraIntrinsics(**meta['depth_intrinsics']),
            color_intrinsics=CameraIntrinsics(**meta['color_intrinsics']),
            camera_to_world_cv=np.asarray(meta['depth_to_base_cv']),serial_number=meta['serial_number'],
            calibration_id=meta['calibration_id'],timestamp_metadata=meta['timestamp_metadata'])
        processor=CameraPreview(self.config,self.calibration)
        try:
            # Each file is ONE historical exposure. Never repeat it as new
            # independent samples or manufacture a speed from a still image.
            snap=processor.process(frame,np.asarray(meta['depth_to_color_cv']),robot_state=meta.get('robot_state'))
            if processor.future is not None:
                processor.future.result(timeout=15.)
                processor.finish_geometry()
            if processor.scene_future is not None:
                processor.scene_future.result(timeout=30.)
                processor.finish_scene()
            if processor.filtered_scene is not None:
                filtered=processor.filtered_scene
                if 'points' in filtered:
                    snap.update(scene=filtered['points'],scene_source=filtered['source'],
                        scene_age_s=0.,scene_filter_report=filtered['report'])
                else:snap['scene_filter_error']=filtered.get('error')
            snap['live_geometry']=None if processor.geometry is None else processor.geometry.get('geometry')
            snap.update(replay=True,replay_source=stem,frozen=None,d2c=np.asarray(meta['depth_to_color_cv']),
                geometry_wall_s=None if processor.geometry is None else processor.geometry['wall_s'],
                status='历史真实曝光回放 · 单帧不能测速；'+
                    ('实时几何提取尚未成功' if processor.geometry is None else processor.geometry['status']))
            self.publish(snap)
            while not self.stop.wait(.1):
                if self.rearm.is_set():self.rearm.clear() # a historical file cannot become a fresh exposure
        finally:processor.close()

    def run(self):
        if self.replay_stem is not None:
            try:self.run_replay()
            except Exception as error:self.publish({'status':f'历史曝光回放失败：{error}','error':str(error),'replay':True})
            return
        while not self.stop.is_set():
            camera=processor=feedback=None
            try:
                cfg=dict(self.config.raw['camera'])
                cfg.update(enable_color=True,require_color=True,timestamp_source='device',
                    serial_number=self.calibration['serial_number'],calibration_id=self.calibration['calibration_id'],
                    camera_to_world_cv=self.calibration['depth_to_base_cv'])
                camera=OrbbecCameraBackend(cfg);camera.start()
                self.camera_connections+=1
                camera.read_latest(timeout_s=3.,wait_for_fresh=True)
                d2c=camera.get_depth_to_color_cv();processor=CameraPreview(self.config,self.calibration,
                    trajectory_output=self.directory/'paths' if self.trajectory_preview else None)
                last=None;saved=None
                if self.read_robot_feedback:
                    # The existing adapter installs a hard send prohibition
                    # for execute=False, including accidental SDK sends.
                    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
                    try:
                        feedback=Hardware(self.config.raw['hardware']['can_channel'],
                            self.config.raw['hardware']['firmware'],False,
                            self.config.raw['hardware']['joint_receive_offset_s'])
                        feedback.start()
                        self.can_connections+=1
                    except Exception as error:
                        if feedback is not None:feedback.close()
                        feedback=None
                        print('只读反馈不可用，未发送CAN：',error,flush=True)
                reader=getattr(camera,'read_next',None) or camera.read_latest
                while not self.stop.is_set():
                    if self.rearm.is_set():processor.reset();self.rearm.clear()
                    frame=reader(timeout_s=1.,after_sequence=last,wait_for_fresh=True)
                    last=frame.header.sequence
                    state=None;feedback_error=None
                    if feedback is not None:
                        try:state=feedback.snapshot_nowait()
                        except Exception as error:feedback_error=str(error)
                    if state is not None:
                        from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
                        from nero_core.two_second.measured_rgbd import depth_capture_timing
                        try:
                            capture=depth_capture_timing(frame)['depth_sample_monotonic_estimate']
                            stamps=[mapped_stamp(state,k) for k in ('joint','pose')]
                            if any(t is None or abs(t-capture)>.2 for t in stamps):
                                raise ValueError('机械臂反馈与当前曝光未配对，暂不做自体过滤')
                        except (ValueError,RuntimeError) as error:state=None;feedback_error=str(error)
                    snap=processor.process(frame,d2c,robot_state=state)
                    if feedback_error:snap['robot_feedback_error']=feedback_error
                    snap.update(d2c=d2c,updated_s=time.monotonic(),
                        camera_capture=camera.get_capture_diagnostics())
                    frozen=snap.get('frozen')
                    if frozen is not None and frozen['identity']!=saved:
                        self.freeze_count+=1;saved=frozen['identity']
                        stem=self.directory/f'freeze_{self.freeze_count:03d}'
                        def serial(v):
                            if isinstance(v,np.ndarray):return v.tolist()
                            if isinstance(v,np.generic):return v.item()
                            raise TypeError(type(v).__name__)
                        report={k:v for k,v in frozen.items() if k not in ('rgb_frame','points','d2c')}
                        stem.with_suffix('.json').write_text(json.dumps(report,default=serial,ensure_ascii=False,indent=2)+'\n')
                        np.savez_compressed(stem.with_suffix('.npz'),points=frozen['points'],scene=snap['scene'])
                        cv2.imwrite(str(stem.with_suffix('.png')),cv2.cvtColor(frozen['rgb_frame'].rgb,cv2.COLOR_RGB2BGR))
                        print('已保存真实曝光冻结快照:',stem,flush=True)
                    self.publish(snap)
            except Exception as error:
                self.publish({'status':f'相机不可用：{error}；5秒后重试', 'error':str(error),
                              'updated_s':time.monotonic()})
                print('相机预览:',error,flush=True)
            finally:
                if camera is not None:camera.close()
                if processor is not None:processor.close()
                if feedback is not None:feedback.close()
            self.stop.wait(5.)


class PreviewWindow(QtWidgets.QWidget):
    def __init__(self, worker, config):
        super().__init__();self.worker,self.config=worker,config
        self.setWindowTitle('相机点云与抓姿预览 · 不发送运动命令')
        if hasattr(worker,'run_directory'):
            self.setWindowTitle('抓取同源相机与碰撞预测 · 只读显示')
        self.resize(1420,880)
        self.setStyleSheet('QWidget{background:#101923;color:#deebf7;font:14px "Noto Sans CJK SC";} QPushButton,QComboBox{background:#23364a;padding:8px;border-radius:5px;}')
        outer=QtWidgets.QVBoxLayout(self)
        self.status=QtWidgets.QLabel('正在打开相机……');self.status.setWordWrap(True);outer.addWidget(self.status)
        self.speed_status=QtWidgets.QLabel('实时卡尔曼：等待观测；冻结速度：尚未冻结')
        self.speed_status.setWordWrap(True);outer.addWidget(self.speed_status)
        row=QtWidgets.QHBoxLayout();outer.addLayout(row,1)
        images=QtWidgets.QVBoxLayout();row.addLayout(images,4)
        self.rgb=QtWidgets.QLabel('实时 RGB：等待相机连接');self.rgb.setMinimumSize(440,270)
        self.frozen_rgb=QtWidgets.QLabel('冻结曝光：尚未满足冻结条件');self.frozen_rgb.setMinimumSize(440,270)
        for label in (self.rgb,self.frozen_rgb):label.setAlignment(QtCore.Qt.AlignCenter);images.addWidget(label)
        self.cloud=CloudView(config);row.addWidget(self.cloud,6)
        self.details=QtWidgets.QLabel('冻结条件：单个合格轨迹、测速跨度 ≥ 0.45秒、入口上游35厘米内、预留 ≥ 1.5秒。')
        self.details.setWordWrap(True);outer.addWidget(self.details)
        self.tracks=QtWidgets.QPlainTextEdit();self.tracks.setReadOnly(True);self.tracks.setMaximumHeight(115)
        outer.addWidget(self.tracks)
        controls=QtWidgets.QHBoxLayout();outer.addLayout(controls)
        rearm=QtWidgets.QPushButton('重新等待冻结（空格）');rearm.clicked.connect(worker.rearm.set);controls.addWidget(rearm)
        if hasattr(worker,'run_directory'):rearm.setEnabled(False)
        self.choice=QtWidgets.QComboBox();self.choice.addItems(['短轴抓姿 1','短轴抓姿 2（旋转180°）'])
        self.choice.currentIndexChanged.connect(self.change_candidate);controls.addWidget(self.choice)
        fit=QtWidgets.QPushButton('适配带面视图');fit.clicked.connect(lambda:self.cloud.fit_surface());controls.addWidget(fit)
        focus=QtWidgets.QPushButton('聚焦碰撞');focus.clicked.connect(self.cloud.focus_collision);controls.addWidget(focus)
        self.collision_choice=QtWidgets.QComboBox();self.collision_choice.setMinimumWidth(180)
        self.collision_choice.currentTextChanged.connect(lambda key:
            worker.select_collision(key) if hasattr(worker,'select_collision') else None)
        controls.addWidget(self.collision_choice)
        controls.addWidget(QtWidgets.QLabel('橙夹爪：曝光抓姿；紫线：预测目标抓姿；白色：预测接触抓姿'))
        notice=QtWidgets.QLabel('只读显示；关闭窗口只关闭画面，停止抓取请在抓取终端按Ctrl+C。碰撞图是预测叠加，不代表实拍撞击。')
        notice.setWordWrap(True);outer.addWidget(notice)
        QtWidgets.QShortcut(QtGui.QKeySequence('Space'),self,activated=worker.rearm.set)
        QtWidgets.QShortcut(QtGui.QKeySequence('Escape'),self,activated=self.close)
        self.timer=QtCore.QTimer(self);self.timer.timeout.connect(self.refresh);self.timer.start(120)
        self.last_display=None

    def change_candidate(self,index):self.cloud.candidate=index;self.refresh()

    def image(self,label,frame,d2c=None,frozen=None,tracks=(),roi=None,title='',production_plan=None):
        rgb=frame.rgb.copy()
        if roi is not None:
            x0,y0,x1,y1=map(int,roi);cv2.rectangle(rgb,(x0,y0),(x1,y1),(90,230,150),2)
        if tracks:
            from nero_core.two_second.measured_rgbd import project_to_rgb
            for track in tracks:
                base=np.asarray([track['position']]);optical=(base-frame.camera_to_world_cv[:3,3])@frame.camera_to_world_cv[:3,:3]
                uv,positive=project_to_rgb(optical,frame,d2c)
                if positive.all() and np.isfinite(uv).all() and np.max(np.abs(uv))<100000:
                    xy=tuple(np.rint(uv[0]).astype(int));color=(255,180,80) if track['reasons'] else (80,240,140)
                    cv2.circle(rgb,xy,12,color,3)
                    speed='warming' if track['velocity'] is None else f'{np.asarray(track["velocity"])@self.config.direction*100:.2f}cm/s'
                    cv2.putText(rgb,f'ID {track["id"]} {speed}',(xy[0]+12,xy[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        if title:cv2.putText(rgb,title,(16,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,240,180),2)
        if frozen:
            from nero_core.two_second.measured_rgbd import project_to_rgb
            segments=box_lines(frozen)
            pose=display_grasp(frozen,self.choice.currentIndex(),production_plan)
            jaws=grasp_lines(pose,self.config.settings.open_width_m)
            for group,color in ((segments,(80,245,135)),(jaws,(255,155,84))):
                for a,b in group:
                    base=np.array([a,b]);optical=(base-frame.camera_to_world_cv[:3,3])@frame.camera_to_world_cv[:3,:3]
                    uv,positive=project_to_rgb(optical,frame,d2c)
                    if positive.all() and np.isfinite(uv).all() and np.max(np.abs(uv))<100000:
                        cv2.line(rgb,tuple(np.rint(uv[0]).astype(int)),tuple(np.rint(uv[1]).astype(int)),color,3)
        rgb=np.ascontiguousarray(rgb)
        img=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
        label.setPixmap(QtGui.QPixmap.fromImage(img).scaled(label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def speed_hud(self,label,snap,frame,d2c,*,allow_marker=True,frozen_only=False):
        from .speed_overlay import speed_lines
        from .collision_view import project_base
        original=label.pixmap()
        if original is None or original.isNull():return
        pixmap=QtGui.QPixmap(original);p=QtGui.QPainter(pixmap)
        try:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            frozen=snap.get('frozen_speed')
            lines=speed_lines(snap.get('live_speed'),frozen,now=time.monotonic(),
                maximum_age_s=self.config.raw['maximum_observation_age_s'],historical=snap.get('replay',False))
            if frozen_only and frozen:
                lines=[lines[1],(f'冻结曝光 #{frozen["frame_sequence"]} · 黄色点为当时测速位置',(255,224,80))]
            p.fillRect(0,0,pixmap.width(),57,QtGui.QColor(0,0,0,205))
            p.setFont(QtGui.QFont('Noto Sans CJK SC',12))
            for i,(text,color) in enumerate(lines):
                p.setPen(QtGui.QColor(*color))
                p.drawText(8,22+i*25,p.fontMetrics().elidedText(text,QtCore.Qt.ElideRight,max(1,pixmap.width()-16)))
            if (allow_marker and frozen and frozen.get('position_base_m') is not None and
                    frozen.get('clock_epoch')==frame.header.clock_epoch and
                    frozen.get('calibration_id')==frame.calibration_id and
                    frozen.get('serial_number')==frame.serial_number):
                uv,visible=project_base([frozen['position_base_m']],frame,d2c)
                if visible[0]:
                    xy=uv[0]*[pixmap.width()/frame.rgb.shape[1],pixmap.height()/frame.rgb.shape[0]]
                    p.setPen(QtGui.QPen(QtGui.QColor('#ffe050'),3));p.setBrush(QtGui.QColor('#ffe050'))
                    p.drawEllipse(QtCore.QPointF(*xy),6,6)
                    p.drawText(QtCore.QPointF(max(8,min(pixmap.width()-110,float(xy[0])+10)),
                        max(75,min(pixmap.height()-8,float(xy[1])-10))),'冻结测速位置')
        finally:p.end()
        label.setPixmap(pixmap)

    def refresh(self):
        with self.worker.lock:snap=self.worker.latest
        status=snap.get('status','');frame=snap.get('frame');frozen=snap.get('frozen')
        if snap.get('path_status'):status+='；'+snap['path_status']
        if frozen and not snap.get('frozen_speed'):
            snap=dict(snap)
            snap['frozen_speed']={'identity':list(frozen['identity']),
                'source':frozen.get('velocity_source'),'configured_speed_m_s':frozen.get('configured_speed_m_s'),
                'frame_sequence':frozen['source_sequence'],'clock_epoch':frozen['clock_epoch'],
                'capture_s':frozen['capture_s'],'position_base_m':np.asarray(frozen['track_position_base_m']).tolist(),
                'velocity_base_m_s':np.asarray(frozen['velocity_base_m_s']).tolist(),
                'along_belt_m_s':float(np.asarray(frozen['velocity_base_m_s'])@self.config.direction),
                'calibration_id':frozen['rgb_frame'].calibration_id,'serial_number':frozen['rgb_frame'].serial_number,
                'sample_count':frozen['sample_count'],'span_s':frozen['span_s']}
        synthetic=frame is not None and 'synthetic' in frame.source.lower()
        stale=frame is not None and not synthetic and not snap.get('replay') and time.monotonic()-frame.header.received_time>1.
        if stale:status='画面已过期，等待新的真实曝光；'+status
        if snap.get('sampling_warning'):status+='；'+snap['sampling_warning']
        if snap.get('scene_filter_error'):status+='；原场景过滤未通过：'+snap['scene_filter_error']
        if snap.get('robot_feedback_error'):status+='；'+snap['robot_feedback_error']
        self.status.setText(status)
        from .speed_overlay import speed_lines
        speed_text=speed_lines(snap.get('live_speed'),snap.get('frozen_speed'),now=time.monotonic(),
            maximum_age_s=self.config.raw['maximum_observation_age_s'],historical=snap.get('replay',False))
        self.speed_status.setText('　'.join(f'<span style="color:rgb{color}">{html.escape(text)}</span>' for text,color in speed_text))
        live=snap.get('live_geometry')
        self.choice.setEnabled(not bool(snap.get('production_plan')))
        display_key=(id(snap),self.choice.currentIndex(),self.rgb.width(),self.cloud.width())
        redraw=display_key!=self.last_display
        self.last_display=display_key
        if frame is not None:
            if redraw:self.image(self.rgb,frame,snap.get('d2c'),tracks=snap.get('track_labels',()),
                roi=snap.get('roi_px'),title='SYNTHETIC OFFLINE FRAME' if synthetic else 'HISTORICAL FRAME' if snap.get('replay') else 'LIVE RGB / TRACKS')
        else:
            self.rgb.setText('实时 RGB：相机未连接');self.frozen_rgb.setText('尚无真实冻结曝光')
        if redraw:self.cloud.set_snapshot(snap)
        labels=snap.get('track_labels',())
        lines=[f'ID {t["id"]} · 沿带位置 {t["s_m"]:.3f}m · '+('测速已合格' if not t['reasons'] else ' / '.join(t['reasons'])) for t in labels]
        admission=snap.get('depth_admission') or {}
        for row in admission.get('decisions',[]):
            if not row['accepted']:
                heights=row.get('height_range_m')
                measured='' if heights is None else f'，实测离带高度{np.round(np.asarray(heights)*1000,1)}mm'
                lines.append('未分配物块ID：'+row['reason']+measured)
        diag=snap.get('meter_diagnostics',{})
        gap=snap.get('source_gap_s');fps='' if not gap or gap<=0 else f'有效采样 {1/gap:.1f}Hz；'
        source=(f'原有过滤后的场景点云（曝光间隔{snap.get("scene_age_s",0):.2f}秒）' if snap.get('scene_source')=='original_RobotModel.install_scene'
                else '环境仅使用实测带面点云（已排除非带面场景点）' if snap.get('scene_source')=='measured_belt_surface_only'
                else '原始场景参考（没有可用的完整过滤结果）')
        self.tracks.setPlainText(f'{source}；{fps}处理 {snap.get("processing_ms",0):.0f}ms；几何计算 {1000*(snap.get("geometry_wall_s") or 0):.0f}ms\n'+'\n'.join(lines))
        if frozen:
            if redraw:self.image(self.frozen_rgb,frozen['rgb_frame'],frozen['d2c'],frozen,title='FROZEN EXPOSURE',production_plan=snap.get('production_plan'))
            v=frozen['velocity_base_m_s'];pos=frozen['track_position_base_m'];pose=display_grasp(frozen,self.choice.currentIndex(),snap.get('production_plan'))
            age=time.monotonic()-frozen['capture_s']
            if frozen.get('velocity_source')=='operator_fixed_belt_speed':
                velocity_text=f'用户设定带速 {frozen["configured_speed_m_s"]*100:.2f} cm/s；目标跟踪 {frozen["sample_count"]} 帧'
            else:
                velocity_text=f'视觉速度 XYZ = {np.round(v*100,2)} cm/s；测速跨度 {frozen["span_s"]:.2f}秒 / {frozen["sample_count"]}点'
            fits=[c['approach_s'] for c in frozen['candidates'] if c['window_fits']]
            window=f'空间窗口可容纳时长：{fits}秒（未做IK/避障）' if fits else '所有时长均超出当前空间窗口'
            self.details.setText(f'冻结曝光 #{frozen["source_sequence"]} / 轨迹 {frozen["identity"][2]}，距今 {age:.1f}秒；'
                f'{velocity_text}\n'
                f'冻结测速位置 XYZ = {np.round(pos,4)} m；候选TCP = {np.round(pose[:3,3],4)} m\n{window}。'
                '橙色夹爪仅表示方向，开口按配置100mm绘制，尚未选出可执行抓姿。')
        else:
            if live is not None and redraw:
                self.image(self.frozen_rgb,live['frame'],live['d2c'],live,
                    title='PRODUCTION FROZEN EXPOSURE' if snap.get('frozen_speed') else 'LIVE GEOMETRY / NOT FROZEN',
                    production_plan=snap.get('production_plan'))
            elif redraw and frame is not None and live is None:
                self.frozen_rgb.clear();self.frozen_rgb.setText(snap.get('geometry_status','等待实时几何提取'))
            self.details.setText('未冻结也持续显示检测位置、实时OBB和抓姿候选。冻结仍要求：单个合格轨迹、测速跨度≥0.45秒、入口上游35厘米内、预留≥1.5秒。')
        choices=snap.get('collision_choices',[])
        if choices!=[self.collision_choice.itemText(i) for i in range(self.collision_choice.count())]:
            current=snap.get('selected_collision',self.collision_choice.currentText())
            self.collision_choice.blockSignals(True);self.collision_choice.clear();self.collision_choice.addItems(choices)
            self.collision_choice.setCurrentIndex(choices.index(current) if current in choices else len(choices)-1)
            self.collision_choice.blockSignals(False)
        collision=snap.get('collision')
        if collision:
            m=collision['metadata'];f=collision['frame'];a=collision['arrays']
            if redraw and f is not None and f.rgb is not None:
                from .collision_view import overlay_collision
                rgb=overlay_collision(f.rgb,f,collision['d2c'],a)
                image=QtGui.QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QtGui.QImage.Format_RGB888).copy()
                self.frozen_rgb.setPixmap(QtGui.QPixmap.fromImage(image).scaled(self.frozen_rgb.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
            context=m.get('geometry_context',{}); candidate=context.get('candidate_index')
            index='未知' if candidate is None else str(candidate+1)
            source_index=context.get('input_grasp_index')
            if source_index is not None:index+=f'（输入抓姿 {source_index+1}）'
            t=m.get('trajectory_elapsed_s');elapsed='未知' if t is None else f'{t:.3f}s'
            self.details.setText(f'预测碰撞 {m["collision_id"]} · 候选 {index} · {m["geometry"]} / {m["robot_body"]}\n'
                f'轨迹时刻 {elapsed} · 曝光 #{m["frame_sequence"]} · 目标 {m["target_identity"]} · '
                f'碰撞点 {m["hit_count"]} 个：原凸包内部 {m["inside_nominal_hull_count"]}，仅{float(m["collision_margin_m"])*1000:g}mm余量内 {m["margin_only_count"]}\n'
                f'首个预测触发点 XYZ = {np.round(np.asarray(m["first_hit_base_m"])*1000,2)} mm（基座系）。'
                '红点和机器人姿态来自规划预测；左上画面与下方原始曝光分别保留。')
        speed=snap.get('frozen_speed')
        if speed and not collision and snap.get('shared_run'):
            span=speed.get('span_s');span_text='未知' if span is None else f'{span:.3f}'
            if speed.get('source')=='operator_fixed_belt_speed':
                self.details.setText(f'本次用户设定带速 {speed["configured_speed_m_s"]*100:.2f} cm/s · '
                    f'ID{speed["identity"][2]} · 曝光 #{speed["frame_sequence"]}；'
                    '黄色点是冻结时的观测位置，规划不使用视觉测速值。')
            else:
                self.details.setText(f'已冻结 ID{speed["identity"][2]} · 曝光 #{speed["frame_sequence"]} · '
                    f'{speed.get("sample_count","未知")}帧 / {span_text}秒；黄色点是冻结时的测速位置，不是当前物体位置。')
        if snap.get('path_preview'):
            path=snap['path_preview'];report=path['report']
            self.details.setText(f'黄色 TCP 曲线：曝光 #{report["frame_sequence"]}，模型规划 {report["planning_wall_s"]:.2f} 秒。'
                f'实际发送截止余量 {report["actual_remaining_send_lead_s"]:.2f} 秒。'
                '起点为等待位假设；规划时钟冻结、未读取机械臂反馈，不能直接执行。')
        if snap.get('production_plan') and not collision:
            m=snap['production_plan']['metadata']
            historical=m['meaning']=='historical_dispatched_commands_not_full_planned_path'
            label=('历史实际发送指令的 TCP 回放（可能不完整）' if historical else '抓取程序规划输出')
            note=('路径由历史发送关节指令与标称 TCP 正运动学计算，并非机械臂实际反馈。' if historical else
                  '轨迹为规划关节曲线的正运动学结果；是否已发送和机械臂实际位置以任务状态和反馈为准。')
            self.details.setText(f'{label} · ID{m["identity"][2]} · 曝光 #{m["frame_sequence"]} · 版本 {m["path_version"]}\n'
                f'先原位旋转 {m.get("rotate_first_s",0.):.2f}s，再接近/同步 {m["contact_elapsed_s"]-m.get("rotate_first_s",0.):.2f}s；闭爪跟随 {m["follow_end_elapsed_s"]-m["contact_elapsed_s"]:.2f}s，'
                f'减速 {m["path_duration_s"]-m["follow_end_elapsed_s"]:.2f}s。\n'
                f'{note}')
        if redraw:
            if frame is not None:self.speed_hud(self.rgb,snap,frame,snap.get('d2c'))
            frozen_frame=(collision['frame'] if collision else frozen['rgb_frame'] if frozen else live['frame'] if live else None)
            frozen_d2c=(collision['d2c'] if collision else frozen['d2c'] if frozen else live['d2c'] if live else None)
            if frozen_frame is not None:
                same_target=not collision or not speed or speed['identity']==collision['metadata']['target_identity']
                if same_target:self.speed_hud(self.frozen_rgb,snap,frozen_frame,frozen_d2c,frozen_only=True)

    def closeEvent(self,event):self.worker.stop.set();event.accept()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    from ..move_to_wait import default_config
    p.add_argument('--config',type=Path)
    p.add_argument('--output',type=Path,default=Path('runs/live_camera_grasp')/
        datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S'))
    p.add_argument('--seconds',type=float,default=0,help='0 means keep open')
    p.add_argument('--screenshot',type=Path,help='Save the actual displayed window at exit')
    p.add_argument('--replay-stem',type=Path,help='Offline historical RGB-D file stem; opens no devices')
    p.add_argument('--read-robot-feedback',action='store_true',help='Read existing CAN feedback for original self/scene filters; every CAN send is forbidden')
    p.add_argument('--follow-run',type=Path,help='只读跟随抓取目录，共用相机画面和碰撞记录；不连接设备')
    p.add_argument('--trajectory-preview',action='store_true',help='仅相机，使用假设等待位在点云上显示模型规划轨迹')
    p.add_argument('--fixed-belt-speed-m-s',type=float,help='轨迹预览的已知带速，单位 m/s')
    args=p.parse_args(argv)
    if args.follow_run and (args.replay_stem or args.read_robot_feedback):
        p.error('--follow-run 不能与独立相机回放或CAN读取选项同时使用')
    if args.trajectory_preview and (args.follow_run or args.replay_stem or args.read_robot_feedback):
        p.error('--trajectory-preview 仅支持独立相机且不连接 CAN')
    config=CellConfig.load(args.config or default_config())
    if args.fixed_belt_speed_m_s is not None:
        from .velocity_source import with_fixed_belt_speed
        config=with_fixed_belt_speed(config,args.fixed_belt_speed_m_s)
    calibration=None if args.follow_run else load_runtime_depth_calibration(config.raw['camera_calibration_file'])
    args.output.mkdir(parents=True,exist_ok=True)
    # OpenCV ships Qt plugin paths for its own bundled Qt; use PyQt's plugins here.
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    app=QtWidgets.QApplication([])
    if args.follow_run:
        from .run_preview import RunPreviewWorker
        worker=RunPreviewWorker(args.follow_run,direction=config.direction)
    else:
        worker=CameraWorker(config,calibration,args.output,replay_stem=args.replay_stem,
            read_robot_feedback=args.read_robot_feedback,trajectory_preview=args.trajectory_preview)
    window=PreviewWindow(worker,config)
    window.show();worker.start()
    signal.signal(signal.SIGINT,lambda *_:window.close())
    signal.signal(signal.SIGTERM,lambda *_:window.close())
    def save():
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            window.grab().save(str(args.screenshot))
    app.aboutToQuit.connect(save)
    if args.seconds>0:QtCore.QTimer.singleShot(int(args.seconds*1000),window.close)
    print(('只读共享抓取画面，不连接设备；' if args.follow_run else '离线历史曝光，不连接设备；' if args.replay_stem else
        '相机与被动机器人反馈；' if args.read_robot_feedback else '仅打开相机；')+
        '机械臂/夹爪命令=0。空格重新等待冻结，Esc退出。',flush=True)
    code=app.exec_();worker.stop.set();worker.join(timeout=6.)
    (args.output/'session.json').write_text(json.dumps({'can_connections':worker.can_connections,'robot_commands_sent':0,
        'camera_connections':worker.camera_connections,
        'freeze_count':worker.freeze_count,'last_status':worker.latest.get('status'),
        'camera_only':not args.follow_run and not args.read_robot_feedback,'replay_stem':None if args.replay_stem is None else str(args.replay_stem),
        'follow_run':None if args.follow_run is None else str(args.follow_run),
        'worker_stopped':not worker.is_alive()},ensure_ascii=False,indent=2)+'\n')
    return code


if __name__=='__main__':raise SystemExit(main())
