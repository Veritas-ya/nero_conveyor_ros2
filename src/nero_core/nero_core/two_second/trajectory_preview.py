"""Camera-only, counterfactual trajectory preview; never opens CAN or sends commands."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from .collision_capture import frame_metadata
from .diagnostics import json_value
from .planning import MeasuredPlanner, calibrated_model
from .velocity_source import prediction_velocity, speed_report
from .vision import MeasuredTarget, obb_grasps


class TrajectoryPreview:
    def __init__(self, config, output):
        self.config = config
        self.output = Path(output)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-trajectory-preview')
        self.future = None
        self.generation = 0
        self.last_submitted = -float('inf')
        self.result = None
        self.status = '等待相机关联到工件'
        self.planner = None  # Owned exclusively by the single planning worker.
        self.archive_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-path-archive')
        self.archives = []

    def reset(self):
        self.generation += 1
        self.result = None
        self.status = '等待新的相机观测'

    def close(self):
        self.generation += 1
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.archive_pool.shutdown(wait=True)

    def flush_archives(self):
        for future in self.archives:
            future.result()

    def submit(self, done, gate):
        from .preview import qualified_tracks
        if self.future is not None or done.get('geometry') is None or done.get('capture') is None:
            return
        tracks = qualified_tracks(self.config, done['peer'])
        if len(tracks) != 1 or time.monotonic()-self.last_submitted < 1.:
            return
        self.last_submitted = time.monotonic()
        self.status = '假设等待位，正在计算模型轨迹'
        self.future = self.pool.submit(self._calculate, done, tracks[0], gate, self.generation)

    def poll(self):
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if result['generation'] == self.generation:
                    self.result = result
                    self.status = result['status']
            except Exception as error:
                self.status = f'轨迹预览未生成：{type(error).__name__}: {error}'
            self.future = None
        for future in self.archives:
            if future.done() and future.exception() is not None:
                self.status = f'轨迹存档失败：{future.exception()}'
        return self.result

    def _calculate(self, done, track, gate, generation):
        started = time.monotonic()
        d = dict(done['detection'])
        frame, d2c = done['geometry']['frame'], done['geometry']['d2c']
        capture = done['capture']
        points = np.asarray(d['object_points'])
        observed = np.asarray(track.observed_position_base)
        if np.linalg.norm(np.median(points, axis=0)[:2]-observed[:2]) > gate:
            raise ValueError('工件点云与跟踪目标不匹配')
        grasps, box = obb_grasps(points, d['belt_region']['plane_to_base'], self.config.canonical_tcp)
        if self.planner is None:
            self.planner = MeasuredPlanner(self.config, model=calibrated_model(self.config), clock=time.monotonic)
        planner, model = self.planner, self.planner.model
        model.set_state(self.config.standby, self.config.settings.open_width_m)
        flange = model.flange()
        state = {'joints': self.config.standby.tolist(), 'width_m': self.config.settings.open_width_m,
                 'pose': np.r_[flange[:3, 3], Rotation.from_matrix(flange[:3, :3]).as_euler('xyz')].tolist()}
        velocity = prediction_velocity(self.config, track.velocity_base_m_s, d['plane']['normal_base'])
        d.update(frame=frame, depth_to_color_cv=d2c, scene_state=state, obb=box)
        peer = done['peer']
        identity = (peer.clock_epoch, peer.track_generation, track.track_id)
        target = MeasuredTarget(identity, capture, frame.header.received_time, frame.header.source_time,
            peer.clock_epoch, frame.header.sequence, np.median(points, axis=0), velocity,
            tuple(grasps), points, d, state, speed_report(self.config, track, velocity), observed)
        # Use the production measured-time replanning policy. A frozen exposure
        # clock conceals candidate cost and can deliver a stale intercept.
        wall_origin = time.monotonic()
        plan = planner.intercept(target, state, cancelled=lambda: generation != self.generation or
                                 time.monotonic()-wall_origin > 20.)
        planning_wall = time.monotonic()-wall_origin
        times = np.linspace(0., plan.path.duration, max(2, int(plan.path.duration*20)+1))
        joints, xyz = [], []
        for stamp in times:
            q = plan.path.position(stamp)
            model.set_state(q, self.config.settings.open_width_m)
            joints.append(q); xyz.append(model.tcp()[:3, 3].copy())
        from .preview import CameraPreview
        scene = CameraPreview.scene_cloud(frame)
        geometry = done['geometry']
        contact_xyz = np.asarray(plan.reference.contact_xyz)
        predicted_object_points = points + velocity * (plan.reference.contact_s-capture)
        contact_grasp = np.asarray(plan.selected_grasp).copy()
        contact_grasp[:3, 3] = contact_xyz
        report = {'mode':'camera_only_counterfactual', 'can_connections':0, 'robot_commands':0,
            'initial_state':'assumed_configured_standby_not_measured',
            'scene_self_filter':'not_applied_without_measured_robot_feedback',
            'timing':'real_monotonic_planning_and_result_preparation_no_hardware_delivery_certification',
            'actual_age_at_planning_start_s':wall_origin-capture,
            'actual_send_deadline_monotonic_s':plan.reference.start_s-self.config.raw['command_advance_s'],
            'planning_remaining_send_lead_s':plan.reference.start_s-self.config.raw['command_advance_s']-(wall_origin+planning_wall),
            'not_authorized_for_hardware_execution':True,
            'identity':identity,'frame_sequence':frame.header.sequence,
            'capture_monotonic_s':capture,'velocity_report':target.speed_report,
            'planning_wall_s':planning_wall,'total_wall_s':time.monotonic()-started,
            'trajectory_duration_s':plan.path.duration,'candidate_failures':plan.failures,
            'planning_components_s':plan.timings_s,'selected_grasp':plan.selected_grasp,
            'predicted_contact_xyz_m':contact_xyz,
            'camera':frame_metadata(frame,d2c)}
        stem = self.output / f'path_epoch_{peer.clock_epoch:03d}_frame_{frame.header.sequence:09d}'
        arrays = dict(times=times, joints=np.asarray(joints), tcp_xyz=np.asarray(xyz),
            scene=scene, object_points=points, frame_rgb=frame.rgb, frame_depth_m=frame.depth_m,
            grasps=np.asarray(grasps), selected_grasp=plan.selected_grasp,
            predicted_contact_xyz=contact_xyz, predicted_object_points=predicted_object_points,
            contact_grasp=contact_grasp)
        # Sampling/FK/scene preparation is part of result readiness; disk I/O
        # belongs to a separate worker and must not delay intercept planning.
        ready = time.monotonic()
        planner.measured_post_planning_s = max(getattr(planner, 'measured_post_planning_s', 0.),
                                              ready-(wall_origin+planning_wall))
        report.update(actual_remaining_send_lead_s=report['actual_send_deadline_monotonic_s']-ready,
                      result_ready_monotonic_s=ready, total_wall_s=ready-started,
                      result_preparation_wall_s=ready-(wall_origin+planning_wall))
        archive = self.archive_pool.submit(self._archive, stem, arrays, report)
        self.archives.append(archive)
        status = ('已生成模型轨迹（假设等待位、真实规划时钟，仅供分析）' if report['actual_remaining_send_lead_s'] >= 0
                  else '模型轨迹已生成，但结果准备后错过发送截止')
        return {'generation':generation,'status':status,
            'tcp_xyz':np.asarray(xyz),'scene':scene,'geometry':geometry,'frame':frame,
            'contact_xyz':contact_xyz,'predicted_object_points':predicted_object_points,'contact_grasp':contact_grasp,
            'd2c':d2c,'report':report,'file':str(stem),'archive_future':archive}

    @staticmethod
    def _archive(stem, arrays, report):
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_suffix('.npz'), **arrays)
        stem.with_suffix('.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2))
