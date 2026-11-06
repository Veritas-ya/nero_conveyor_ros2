"""Camera-only OBB preview. No robot device, sender or execution owner exists."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .reference import predict_entry_time, EntryNotReady
from .synchronized_reference import freeze_window_for_speed, admit_synchronized_observation
from .vision import obb_grasps
from .velocity_source import is_fixed, prediction_velocity, FIXED


def box_in_base(box, plane):
    """obb_grasps returns belt-frame dimensions; the renderer uses robot base."""
    result=dict(box)
    result['center_world_m']=(plane[:3,:3]@np.asarray(box['center_world_m'])+plane[:3,3]).tolist()
    result['axes_world_columns']=(plane[:3,:3]@np.asarray(box['axes_world_columns'])).tolist()
    result['coordinate_frame']='robot_base'
    return result


def freeze_candidate(config, track, peer, detection, capture_s, now_s, gate_radius):
    """Freeze one associated, fresh exposure using the production entry rule.

    This supplies geometric candidates only: no robot feedback is available,
    so neither self filtering nor IK/collision certification is claimed.
    """
    fixed=is_fixed(config)
    if (not peer.current or not track.confirmed or track.misses or
            (not fixed and (not track.speed_valid or track.velocity_base_m_s is None or
                            track.span_s < config.raw['minimum_speed_span_s']))):
        raise ValueError('测速尚未满足确认条件')
    if not 0 <= now_s-capture_s <= config.raw['maximum_observation_age_s']:
        raise ValueError('观测已过期，继续等待新曝光')
    velocity = prediction_velocity(config,track.velocity_base_m_s,detection['plane']['normal_base'])
    observed = np.asarray(track.observed_position_base, dtype=float)
    points = np.asarray(detection['object_points'])
    center = np.median(points, axis=0)
    if np.linalg.norm(center[:2]-observed[:2]) > gate_radius:
        raise ValueError('OBB 点云与测速轨迹未关联')
    if config.raw.get('intercept_strategy', 'classic') == 'synchronized':
        window = freeze_window_for_speed(config, velocity@config.direction)
        entry = admit_synchronized_observation(capture_s, observed@config.direction,
            velocity@config.direction, config.follow_bounds, now_s, window['distance_m'],
            minimum_upstream_distance_m=window.get('minimum_distance_m'))
    else:
        entry = predict_entry_time(capture_s, observed@config.direction,
            velocity@config.direction, config.follow_bounds[0], now_s,
            config.raw['planning_reserve_s'] + config.raw.get('prealign_orientation_s', 0.),
            maximum_upstream_distance_m=config.raw['freeze_upstream_distance_m'])
    grasps, box = obb_grasps(points, detection['belt_region']['plane_to_base'], config.canonical_tcp)
    box = box_in_base(box,np.asarray(detection['belt_region']['plane_to_base']))
    candidates = []
    for duration in config.raw['approach_duration_candidates_s']:
        contact = grasps[0][:3, 3]+velocity*(entry+duration-capture_s)
        end = contact+velocity*config.settings.full_plan_close_window_s
        along = np.array([contact@config.direction, end@config.direction])
        endpoints = np.stack((contact, end))
        fits = bool(along.min() >= config.follow_bounds[0]
            and along.max() <= config.follow_bounds[1]
            and np.all(endpoints >= config.workspace[:, 0])
            and np.all(endpoints <= config.workspace[:, 1]))
        candidates.append({'approach_s': duration, 'contact_base_m': contact,
                           'follow_end_base_m': end, 'window_fits': fits})
    return {'capture_s': capture_s, 'freeze_s': now_s, 'entry_s': entry,
        'source_sequence': peer.sequence, 'clock_epoch': peer.clock_epoch,
        'identity': (peer.clock_epoch, peer.track_generation, track.track_id),
        'track_position_base_m': observed.copy(), 'velocity_base_m_s': velocity.copy(),
        'velocity_source': FIXED if fixed else config.raw.get('velocity_source','nero7_xy_kalman'),
        'configured_speed_m_s': config.raw.get('fixed_belt_speed_m_s') if fixed else None,
        'sample_count': track.sample_count, 'span_s': track.span_s,
        'points': points.copy(), 'grasps': tuple(g.copy() for g in grasps), 'obb': box,
        'candidates': candidates, 'robot_feedback_available': False,
        'robot_self_filter_applied': False, 'trajectory_certified': False,
        'commands_sent': 0, 'can_connections': 0}


class CameraPreview:
    def __init__(self, config, calibration, *, clock=time.monotonic, trajectory_output=None):
        self.config, self.calibration = config, calibration
        self.surface = self.meter = self.frozen = None
        self.epoch = None
        self.clock = clock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-only-geometry')
        self.future = self.geometry = None
        self.generation = 0
        self.last_geometry_s = float('-inf')
        self.previous_source_s = None
        self.last_reason = '等待有效测速'
        self.detector_model = self.scene_model = None
        self.scene_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='original-scene-filters')
        self.scene_future = self.filtered_scene = None
        self.admission = self.admission_model = None
        self.path_preview = None
        if trajectory_output is not None:
            from .trajectory_preview import TrajectoryPreview
            self.path_preview = TrajectoryPreview(config, trajectory_output)

    def reset(self):
        if self.path_preview is not None:self.path_preview.reset()
        self.frozen = None
        self.geometry = None
        self.filtered_scene = None
        self.generation += 1
        self.last_reason = '已重置：重新积累独立观测'
        if self.meter is not None:
            self.meter.reset()

    def close(self):
        self.generation += 1
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.scene_pool.shutdown(wait=True, cancel_futures=True)
        if self.path_preview is not None:self.path_preview.close()

    @staticmethod
    def scene_cloud(frame):
        """Deproject only display pixels; do not allocate a full-frame cloud."""
        z = frame.depth_m[::6, ::6]
        y, x = np.mgrid[0:frame.depth_m.shape[0]:6, 0:frame.depth_m.shape[1]:6]
        k = frame.depth_intrinsics
        local = np.stack(((x-k.cx)*z/k.fx, (y-k.cy)*z/k.fy, z), axis=-1)
        valid = np.isfinite(z) & (z > 0)
        cloud = local[valid]@frame.camera_to_world_cv[:3,:3].T+frame.camera_to_world_cv[:3,3]
        return cloud[np.linalg.norm(cloud, axis=1)<3]

    def geometry_job(self, frame, d2c, surface, peer, capture, generation, robot_state=None):
        from mujoco_conveyor_sorting.runtime.single_pick import detect_target
        from .measured_rgbd import fit_plane, green_belt_points
        started = time.perf_counter()
        try:
            # Reuse the migrated detector and its original self filter. Never
            # substitute configured standby joints for measured encoder data.
            if robot_state is not None:
                if self.detector_model is None:
                    from .planning import calibrated_model
                    self.detector_model = calibrated_model(self.config)
                self.detector_model.check_feedback(robot_state)
            detection = detect_target(frame, d2c, fit_plane=fit_plane,
                green_belt_points=green_belt_points, target_shape='simple',
                robot_model=self.detector_model if robot_state is not None else None,
                robot_state=robot_state,
                measured_surface=surface,
                maximum_target_height_m=self.config.raw['maximum_target_height_m'])
            grasps, box = obb_grasps(detection['object_points'],
                detection['belt_region']['plane_to_base'], self.config.canonical_tcp)
            box = box_in_base(box,np.asarray(detection['belt_region']['plane_to_base']))
            geometry = {'frame':frame, 'rgb_frame':frame, 'd2c':d2c,
                'points':np.asarray(detection['object_points']).copy(), 'grasps':grasps,
                'obb':box, 'source_sequence':frame.header.sequence,
                'source_device_s':frame.header.source_time}
            return dict(generation=generation, geometry=geometry, detection=detection,
                peer=peer, capture=capture, wall_s=time.perf_counter()-started,
                robot_state=robot_state,
                status='实时 OBB / 抓姿候选已显示')
        except (ValueError, RuntimeError) as error:
            return dict(generation=generation, geometry=None, status=str(error),
                        details=getattr(error,'details',{}), wall_s=time.perf_counter()-started)

    def scene_job(self, done):
        """Show precisely the retained scene used by the original planner."""
        try:
            if self.scene_model is None:
                from .planning import calibrated_model
                self.scene_model=calibrated_model(self.config)
            model=self.scene_model;d=done['detection'];state=done['robot_state']
            frame=done['geometry']['frame']
            model.check_feedback(state)
            report=model.install_scene(d['scene_points'],d['object_points'],state,
                camera_origin=d.get('camera_origin_base_m'),belt_region=d['belt_region'],
                depth_frame=frame,target_mask=d.get('mask'),scene_pixel_indices=d.get('scene_pixel_indices'))
            return {'generation':done['generation'],'points':model.scene.copy(),
                'source_device_s':frame.header.source_time,'source_sequence':frame.header.sequence,
                'report':report,'source':'original_RobotModel.install_scene'}
        except (ValueError,RuntimeError) as error:
            return {'generation':done['generation'],'error':str(error),
                    'details':getattr(error,'details',{})}

    def finish_scene(self):
        if self.scene_future is not None and self.scene_future.done():
            result=self.scene_future.result();self.scene_future=None
            if result['generation']==self.generation:self.filtered_scene=result

    def finish_geometry(self):
        if self.future is None or not self.future.done():
            return
        done = self.future.result()
        self.future = None
        if done['generation'] != self.generation:
            return  # a rearm/clock change invalidated the in-flight exposure
        self.geometry = done
        if self.path_preview is not None and self.meter is not None:
            self.path_preview.submit(done, float(self.meter.parameters['tracker_gate_radius']))
        if done.get('geometry') is not None and done.get('robot_state') is not None and self.scene_future is None:
            self.scene_future=self.scene_pool.submit(self.scene_job,done)
        if self.frozen is not None or done.get('geometry') is None:
            return
        peer, capture = done['peer'], done['capture']
        tracks = qualified_tracks(self.config, peer)
        if len(tracks) != 1:
            self.last_reason = f'冻结未触发：合格测速 {len(tracks)} 个，要求恰好1个'
            return
        if capture is None:
            self.last_reason = '冻结未触发：曝光时钟不可用'
            return
        try:
            self.frozen = freeze_candidate(self.config, tracks[0], peer, done['detection'],
                capture, self.clock(), float(self.meter.parameters['tracker_gate_radius']))
            self.frozen.update(rgb_frame=done['geometry']['frame'],d2c=done['geometry']['d2c'])
            self.last_reason = '已按生产条件冻结；下方仍保留实时检测'
        except ValueError as error:
            self.last_reason = f'冻结未触发：{error}'

    def process(self, frame, d2c, *, robot_state=None):
        from mujoco_conveyor_sorting.runtime.depth_calibration import apply_runtime_depth_calibration
        from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
        from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
        from .measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
        started = time.perf_counter()
        frame = apply_runtime_depth_calibration(frame, self.calibration, d2c)
        if self.epoch != frame.header.clock_epoch:
            if self.path_preview is not None:self.path_preview.reset()
            self.generation += 1
            self.surface = self.meter = self.frozen = None
            self.geometry = None
            self.filtered_scene = None
            self.previous_source_s = None
            self.epoch = frame.header.clock_epoch
        result = {'frame': frame, 'frozen': self.frozen, 'tracks': (), 'status': '拟合实测带面',
                  'scene':self.scene_cloud(frame), 'stage':'belt_surface'}
        capture = None
        try:
            capture = depth_capture_timing(frame)['depth_sample_monotonic_estimate']
        except (ValueError, RuntimeError) as error:
            self.last_reason = f'冻结未触发：{error}'
            self.frozen = None
        try:
            if self.surface is None or not self.surface.matches_frame(frame, d2c):
                self.generation += 1
                self.frozen = None
                self.geometry = None
                self.surface = BeltSurface.from_frame(frame, d2c,
                    fit_plane=fit_plane, green_belt_points=green_belt_points)
                from .perception import make_meter
                self.meter=make_meter(self.config,self.surface,previous=self.meter)
                from mujoco_conveyor_sorting.runtime.workpiece_admission import MeasuredWorkpieceAdmission
                self.admission=MeasuredWorkpieceAdmission(self.surface,self.config.raw['maximum_target_height_m'],
                    include_speed_position=self.meter.pointcloud_speed)
            if robot_state is not None:
                if self.admission_model is None:
                    from .planning import calibrated_model
                    self.admission_model=calibrated_model(self.config)
                self.admission_model.check_feedback(robot_state)
            result['stage'] = 'tracking'
            report = self.meter.update(frame, d2c,candidate_filter=lambda *a:self.admission.filter(*a,
                robot_model=self.admission_model if robot_state is not None else None,
                robot_state=robot_state))
            peer = self.meter.latest_snapshot
            if capture is not None:
                from .speed_overlay import live_speed_snapshot
                if is_fixed(self.config):
                    report={**(report or {}),'control_velocity_source':FIXED,
                            'fixed_control_speed_m_s':self.config.raw['fixed_belt_speed_m_s']}
                result['live_speed']=live_speed_snapshot(frame,peer,report,self.config.direction,capture)
            gap = None if self.previous_source_s is None else frame.header.source_time-self.previous_source_s
            self.previous_source_s = frame.header.source_time
            result.update(surface=self.surface, tracks=peer.tracks if peer else (),
                meter_diagnostics=report.get('diagnostics',{}), roi_px=report.get('roi_px'),
                source_gap_s=gap, qualified_count=len(qualified_tracks(self.config,peer)),
                track_labels=track_labels(self.config, peer, self.meter.parameters))
            result['depth_admission']=report.get('diagnostics',{}).get('depth_admission')
            self.finish_geometry()
            self.finish_scene()
            # Geometry has a separate single worker. Its full-resolution
            # extraction must not reduce the rate of independent speed samples.
            fixed_position = self.config.raw.get('freeze_position_upstream_m') is not None
            geometry_eligible = True
            if fixed_position:
                tracks = qualified_tracks(self.config, peer)
                geometry_eligible = capture is not None and len(tracks) == 1
                if geometry_eligible:
                    velocity = prediction_velocity(self.config, tracks[0].velocity_base_m_s,
                                                   self.surface.normal_base)
                    window = freeze_window_for_speed(self.config, velocity@self.config.direction)
                    try:
                        admit_synchronized_observation(capture,
                            np.asarray(tracks[0].observed_position_base)@self.config.direction,
                            velocity@self.config.direction, self.config.follow_bounds, self.clock(),
                            window['distance_m'], minimum_upstream_distance_m=window['minimum_distance_m'])
                    except EntryNotReady as error:
                        geometry_eligible = False
                        self.last_reason = f'固定位置冻结未触发：{error}'
            # A 2 cm capture band can last less than the normal 200 ms preview
            # cadence. In fixed mode submit the eligible exposure immediately.
            if (self.future is None and geometry_eligible and
                    (fixed_position or self.clock()-self.last_geometry_s >= .20)):
                self.future = self.pool.submit(self.geometry_job, frame, np.array(d2c,copy=True),
                    self.surface, peer, capture, self.generation, robot_state)
                self.last_geometry_s = self.clock()
            geometry = None if self.geometry is None else self.geometry.get('geometry')
            if geometry is not None and 0 <= frame.header.source_time-geometry['source_device_s'] <= .65:
                result['live_geometry'] = geometry
            geometry_status = ('实时几何提取中' if self.geometry is None else self.geometry['status'])
            result['geometry_status'] = geometry_status
            result['scene_source']='raw_display_only'
            result['robot_feedback_available']=robot_state is not None
            if self.filtered_scene is not None:
                if 'error' in self.filtered_scene:
                    result['scene_filter_error']=self.filtered_scene['error']
                else:
                    age=frame.header.source_time-self.filtered_scene['source_device_s']
                    if 0<=age<=1.5:
                        result.update(scene=self.filtered_scene['points'],
                            scene_source=self.filtered_scene['source'],scene_age_s=age,
                            scene_filter_report=self.filtered_scene['report'])
            result['geometry_wall_s'] = None if self.geometry is None else self.geometry['wall_s']
            admitted=result.get('depth_admission') or {}
            result['status'] = (f'原候选 {admitted.get("raw_candidates",0)} 个 / '
                f'深度确认物块 {admitted.get("accepted_candidates",0)} 个 / '
                f'跟踪 {len(result["tracks"])} 个 / 合格测速 {result["qualified_count"]} 个；'
                f'{geometry_status}；{self.last_reason}')
            if gap and gap>.6/8:
                result['sampling_warning'] = '当前采样不足以在0.6秒窗口积累8帧，不能产生合格速度'
        except (ValueError, RuntimeError) as error:
            result['status'] = str(error)
        result['frozen'] = self.frozen
        if self.path_preview is not None:
            result['path_preview'] = self.path_preview.poll()
            result['path_status'] = self.path_preview.status
        result['processing_ms'] = (time.perf_counter()-started)*1000
        return result


def qualified_tracks(config, peer):
    return [] if peer is None or not peer.current else [t for t in peer.tracks
        if t.confirmed and not t.misses and (is_fixed(config) or (
            t.speed_valid and t.velocity_base_m_s is not None
            and t.span_s >= config.raw['minimum_speed_span_s']
            and np.asarray(t.velocity_base_m_s)@config.direction>0))]


def track_labels(config, peer, parameters):
    if peer is None:
        return []
    rows=[]
    for t in peer.tracks:
        reasons=[]
        if not t.confirmed:reasons.append('尚未确认')
        if t.misses:reasons.append(f'丢帧{t.misses}')
        if t.sample_count<parameters['speed_min_samples']:
            reasons.append(f'观测{t.sample_count}/{parameters["speed_min_samples"]}帧')
        if t.span_s<config.raw['minimum_speed_span_s']:
            reasons.append(f'跨度{t.span_s:.2f}/{config.raw["minimum_speed_span_s"]:.2f}秒')
        if t.speed_std_m_s>parameters['kalman_max_velocity_std']:
            reasons.append(f'速度不确定度{t.speed_std_m_s*100:.2f}cm/s过大')
        if not t.speed_valid and not reasons:reasons.append('卡尔曼速度尚未合格')
        if t.velocity_base_m_s is not None and np.asarray(t.velocity_base_m_s)@config.direction<=0:
            reasons.append('沿带速度非正')
        rows.append({'id':t.track_id,'position':t.observed_position_base,
                     'velocity':t.velocity_base_m_s,'reasons':reasons,
                     'area_px':t.area_px, 's_m':float(np.asarray(t.observed_position_base)@config.direction)})
    return rows
