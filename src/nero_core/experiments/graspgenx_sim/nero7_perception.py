"""Rendered RGB-D transport for the deployed, unchanged Nero7 estimator.

Static simulation geometry defines calibration/ROI. Object truth, encoder belt
speed and existing MotionPredictor never enter this adapter. The independent
MuJoCo evaluator remains the sole reader of physical object/contact identity.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from types import MethodType, SimpleNamespace
import time

import cv2
import numpy as np

from .camera import FixedGeminiSimulation, cloud_for_track
from .nero7_identity import Nero7StableIdentityMap
from .sensor_control import SensorControlEnvironment, VisualTask
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.perception.rgbd_vision_pipeline import VisionPoseDetection


class CoLocatedNero7Meter(Nero7BeltVelocityMeter):
    """Exact identity depth-to-color transport for a shared simulation pinhole.

    Detection, plane fitting, peer association and Kalman settings are inherited
    unchanged. This path is valid only for the explicit same-K, same-grid,
    identity-d2c synthetic camera; pixel equivalence is covered by a regression.
    """
    def _registered_depth(self, depth, d2c, selected):
        if (np.array_equal(d2c, np.eye(4)) and self._camera_key == self._depth_key
                and not any(self._camera.D)):
            return np.where(selected, depth, np.nan).astype(np.float32)
        return super()._registered_depth(depth, d2c, selected)


def _initialize(surface, classifier, base_to_world, maximum_measured_speed_m_s=None, visual_workspace_x_base=None, simulation_max_height_m=None):
    global _meter, _classifier, _base_to_world, _categories
    _meter = CoLocatedNero7Meter(surface,maximum_measured_speed_m_s=maximum_measured_speed_m_s)
    if visual_workspace_x_base is not None:
        _meter.parameters['workspace_x'] = list(visual_workspace_x_base)
    if simulation_max_height_m is not None:
        _meter.parameters['block_height_max'] = float(simulation_max_height_m)
    _classifier, _base_to_world, _categories = classifier, np.asarray(base_to_world), {}


def _ready():
    return True


def _pack_labels(labels):
    """Losslessly transfer only original foreground pixel bits and bounds."""
    rows, columns = np.nonzero(labels >= 0)
    regions = []
    if len(rows):
        identifiers = labels[rows, columns]
        for identifier in np.unique(identifiers):
            selected = identifiers == identifier
            y0, y1 = int(rows[selected].min()), int(rows[selected].max())+1
            x0, x1 = int(columns[selected].min()), int(columns[selected].max())+1
            crop = labels[y0:y1, x0:x1] == identifier
            regions.append((int(identifier), y0, x0, y1-y0, x1-x0, np.packbits(crop)))
    return labels.shape, regions


def _unpack_labels(shape, regions):
    labels = np.full(shape, -1, np.int32)
    for identifier, y0, x0, height, width, bits in regions:
        mask = np.unpackbits(bits, count=height*width).reshape(height, width).astype(bool)
        labels[y0:y0+height, x0:x0+width][mask] = identifier
    return labels


def _measure_shared(metadata, name, rgb_shape, depth_shape, offset, calibration):
    memory = SharedMemory(name=name)
    try:
        rgb = np.ndarray(rgb_shape, np.uint8, buffer=memory.buf)
        depth = np.ndarray(depth_shape, np.float32, buffer=memory.buf, offset=offset)
        rgb.setflags(write=False); depth.setflags(write=False)
        return _measure((SimpleNamespace(**metadata, rgb=rgb, depth_m=depth), calibration))
    finally:
        memory.close()


def _measure(payload):
    """Only camera pixels, immutable calibration and original capture times."""
    frame, calibration = payload
    report = _meter.update(frame, np.eye(4))
    peer = _meter.latest_snapshot
    masks = np.full(frame.depth_m.shape, -1, np.int32)
    components = None
    if peer is not None and peer.current:
        _, labels = cv2.connectedComponents(np.asarray(_meter._mask, np.uint8), 8)
        for track in peer.tracks:
            if track.misses or abs(track.observation_source_time-peer.source_time) > 1e-9:
                continue
            matching = [d for d in _meter._detections if np.linalg.norm(
                np.asarray(d['centroid_base'])-track.observed_position_base) < 1e-7]
            if len(matching) != 1:
                continue
            detection = matching[0]
            u, v = int(round(detection['u'])), int(round(detection['v']))
            # A connected component's centroid can lie in a hole. Identify by
            # the original component centroid rather than silently losing it.
            ids = np.unique(labels[labels > 0])
            candidate = []
            for label in ids:
                yy, xx = np.nonzero(labels == label)
                candidate.append(((float(xx.mean())-detection['u'])**2+
                                  (float(yy.mean())-detection['v'])**2, int(label)))
            if not candidate:
                continue
            label = min(candidate)[1]
            masks[labels == label] = track.track_id
            if track.track_id not in _categories:
                if components is None:
                    components = _classifier.extract_components(
                        frame.rgb, frame.depth_m, calibration['intrinsics'],
                        calibration['camera_to_world'])
                world = _base_to_world[:3, :3] @ np.asarray(track.observed_position_base)+_base_to_world[:3, 3]
                nearest = sorted(components, key=lambda c: np.linalg.norm(c.position[:2]-world[:2]))
                if nearest and np.linalg.norm(nearest[0].position[:2]-world[:2]) <= .05:
                    component = nearest[0]
                    category, confidence, _ = _classifier.classify(component.feature_vector)
                    _categories[track.track_id] = (category, confidence, component.feature_vector,
                        component.orientation_confidence)
    shape, regions = _pack_labels(masks)
    return {'peer': peer, 'label_shape': shape, 'label_regions': regions, 'categories': deepcopy(_categories),
            'diagnostics': {k: v for k, v in report.items() if k != 'peer_tracks'},
            'processing_finished_wall_s': time.perf_counter()}


def propagated_record(record, now, maximum_age):
    """Return a causal current pose; unavailable speed stays unavailable."""
    stamp = float(record['measurement_timestamp'])
    age = float(now)-stamp
    if age < -1e-9 or age > maximum_age:
        return None
    velocity = record.get('qualified_velocity_world_m_s')
    if velocity is None:
        return None
    velocity = np.asarray(velocity, float)
    velocity_age = float(now)-float(record['velocity_measurement_timestamp'])
    if velocity_age < -1e-9 or velocity_age > maximum_age:
        return None
    position = np.asarray(record['measured_position_world_m'], float)+velocity*max(0., age)
    return position, velocity, age, velocity_age


class Nero7FixedSimulation(FixedGeminiSimulation):
    """Same MuJoCo plant; asynchronous CPU Nero7 with a latest-frame slot."""
    def __init__(self, *, preserve_control_profile=False, maximum_measured_speed_m_s=None,
                 simulation_camera_world_xyz=None, visual_workspace_x_world=None, simulation_max_height_m=None, **kwargs):
        if simulation_max_height_m is not None and (not np.isfinite(simulation_max_height_m) or simulation_max_height_m <= .008):
            raise ValueError('simulation max height must exceed 0.008m')
        self.simulation_max_height_m=simulation_max_height_m
        self.simulation_camera_world_xyz = np.asarray(
            [-.35, 0., 1.25] if simulation_camera_world_xyz is None else simulation_camera_world_xyz, dtype=float)
        if self.simulation_camera_world_xyz.shape != (3,) or not np.isfinite(self.simulation_camera_world_xyz).all():
            raise ValueError('simulation camera position must be a finite world XYZ')
        if visual_workspace_x_world is not None:
            bounds = np.asarray(visual_workspace_x_world, dtype=float)
            if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
                raise ValueError('visual workspace must have finite increasing world X bounds')
            self.visual_workspace_x_world = bounds
        else:
            self.visual_workspace_x_world = None
        self._nero_worker = self._nero_future = self._nero_pending_frame = None
        self._nero_next_capture = 0.
        self._nero_sequence = 0
        self._nero_records = {}
        coast = kwargs.get("config_overrides", {}).get("cameras", {}).get("perception", {}).get("max_track_age_s", 1.)
        self._nero_identity = Nero7StableIdentityMap(maximum_velocity_age_s=coast)
        self._nero_active_task_id = None
        self.nero7_reports = []
        self.nero7_capture_timings = []
        self.nero7_planning_diagnostics = []
        self._nero_inflight_frame = None
        self._nero_memory = None
        self._nero_memory_shape = None
        self._nero_captured = self._nero_skipped = 0
        overrides = deepcopy(kwargs.get("config_overrides", {}))
        self.preserve_control_profile=bool(preserve_control_profile)
        self.maximum_measured_speed_m_s=maximum_measured_speed_m_s
        if not self.preserve_control_profile:
            overrides.setdefault("scene", {}).setdefault("simulation", {})["control_dt"] = .01
            robot = overrides.setdefault("robot", {}).setdefault("robot", {})
            robot.update(control_timestep=.01, velocity_limits=[2.] * 7,
                         acceleration_limits=[2.] * 7, max_joint_velocity=[2.] * 7,
                         max_joint_acceleration=[2.] * 7)
            robot.setdefault("controller", {})["nominal_cartesian_speed"] = .25
            robot.setdefault("trajectory", {}).update(cartesian_linear_speed=.25, tcp_speed_limits_enabled=True)
        kwargs["config_overrides"] = overrides
        super().__init__(**kwargs)
        import mujoco
        base_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'base_link'))
        self.nero_base_to_world = np.eye(4)
        self.nero_base_to_world[:3, :3] = self.data.xmat[base_id].reshape(3, 3)
        self.nero_base_to_world[:3, 3] = self.data.xpos[base_id]
        self.nero_world_to_base = np.linalg.inv(self.nero_base_to_world)
        xy = np.array([[self.belt_min_x, self.belt_center_y-self.belt_width/2],
                       [self.belt_max_x, self.belt_center_y-self.belt_width/2],
                       [self.belt_max_x, self.belt_center_y+self.belt_width/2],
                       [self.belt_min_x, self.belt_center_y+self.belt_width/2]])
        base_points = np.c_[xy, np.full(4, self.belt_height)] @ self.nero_world_to_base[:3, :3].T+self.nero_world_to_base[:3, 3]
        plane = np.eye(4)
        plane[:3, :3] = self.nero_world_to_base[:3, :3]
        plane[:3, 3] = self.nero_world_to_base[:3, :3] @ [0., 0., self.belt_height]+self.nero_world_to_base[:3, 3]
        flat = (base_points-plane[:3, 3]) @ plane[:3, :3]
        self.nero_surface = BeltSurface(plane, np.c_[flat[:, :2].min(0), flat[:, :2].max(0)], flat[:, :2].astype(np.float32),
                                       {'source': 'declared_static_simulation_belt_geometry'})
        visual_workspace_x_base = None
        if self.visual_workspace_x_world is not None:
            endpoints = np.array([[x, self.belt_center_y, self.belt_height, 1.]
                                  for x in self.visual_workspace_x_world])
            base_x = (endpoints @ self.nero_world_to_base.T)[:, 0]
            visual_workspace_x_base = (float(base_x.min()), float(base_x.max()))
        self._nero_worker = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize, initargs=(self.nero_surface, self.vision_pipeline, self.nero_base_to_world,
                                             maximum_measured_speed_m_s, visual_workspace_x_base, simulation_max_height_m))
        self._nero_worker.submit(_ready).result()

    def _read_model_names(self):
        super()._read_model_names()
        # Declared synthetic near-normal camera. The historical strongly
        # oblique view violates legacy_depth's affine image-depth plane model;
        # it is retained in historical entries and in the failed probes.
        import mujoco
        camera_id = self.top_camera_id
        rig_id = int(self.model.cam_bodyid[camera_id])
        parent_id = int(self.model.body_parentid[rig_id])
        world_cv = np.diag([1., -1., -1., 1.])
        world_cv[:3, 3] = self.simulation_camera_world_xyz
        world_parent = np.eye(4)
        world_parent[:3, :3] = self.data.xmat[parent_id].reshape(3, 3)
        world_parent[:3, 3] = self.data.xpos[parent_id]
        parent_gl = np.linalg.inv(world_parent) @ world_cv @ np.diag([1., -1., -1., 1.])
        self.model.body_pos[rig_id] = parent_gl[:3, 3]
        mujoco.mju_mat2Quat(self.model.body_quat[rig_id], parent_gl[:3, :3].reshape(9))
        self.world_depth_cv = world_cv
        mujoco.mj_forward(self.model, self.data)

    def get_observation(self, **kwargs):
        # env.step returns this unused transport value. Formal policy reads the
        # sensor-only boundary below, avoiding the old predictor entirely.
        return {"timestamp": float(self.data.time), "perception_mode": "rendered_rgbd"}

    def reset(self, *args, **kwargs):
        # The entry uses one scene epoch per process. Never reuse observations
        # across a reset, and do not pretend old geometry is a fresh capture.
        if self._nero_sequence:
            raise RuntimeError('Use a new Nero7 simulation process for a new scene epoch')
        return super().reset(*args, **kwargs)

    def _render_paired_rgbd(self):
        """Render identical native pixels from one immutable MuJoCo scene.

        Physical state cannot advance within this function. A single scene
        update is therefore exactly the same camera geometry for depth/RGB.
        """
        import mujoco
        renderer = self._renderer_for(self.top_resolution)
        if renderer is None:
            raise RuntimeError('MuJoCo RGB-D renderer unavailable')
        option = mujoco.MjvOption()
        option.geomgroup[5] = 0
        option.sitegroup[:] = 0
        renderer.update_scene(self.data, camera=self.top_camera_name, scene_option=option)
        renderer.disable_segmentation_rendering()
        renderer.enable_depth_rendering()
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
        rgb = renderer.render().copy()
        return depth, rgb

    def _capture_cameras(self, *, force=False):
        if self._nero_worker is None:
            return
        now = float(self.data.time)
        self.poll_nero7()
        if not force and now < self._nero_next_capture-1e-12:
            return
        self._nero_next_capture = max(self._nero_next_capture+1/30., now+1e-6)
        calibration = self._camera_calibration(self.top_camera_id, self.top_resolution)
        render_started = time.perf_counter()
        depth, rgb = self._render_paired_rgbd()
        self.nero7_capture_timings.append({"capture_sim_s": now,
            "paired_render_wall_s": time.perf_counter()-render_started})
        K = np.asarray(calibration['intrinsics'])
        intrinsics = SimpleNamespace(matrix=K, width=depth.shape[1], height=depth.shape[0], distortion=(0.,)*5)
        source = SimpleNamespace(header=SimpleNamespace(source_time=now, sequence=self._nero_sequence,
            clock_epoch=int(self._scene_epoch)), rgb=rgb, depth_m=np.asarray(depth, np.float32),
            depth_intrinsics=intrinsics, color_intrinsics=intrinsics,
            camera_to_world_cv=self.nero_world_to_base @ np.asarray(calibration['camera_to_world_opencv']))
        # Sample only robot/jaw encoders at the same physical instant as RGB-D.
        # Object poses, lifecycle and contacts do not enter this camera frame.
        jaw = self._gripper_observation()
        robot_feedback = {'joints': np.asarray(self.data.qpos[self.arm_qpos_adrs], dtype=float).copy(),
                          'width_m': float(jaw['opening_width']), 'sample_time_s': now}
        frame = {'timestamp': now, 'top_depth': depth, 'top_rgb': rgb, 'top_camera': calibration,
                 'scene_epoch': int(self._scene_epoch), 'source': source,
                 'capture_robot_feedback': robot_feedback, 'capture_wall_s': time.perf_counter()}
        self._nero_sequence += 1
        self._nero_captured += 1
        if self._nero_pending_frame is not None:
            self._nero_skipped += 1
        self._nero_pending_frame = frame
        self._last_camera_capture = now
        self.poll_nero7()

    def poll_nero7(self):
        if self._nero_future is not None and self._nero_future.done():
            processed = self._nero_future.result()
            self._nero_future = None
            frame = self._nero_inflight_frame
            peer = processed['peer']
            processed['labels'] = _unpack_labels(processed.pop('label_shape'), processed.pop('label_regions'))
            if peer is not None:
                self._publish_peer(processed, frame)
            self._nero_inflight_frame = None
        if self._nero_future is None and self._nero_pending_frame is not None and self._nero_worker is not None:
            frame, self._nero_pending_frame = self._nero_pending_frame, None
            self._nero_inflight_frame = frame
            source = frame['source']
            shape = (source.rgb.shape, source.depth_m.shape)
            offset = (source.rgb.size+7)//8*8
            if shape != self._nero_memory_shape:
                if self._nero_memory is not None:
                    self._nero_memory.close(); self._nero_memory.unlink()
                self._nero_memory = SharedMemory(create=True, size=offset+source.depth_m.size*4)
                self._nero_memory_shape = shape
            np.copyto(np.ndarray(shape[0], np.uint8, buffer=self._nero_memory.buf), source.rgb)
            np.copyto(np.ndarray(shape[1], np.float32, buffer=self._nero_memory.buf, offset=offset), source.depth_m)
            metadata = {key: getattr(source, key) for key in
                ('header', 'color_intrinsics', 'depth_intrinsics', 'camera_to_world_cv')}
            self._nero_future = self._nero_worker.submit(_measure_shared, metadata,
                self._nero_memory.name, *shape, offset, frame['top_camera'])

    def _publish_peer(self, processed, frame):
        peer = processed['peer']
        now = float(self.data.time)
        assignments, association = self._nero_identity.update(peer,
            processed['diagnostics']['parameters'], active_task_id=self._nero_active_task_id)
        if association['clock_or_generation_reset']:
            self._nero_records.clear()
        for task_id in association['dropped_provisional_task_ids']:
            self._nero_records.pop(task_id, None)
        report_record = {'capture_sim_s': peer.source_time, 'publication_sim_s': now,
            'capture_to_publication_s': now-peer.source_time, 'peer': asdict(peer),
            'diagnostics': processed['diagnostics'], 'capture_wall_s': frame['capture_wall_s'],
            'processing_finished_wall_s': processed['processing_finished_wall_s'],
            'stable_task_association': association, 'geometry_binding_events': []}
        self.nero7_reports.append(report_record)
        states, detections, frame_measurements = {}, [], {}
        labels = np.full(processed['labels'].shape, -1, np.int32)
        for task_id, track in assignments.items():
            previous = self._nero_records.get(task_id, {})
            category_info = processed['categories'].get(track.track_id)
            if previous:
                category_info = (previous['category_id'], previous['category_confidence'],
                    previous['feature_vector'], previous['orientation_confidence'])
            if category_info is None:
                continue
            category, confidence, feature, orientation = category_info
            labels[processed['labels'] == track.track_id] = task_id
            pbase = np.asarray(track.observed_position_base)
            position = self.nero_base_to_world[:3, :3] @ pbase+self.nero_base_to_world[:3, 3]
            binding = self._nero_identity.bindings[task_id]
            if binding.geometry_begin is None and 'top_depth' in frame and 'top_camera' in frame:
                captured = {**frame, 'top_instance_labels': processed['labels']}
                cloud_world = cloud_for_track(captured, track.track_id)
                world_to_base = np.linalg.inv(self.nero_base_to_world)
                points_base = cloud_world @ world_to_base[:3, :3].T+world_to_base[:3, 3]
                begin = binding.bind_measured_geometry(points_base, peer)
                report_record['geometry_binding_events'].append({
                    'operation': 'begin_target_binding', 'task_id': task_id,
                    'actual_peer_track_id': track.track_id,
                    'geometry_capture': {'clock_epoch': peer.clock_epoch,
                        'sequence': peer.sequence, 'device_timestamp': peer.source_time,
                        'host_capture_timestamp': peer.source_time,
                        'host_clock': 'simulation_capture_time'},
                    'captured_point_count': len(points_base),
                    'result': deepcopy(begin),
                    'generation_pending': bool(begin.get('binding_reset')
                        and begin.get('track_generation') is None)})
                if begin['accepted']:
                    current_record = SimpleNamespace(peer=peer)
                    binding.accept(track, current_record, binding.query(track, current_record))
            measurement = binding.measurement
            qualified = (None if not measurement['speed_valid'] else self.nero_base_to_world[:3, :3]
                         @ np.asarray(measurement['velocity_base_m_s']))
            velocity_stamp = measurement.get('device_timestamp', track.motion_source_time)
            if qualified is None:
                qualified = previous.get('qualified_velocity_world_m_s')
                velocity_stamp = previous.get('velocity_measurement_timestamp', track.motion_source_time)
            record = {'object_id': task_id, 'category_id': category,
                'category_confidence': float(confidence), 'quaternion': np.array([1., 0., 0., 0.]),
                'measured_position_world_m': position, 'measurement_timestamp': track.observation_source_time,
                'qualified_velocity_world_m_s': qualified, 'velocity_measurement_timestamp': velocity_stamp,
                'speed_valid_current_frame': track.speed_valid,
                'velocity_query': measurement, 'target_geometry_binding': binding.geometry_begin,
                'sample_count': track.sample_count,
                'span_s': track.span_s, 'track_generation': peer.track_generation,
                'actual_peer_track_id': track.track_id,
                'task_reassociation_reason': association['task_reassociation_reasons'][task_id],
                'mask_pixels': int((labels == task_id).sum()), 'feature_vector': feature,
                'orientation_confidence': float(orientation), 'publication_timestamp': now}
            self._nero_records[task_id] = record
            frame_measurements[int(task_id)] = {'actual_peer_track_id': int(track.track_id),
                'track_generation': int(peer.track_generation), 'velocity_measurement': deepcopy(measurement)}
            states[str(task_id)] = {'object_id': task_id, 'category_id': category,
                'category_confidence': confidence, 'position': position, 'quaternion': record['quaternion'],
                'linear_velocity': qualified, 'timestamp': peer.source_time, 'track_generation': peer.track_generation,
                'actual_peer_track_id': track.track_id,
                'confidence': confidence, 'visible': True, 'track_observation_age': 0.,
                'mask_pixels': record['mask_pixels'], 'valid_depth_pixels': record['mask_pixels'],
                'feature_vector': feature, 'orientation_confidence': orientation,
                'pose_source': 'nero7_real_rendered_capture_no_speed_prior'}
            detections.append(VisionPoseDetection(task_id, category, position, record['quaternion'],
                peer.source_time, confidence, confidence, record['mask_pixels'], record['mask_pixels'],
                0., orientation, True, 0., np.asarray(feature)))
        self.perception_buffer.append_frame(peer.source_time, states, mark_missing=True)
        self._associate_visual_tracks_for_evaluation(detections)
        visible_frame = {k: v for k, v in frame.items() if k != 'source'}
        visible_frame['nero_capture'] = {'clock_epoch': int(peer.clock_epoch),
            'sequence': int(peer.sequence), 'device_time_s': float(peer.source_time),
            'publication_time_s': now, 'measurements': frame_measurements}
        visible_frame['top_instance_labels'] = labels
        visible_frame['top_segmentation'] = np.stack((labels, np.where(labels >= 0, 0, -1)), axis=-1)
        self._camera_frames.append(visible_frame)
        self.vision_pipeline.last_instance_labels = labels

    def camera_record(self):
        result = super().camera_record()
        result.update(nero7_capture_hz_upper_bound=30., rgb_depth_same_capture=True,
            original_control_profile_preserved=self.preserve_control_profile,
            explicit_maximum_measured_speed_m_s=self.maximum_measured_speed_m_s,
            depth_color_transform='identity for co-located native MuJoCo RGB-D pinhole',
            device_timestamp='actual MuJoCo capture time, not processing completion',
            base_to_world=self.nero_base_to_world.tolist(),
            initial_speed_prior=None, old_motion_predictor_used=False,
            surface_source='static simulation belt geometry; not field calibration',
            pose_source='declared synthetic near-top-down view; historical oblique camera fails original legacy affine depth-plane fit',
            simulation_camera_world_xyz_m=self.simulation_camera_world_xyz.tolist(),
            visual_workspace_x_world_m=None if self.visual_workspace_x_world is None else self.visual_workspace_x_world.tolist(),
            scene_note='Declared synthetic near-top-down camera; belt and bin layout are simulated, not replayed field extrinsics.',
            cpu_transport='one spawn process, single-frame shared memory, one in-flight/latest pending capture; lossless original foreground pixel bits')
        return result

    def close(self):
        try:
            if self._nero_worker is not None:
                self._nero_worker.shutdown(wait=True, cancel_futures=True)
        finally:
            self._nero_worker = None
            try:
                if self._nero_memory is not None:
                    self._nero_memory.close()
                    self._nero_memory.unlink()
            finally:
                self._nero_memory = None
                super().close()



def _single_seed_candidate_ik(snapshot, position, quaternion, qpos_seed, *, allow_multiseed=False):
    """Use the field candidate branch's one-seed DLS contract.

    Coordinates are already world coordinates in this simulation snapshot;
    the field wrapper additionally converts its belt frame into robot base.
    All endpoint and vertical-path collision checks remain with the caller.
    """
    if allow_multiseed:
        raise ValueError('候选分支沿用单一关节种子')
    return snapshot.ik.solve(position, quaternion, qpos_seed=qpos_seed)


class Nero7SensorControlEnvironment(SensorControlEnvironment):
    def candidate_snapshot(self):
        """Keep the original private clone, with the field candidate IK mode."""
        snapshot = super().candidate_snapshot()
        snapshot.solve_ik_result = MethodType(_single_seed_candidate_ik, snapshot)
        snapshot.candidate_ik_semantics = 'single_seed_DLS_same_as_field_candidate_wrapper'
        return snapshot

    def provenance(self):
        result = super().provenance()
        result.update(control_source='same_deployed_nero7_detector_tracker_xy_kalman_rendered_rgbd',
            conveyor_encoder=False, belt_speed_prior=None, old_motion_predictor=False,
            maximum_qualified_velocity_age_s=self.config.maximum_track_age_s,
            maximum_position_coast_s=self.config.maximum_track_age_s,
            classification='existing CAD RGB-D classifier only; does not supply velocity',
            sampling='30Hz capture upper bound; actual completed sample stamps recorded',
            task_reassociation='actual Nero7GraspTracking._select_track function with original peer30mm gate',
            candidate_ik_semantics='single_seed_DLS_same_as_field_candidate_wrapper')
        return result

    def refine_selected_grasp_plan(self, track_id, required_motion_time,
                                   deadline_margin=.25, **kwargs):
        """Preserve original checks and identify the actual missing input.

        Legacy SensorControlEnvironment combines task-state, reservation and
        missing-prediction checks under target_not_in_planning. A real visual
        timeout is logged as missing prediction, without extending its age.
        """
        task = self.tasks.get(int(track_id))
        prediction = (self._last_observation or {}).get('predictions', {}).get(int(track_id))
        if (task is not None and task.state == 'PLANNING'
                and int(track_id) == self.selected_target_id and prediction is None):
            now = float(self._physics.data.time)
            observed = self._physics._nero_records.get(int(track_id), {})
            stamp = observed.get('measurement_timestamp')
            diagnostic = {'accepted': False, 'reason': 'visual_prediction_unavailable',
                'timestamp': now, 'target_id': int(track_id), 'task_state': task.state,
                'selected_target_id': self.selected_target_id,
                'last_measurement_timestamp': stamp,
                'measurement_age_s': None if stamp is None else now-float(stamp),
                'maximum_track_age_s': self.config.maximum_track_age_s,
                'actual_peer_track_id': observed.get('actual_peer_track_id'),
                'velocity_query': observed.get('velocity_query')}
            self._physics.nero7_planning_diagnostics.append(deepcopy(diagnostic))
            return diagnostic
        return super().refine_selected_grasp_plan(track_id, required_motion_time,
            deadline_margin, **kwargs)

    retime_selected_grasp_plan = refine_selected_grasp_plan

    def get_observation(self, **_):
        self._physics._nero_active_task_id = self.selected_target_id
        self._physics.poll_nero7()
        now, epoch = float(self._physics.data.time), int(self._physics._scene_epoch)
        if epoch != self._scene_epoch:
            self.tasks.clear(); self.selected_target_id = None
            self.observer.reset(); self._scene_epoch = epoch
        self._jaw_state = self._read_jaws(now)
        robot = self.get_robot_state()
        predictions, records = {}, []
        available = sorted(self._physics._nero_records.items())
        measured_belt = None
        for identifier, observed in available:
            current = propagated_record(observed, now, self.config.maximum_track_age_s)
            if current is None:
                continue
            position, velocity, age, speed_age = current
            task = self.tasks.setdefault(identifier, VisualTask(identifier, int(observed['category_id'])))
            if task.state in self._DONE:
                continue
            if task.state in {'TRACKING', 'PLANNING'}:
                task.category_id = int(observed['category_id'])
            spec = self._physics._object_spec_by_category.get(task.category_id)
            if spec is None:
                continue
            window = self._visual_window(position, velocity, now, task)
            ready = True  # Only independently qualified Nero7 velocities reach this branch.
            stamp = observed['measurement_timestamp']
            quat = observed['quaternion']
            predicted_time = task.planned_time
            if predicted_time is None:
                preferred = float(self.scene_config.get('workspace', {}).get('preferred_grasp_x', .08))
                speed = float(velocity[0])
                predicted_time = now+(max(0., preferred-position[0])/speed if speed > 1e-5 else 0.)
            future = np.r_[position+velocity*max(0., predicted_time-now), quat]
            prediction = {'object_id': identifier, 'observed_timestamp': stamp,
                'capture_timestamp': stamp, 'measurement_timestamp': stamp,
                'publication_timestamp': observed['publication_timestamp'], 'timestamp': now,
                'observation_age': age, 'track_observation_age': age,
                'observed_pose': np.r_[observed['measured_position_world_m'], quat],
                'current_estimated_pose': np.r_[position, quat], 'predicted_grasp_pose': future,
                'estimated_velocity': velocity, 'estimated_acceleration': np.zeros(3),
                'estimated_grasp_time': predicted_time, 'prediction_ready': ready,
                'prediction_confidence': float(observed['category_confidence']),
                'prediction_uncertainty': np.zeros(3),
                'prediction_uncertainty_status': 'not_estimated_numeric_compatibility_placeholder', 'visible': age <= self._physics.camera_period+1e-9,
                'orientation_confidence': observed['orientation_confidence'], 'grasp_window': window,
                'history_sample_count': observed['sample_count'], 'motion_compensation_enabled': True,
                'prediction_model': 'nero7_qualified_xy_velocity_with_original_capture_time',
                'velocity_measurement_timestamp': observed['velocity_measurement_timestamp'],
                'velocity_observation_age': speed_age, 'track_generation': observed['track_generation'],
                'actual_peer_track_id': observed['actual_peer_track_id'],
                'task_reassociation_reason': observed['task_reassociation_reason']}
            if task.planned_time is not None:
                prediction.update(planned_grasp_time=task.planned_time, planned_grasp_pose=task.planned_pose.copy())
            record = {**prediction, 'category_id': task.category_id,
                'category_confidence': observed['category_confidence'], 'bin_id': int(spec.bin_id),
                'position': position, 'quaternion': quat, 'linear_velocity': velocity,
                'state': task.state, 'pose_source': 'nero7_measured_rgbd_no_encoder',
                'window_valid': window['feasible'], 'in_grasp_window': window['current_in_window'],
                'grasp_window_start': window['start_time'], 'grasp_window_end': window['end_time']}
            task.last_record = deepcopy(record)
            records.append(record); predictions[identifier] = prediction
            if identifier == self.selected_target_id or measured_belt is None:
                measured_belt = float(np.linalg.norm(velocity[:2]))
        controller = self._physics.robot_controller
        result = {**robot, 'joint_target': controller.commanded_joint_positions,
            'joint_target_velocity': controller.commanded_joint_velocities,
            'gripper': deepcopy(self._jaw_state), 'gripper_opening': self._jaw_state['opening_width'],
            'belt_speed': measured_belt, 'timestamp': now, 'scene_epoch': epoch,
            'objects': records, 'predictions': predictions, 'selected_target_id': self.selected_target_id,
            'perception': predictions.get(self.selected_target_id, {}), 'perception_mode': self.perception_mode,
            'perception_source': 'nero7_rendered_rgbd_no_encoder',
            'system_latency': {name: float(getattr(self._physics, name+'_latency')) for name in
                               ('visual', 'planning', 'control', 'robot_response')}}
        self._last_observation = result
        return deepcopy(result)
