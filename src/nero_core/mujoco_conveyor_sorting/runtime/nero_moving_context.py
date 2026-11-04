"""Moving RGB-D target adapter for the existing complete sorting policy.

Live conveyor velocity comes from visual measurements. All position updates
come from an immutable visual snapshot; confirmed motor width/force is the
existing hardware grasp proxy. There is no second task state machine here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass, replace
import threading

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from ..controllers.visual_acquisition import VisualAcquisitionSortingPolicy
from ..controllers._math import belt_plane_motion_from_xy
from ..controllers.planning_worker import AsyncPlanningWorker, MuJoCoGlobalPlanningEngine
from .nero_policy_context import NeroPolicyContext, _copy_geometry, _pose
from .single_pick import PickRejected


def _readonly(value, shape=None):
    result = np.asarray(value, dtype=float).copy()
    if shape is not None and result.shape != shape:
        raise ValueError(f'expected shape {shape}, got {result.shape}')
    if not np.isfinite(result).all():
        raise ValueError('moving-target geometry must contain finite values')
    result.setflags(write=False)
    return result


def _matrix(pose):
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(np.asarray(pose[3:])[[1, 2, 3, 0]]).as_matrix()
    return result


@dataclass(frozen=True, init=False)
class _Cloud:
    points: np.ndarray
    _tree: cKDTree | None = field(default=None, repr=False, compare=False)

    def __init__(self, points, tree=None):
        # Preserve the existing optional supplied-index constructor. Normal
        # publications use build(), which owns an immutable copy of points.
        object.__setattr__(self, 'points', points)
        object.__setattr__(self, '_tree', tree)

    @property
    def tree(self):
        # Observation/dispatch can publish another cloud before this one is
        # consumed. Build its spatial index only for an actual query, retaining
        # the original points, timestamp, sweep horizon and collision margin.
        # No cross-cloud lock/cache: independent worker and live publications
        # must not serialize their builds or share changing target geometry.
        tree = self._tree
        if tree is None:
            tree = cKDTree(self.points)
            object.__setattr__(self, '_tree', tree)
        return tree

    @classmethod
    def build(cls, points):
        points = _readonly(points)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('target cloud must be N x 3')
        return cls(points)


@dataclass(frozen=True)
class _CollisionSnapshot:
    """One control-thread publication; workers never read a live target dict."""
    kind: str
    non_finger: _Cloud
    fingers: _Cloud
    contact_poses: tuple[np.ndarray, ...]
    payload_local: np.ndarray | None = None
    drop_pose: np.ndarray | None = None
    non_finger_margin: float = .004


def _planning_target_snapshot(request, template_local, approach_height):
    """Build one future object from immutable request poses and visible shape.

    Worker ETA estimation may replace a request's intercept pose before full
    certification. Its obstacle cloud must refer to that same time and pose,
    never to the original submitted target or mutable live visual state.
    """
    if getattr(request, 'planning_mode', 'legacy') == 'full_preplanned':
        contact = request.target_pose
        if request.full_phase == 'lift':
            empty = _Cloud.build(np.empty((0, 3)))
            close = contact.copy()
            close[:3] += np.asarray(request.continuation_velocity_m_s)*request.full_close_window_s
            return _CollisionSnapshot('payload', empty, empty, (), template_local, _readonly(close))
        if request.full_sample_time_s is not None:
            end = float(request.full_sample_time_s)
            begin = float(request.full_sample_start_time_s if request.full_sample_start_time_s is not None else end)
            velocity = np.asarray(request.continuation_velocity_m_s)
            steps = max(1, int(np.ceil(np.linalg.norm(velocity*(end-begin))/.004)))
            poses = tuple(np.r_[contact[:3]+velocity*(stamp-request.planned_target_time_s), contact[3:]]
                          for stamp in np.linspace(begin, end, steps+1))
            matrices = tuple(_matrix(pose) for pose in poses)
            cloud = _Cloud.build(np.concatenate([template_local@matrix[:3, :3].T+matrix[:3, 3]
                                                for matrix in matrices]))
            return _CollisionSnapshot('planning', cloud, cloud, tuple(_readonly(pose) for pose in poses),
                                      non_finger_margin=.006)
        # Contact already is a TCP pose; do not subtract the legacy high offset.
        matrix = _matrix(contact)
        cloud = _Cloud.build(template_local@matrix[:3, :3].T+matrix[:3, 3])
        return _CollisionSnapshot('planning', cloud, cloud, (_readonly(contact),))
    poses = request.continuation_target_poses
    if not poses:
        if (getattr(request, 'continuous_intercept', False)
                or getattr(request, 'target_is_contact', False)):
            # Continuous targets already describe contact, not a raised wait
            # pose. Keep the workpiece at the certified interception height.
            poses = request.target_poses
        else:
            poses = tuple(np.r_[pose[:3] - np.array([0., 0., approach_height]), pose[3:]]
                          for pose in request.target_poses)
    poses = tuple(_readonly(pose, (7,)) for pose in poses)
    matrices = tuple(_matrix(pose) for pose in poses)
    cloud = _Cloud.build(np.concatenate([template_local @ matrix[:3, :3].T + matrix[:3, 3]
                                        for matrix in matrices]))
    return _CollisionSnapshot('planning', cloud, cloud, poses)


class _MovingPlanningWorker(AsyncPlanningWorker):
    def __init__(self, engine, snapshot_factory, snapshots, lock, *,
                 on_effective_request=None, on_planning_result=None):
        self._snapshot_factory = snapshot_factory
        self._snapshots = snapshots
        self._snapshots_lock = lock
        # Geometry publication follows every proposed revision. The original
        # submitted request must survive a failed revision independently.
        self._submitted_requests = {}
        self._on_effective_request = on_effective_request
        self._on_planning_result = on_planning_result
        super().__init__(engine, name='nero-moving-global-planner')

    def submit(self, request):
        # submit() is called on the policy thread. Publish the complete frozen
        # geometry before waking the background engine, including payload data.
        snapshot = self._snapshot_factory(request)
        with self._snapshots_lock:
            self._snapshots[request.request_id] = (request, snapshot)
            self._submitted_requests[request.request_id] = request
        accepted = super().submit(request)
        if not accepted:
            with self._snapshots_lock:
                self._snapshots.pop(request.request_id, None)
                self._submitted_requests.pop(request.request_id, None)
        return accepted

    def poll(self):
        result = super().poll()
        if result is not None:
            with self._snapshots_lock:
                self._snapshots.pop(result.request_id, None)
                submitted = self._submitted_requests.pop(result.request_id, None)
            effective = getattr(result, 'effective_request', None)
            if effective is not None and self._on_effective_request is not None:
                # This callback runs on the policy thread. Adoption still
                # belongs to the shared policy's existing refine hook.
                self._on_effective_request(effective)
            if self._on_planning_result is not None:
                self._on_planning_result(result, effective if effective is not None else submitted)
        return result

    def cancel_pending(self):
        with self._condition:
            super().cancel_pending()
            # Pending and queued results were discarded. An active job may
            # still return its cancellation result and needs its own clock.
            active_id = None if self._active is None else self._active.request_id
            with self._snapshots_lock:
                self._submitted_requests = {
                    key: value for key, value in self._submitted_requests.items()
                    if key == active_id}

    def close(self):
        stopped = super().close()
        with self._snapshots_lock:
            self._submitted_requests.clear()
        return stopped


class NeroMovingPolicyContext(NeroPolicyContext):
    """One visually tracked moving part, with a fixed world-position put-back."""

    def __init__(self, geometry, detection, initial_state, settings, *,
                 snapshot_provider, belt_direction, belt_origin, s_range,
                 tracking_s_range=None, nominal_speed_m_s=None, moving_approach_mode='high'):
        if moving_approach_mode not in ('high', 'continuous'):
            raise ValueError('moving_approach_mode must be high or continuous')
        self.moving_approach_mode = moving_approach_mode
        direction = _readonly(belt_direction, (3,))
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError('belt direction must be a nonzero base-frame vector')
        self.belt_direction = _readonly(direction / norm)
        self.belt_origin = _readonly(belt_origin, (3,))
        self.s_range = self._bounds(s_range)
        self.tracking_s_range = self._bounds(tracking_s_range if tracking_s_range is not None else s_range)
        if not callable(snapshot_provider):
            raise ValueError('snapshot_provider must be callable')
        self.snapshot_provider = snapshot_provider
        self.nominal_speed_m_s = None if nominal_speed_m_s is None else float(nominal_speed_m_s)
        if self.nominal_speed_m_s is not None and (
                not np.isfinite(self.nominal_speed_m_s) or self.nominal_speed_m_s < 0):
            raise ValueError('nominal belt speed must be finite and nonnegative')
        self._policy_clock_origin = None
        self._motion_snapshot = None
        self._planned_time = None
        self._planned_pose = None
        self._candidate_motion_plan = detection.get('network_provenance', {}).get('candidate_motion_plan')
        self._candidate_plan_handoff = None
        self._capture_pose = None
        self._capture_time = None
        self._capture_width = None
        self._payload_local = None
        self._released_pose = None
        self._release_time = None
        self._release_tcp = None
        self._release_clearance = None
        self._payload_grasp_local = None
        self._perception_error = None
        self._live_collision = None
        self._dispatch_dt = 1. / settings.control_hz
        self._intercept_updates = []
        self._planner_requests = []
        self._planner_results = []
        self._collision_rejection = None
        self._network_grasp_source = detection.get('target_geometry_source', 'geometry_diagnostic')
        super().__init__(geometry, detection, initial_state, settings)
        self.policy.close()
        template = np.asarray(geometry.target, dtype=float)
        if not len(template):
            template = np.asarray(detection.get('object_points', np.empty((0, 3))), dtype=float)
        self._template_local = _readonly(
            (template - self.grasp_transform[:3, 3]) @ self.grasp_transform[:3, :3])
        self._empty_cloud = _Cloud.build(np.empty((0, 3)))
        self._validation_geometry.target = self._empty_cloud.points
        self._target['object_id'] = (self.target_identity.object_id if self.target_identity is not None
                                     else 'measured-moving-part')
        self._target['closure_requires_arm_settle'] = False
        self._target['confirmed_grasp_settle_time'] = 0.
        self._target['grasp_height'] = 0.
        self._target['confirmation_source'] = 'measured_rgbd_moving_target'
        candidate = detection.get('network_candidate', {})
        candidate_seed = candidate.get('intercept_ik' if self.moving_approach_mode == 'continuous'
                                      else 'pregrasp_ik', {}).get('joint_positions')
        if candidate_seed is not None:
            # An NN candidate's IK is only a numerical starting guess. The
            # original shared seed ordering and full planner checks still run.
            self.robot_config['calibration_poses']['belt_intercept_pregrasp'] = {
                'joint_positions': np.asarray(candidate_seed, dtype=float).copy(),
                'source': 'selected_graspgenx_candidate_prefilter_only'}
        self.scene_config = {'workspace': {'dynamic_intercept': {
            'enabled': True, 'belt_direction': self.belt_direction.tolist(),
            'belt_origin': self.belt_origin.tolist(), 's_range': list(self.s_range),
            'tracking_s_range': list(self.tracking_s_range),
            'candidate_step_m': .04, 'servo_lookahead_s': .04,
            'tracking_horizon_s': .35,
        }}}
        config = asdict(self.policy.config)
        # The measured start already satisfies the controller's command
        # limits. Extra pregrasp reserve cannot exclude that same start.
        # Keep the base margin and raw joint limits; cap only the additional
        # reserve using the existing 3 mrad feedback slack.
        start_q = np.asarray(initial_state['joints'])
        start_margin = float(np.min(np.minimum(start_q - geometry.limits[:, 0],
                                               geometry.limits[:, 1] - start_q)))
        config['pregrasp_tracking_joint_margin_reserve'] = min(
            config['pregrasp_tracking_joint_margin_reserve'],
            max(0., start_margin - .003 - config['minimum_joint_limit_margin']))
        config.update(enable_motion_compensation=True, dynamic_intercept_tracking=True,
                      moving_approach_mode=self.moving_approach_mode,
                      pregrasp_fixed_high_enabled=self.moving_approach_mode == 'high',
                      timed_descent_enabled=self.moving_approach_mode == 'high',
                      # Generate the complete descent using joint v/a/jerk.
                      # Do not reject that checked curve again against the
                      # local Cartesian servo's 0.25 m/s and 0.8 rad/s caps.
                      timed_descent_tcp_speed_limits_enabled=False,
                      # No extra 250 mm/s clipping in local descent, closure,
                      # lift or release. Angular speed, original step/feedback
                      # bounds and the joint controller remain unchanged.
                      local_tcp_linear_speed_limit_enabled=False,
                      # Same high-route envelope as the successful simulation.
                      # This only governs certified PREGRASP joint edges, not
                      # local descent/closing, carrying or firmware limits.
                      acquisition_path_tracking_error_limit=.20,
                      # Candidate and path safety are geometric: no world-down
                      # orientation cone in the real measured-scene adapter.
                      minimum_downward_alignment=-1.,
                      acquisition_minimum_downward_alignment=-1.,
                      planner_minimum_downward_alignment=-1.,
                      acquisition_entry_alignment_hysteresis=0.,
                      track_belt_during_close=True,
                      tracking_velocity_feedforward_time=.04, closing_tracking_lead_time=.04,
                      closure_contact_lead_time=.04,
                      # The measured width/force adapter already debounces for
                      # 0.4 s. A second policy dwell would double that period.
                      closure_horizontal_tolerance=.008,
                      closure_vertical_tolerance=.004,
                      closure_orientation_tolerance=.025,
                      cartesian_linear_speed=min(.25, settings.joint_velocity_limit * .15))
        normal = detection.get('plane', {}).get('normal_base')
        if normal is None and geometry.scene_region_plane_to_base is not None:
            normal = geometry.scene_region_plane_to_base[:3, 2]
        self._measured_plane_normal = None if normal is None else _readonly(normal, (3,))
        if normal is None:
            # Offline fixtures without a measured plane keep a normal
            # perpendicular to their explicitly declared conveyor direction.
            normal = np.array([0., 0., 1.])
            normal -= self.belt_direction * float(normal @ self.belt_direction)
            if np.linalg.norm(normal) <= 1e-8:
                normal = np.array([1., 0., 0.])
        self.policy = VisualAcquisitionSortingPolicy(config, env=self, belt_plane_normal=normal)
        self.full_plan_grasp_width_m = float(np.clip(
            np.ptp(np.asarray(self._template_local)[:, 1]), .004, .095))

    @staticmethod
    def _bounds(value):
        value = np.asarray(value, dtype=float)
        if value.shape != (2,) or not np.isfinite(value).all() or value[0] >= value[1]:
            raise ValueError('conveyor longitudinal bounds must be two increasing finite values')
        return tuple(map(float, value))

    def set_policy_clock_origin(self, started_abs):
        value = float(started_abs)
        if not np.isfinite(value):
            raise ValueError('policy clock origin must be finite')
        if self._policy_clock_origin is not None and self._policy_clock_origin != value:
            raise ValueError('moving policy clock origin is already fixed')
        self._policy_clock_origin = value
        if self._candidate_motion_plan is not None:
            candidate_time = float(self._candidate_motion_plan['planned_capture_host_time_s']) - value
            self._planned_time = candidate_time if candidate_time > 0. else None
            self._candidate_plan_handoff = {
                'capture_policy_time_s': candidate_time,
                'source': 'same_network_candidate_capture_clock',
                'expired_before_policy_start': candidate_time <= 0.,
                'full_path_revalidation_required': True}
        provider = getattr(self.snapshot_provider, '__self__', self.snapshot_provider)
        setter = getattr(provider, 'set_policy_clock_origin', None)
        if callable(setter):
            setter(value)

    def _snapshot(self, now, arrival=0.):
        if self._policy_clock_origin is None:
            raise ValueError('set_policy_clock_origin() before using moving perception')
        try:
            return self.snapshot_provider(self._policy_clock_origin + float(now),
                                          robot_arrival_time=max(0., float(arrival)))
        except Exception as error:
            # Camera availability is policy input. Do not introduce a second
            # vision watchdog in the device adapter; the shared policy owns
            # prediction_unavailable/target_lost and ordinary recovery.
            self._perception_error = str(error)
            return None

    def _motion_velocity(self):
        """Prediction vector; the original XY measurement stays in the snapshot."""
        measured = _readonly(self._motion_snapshot.velocity, (3,))
        velocity = getattr(self._motion_snapshot, 'prediction_velocity', None)
        if velocity is not None:
            return _readonly(velocity, (3,)).copy()
        if self._motion_snapshot.velocity_source == 'nero7_xy_kalman':
            return belt_plane_motion_from_xy(measured, self._measured_plane_normal)
        return measured.copy()

    def _coordinate(self, position):
        return float((np.asarray(position) - self.belt_origin) @ self.belt_direction)

    def _window(self, pose, velocity, now):
        s = self._coordinate(pose[:3])
        speed = float(np.asarray(velocity) @ self.belt_direction)
        if abs(speed) < 1e-6:
            start, end = now, now + 300.
            feasible = self.s_range[0] <= s <= self.s_range[1]
        else:
            times = sorted((now + (bound - s) / speed for bound in self.s_range))
            start, end = max(now, times[0]), times[1]
            feasible = end >= start
        return {'start_time': float(start), 'end_time': float(end),
                'current_in_window': self.s_range[0] <= s <= self.s_range[1],
                'is_graspable': self.s_range[0] <= s <= self.s_range[1],
                'feasible': bool(feasible), 'reason': 'ok' if feasible else 'target_left_access_corridor'}

    def _refresh_prediction(self, now):
        current = _pose(self._motion_snapshot.current_grasp)
        velocity = self._motion_velocity()
        window = self._window(current, velocity, now)
        future_time = self._planned_time
        if future_time is None:
            lead = self.policy.config.prediction_lead_time
            if self._candidate_motion_plan is not None:
                from ..controllers.dynamic_intercept import intercept_lead
                lead = intercept_lead(current[:3], velocity, self.geometry.tcp()[:3, 3],
                    grasp_height=0., approach_clearance=self.policy.config.approach_height,
                    cartesian_speed=self.policy.config.cartesian_linear_speed,
                    minimum_motion_time=max(lead, self.policy.config.descent_lead_time + self._dispatch_dt),
                    latency=0., maximum_horizon=max(1e-6, window['end_time'] - now
                                                  - self.policy.config.intercept_deadline_margin))
            future_time = max(now + lead,
                              window['start_time'])
        predicted = current.copy()
        predicted[:3] += velocity * max(0., future_time - now)
        if self._planned_time is not None:
            # The reservation is labelled with its ORIGINAL capture time,
            # including after that time has passed. The acquisition guard
            # projects this pose from that time to now during occlusion.
            # Pairing a current pose with an expired time counted the elapsed
            # belt travel twice and drove the local request ahead of the part.
            self._planned_pose = current.copy()
            self._planned_pose[:3] += velocity * (float(self._planned_time) - now)
        observed_stamp = float(self._motion_snapshot.observed_timestamp) - self._policy_clock_origin
        age = max(0., now - observed_stamp)
        history = tuple({'timestamp': float(sample.timestamp) - self._policy_clock_origin,
                         'position': np.asarray(sample.position, dtype=float).copy()}
                        for sample in self._motion_snapshot.history)
        # The source snapshot describes the decision instant (arrival=0). Keep
        # that validity/confidence unchanged; a future intercept's uncertainty
        # is a separate diagnostic from the same real capture history.
        owner = getattr(self.snapshot_provider, '__self__', None)
        tracker = getattr(owner, 'tracker', owner)
        predictor = getattr(tracker, 'predictor', None)
        intercept_uncertainty = None
        uncertainty_reason = 'source_predictor_unavailable'
        if predictor is not None:
            try:
                future_prediction = predictor.predict(self._motion_snapshot.history,
                    current_time=now + self._policy_clock_origin,
                    robot_arrival_time=max(0., future_time - now),
                    estimated_velocity=velocity)
                intercept_uncertainty = np.asarray(future_prediction.uncertainty).copy()
                uncertainty_reason = 'same_capture_history_future_horizon'
            except (ValueError, TypeError) as error:
                uncertainty_reason = str(error)
        prediction = {'valid': bool(self._motion_snapshot.valid),
                      'reason': str(getattr(self._motion_snapshot, 'reason', 'ok')),
                      'observed_timestamp': observed_stamp, 'observation_age': age,
                      'clock_epoch':getattr(self._motion_snapshot,'clock_epoch',None),
                      'decision_timestamp': float(now),
                      'observed_pose': _pose(self._motion_snapshot.observed_grasp),
                      'current_estimated_pose': current.copy(),
                      'current_ee_grasp_pose': current.copy(),
                      'predicted_grasp_pose': predicted.copy(),
                      'estimated_velocity': velocity.copy(),
                      'prediction_velocity_model': (
                          'nero7_xy_with_measured_belt_plane_z'
                          if self._motion_snapshot.velocity_source == 'nero7_xy_kalman'
                          and self._measured_plane_normal is not None else 'source_velocity'),
                      'estimated_grasp_time': float(future_time),
                      'prediction_confidence': float(getattr(self._motion_snapshot, 'confidence', 0.)),
                      'prediction_uncertainty': np.asarray(getattr(self._motion_snapshot, 'uncertainty', np.zeros(3))).copy(),
                      'intercept_prediction_uncertainty': intercept_uncertainty,
                      'intercept_uncertainty_reason': uncertainty_reason,
                      'measurement_is_visual': bool(getattr(self._motion_snapshot, 'measured', True)),
                      'history_sample_count': len(history),
                      'visible': bool(self._motion_snapshot.visible),
                      'track_observation_age': age,
                      'velocity_source': str(self._motion_snapshot.velocity_source),
                      'grasp_window': window}
        speed_stamp = getattr(self._motion_snapshot, 'velocity_observed_timestamp', None)
        prediction['velocity_measurement'] = {
            'velocity_base_m_s': (np.asarray(self._motion_snapshot.velocity).copy()
                if self._motion_snapshot.velocity_source != 'unavailable' else None),
            'track_id': getattr(self._motion_snapshot, 'velocity_track_id', None),
            'binding_id': getattr(self._motion_snapshot, 'velocity_binding_id', None),
            'track_generation': getattr(self._motion_snapshot, 'velocity_track_generation', None),
            'device_timestamp': getattr(self._motion_snapshot, 'velocity_device_timestamp', None),
            'host_capture_timestamp': speed_stamp,
            'age_s': None if speed_stamp is None else max(0., now+self._policy_clock_origin-speed_stamp),
            'sample_count': getattr(self._motion_snapshot, 'velocity_sample_count', None),
            'span_s': getattr(self._motion_snapshot, 'velocity_span_s', None),
            'speed_std_m_s': getattr(self._motion_snapshot, 'velocity_speed_std_m_s', None),
            'std_interpretation': 'Kalman covariance diagnostic; not independent accuracy'}
        if self._planned_time is not None:
            prediction['planned_grasp_time'] = float(self._planned_time)
            prediction['planned_grasp_pose'] = self._planned_pose.copy()
        if prediction['velocity_source'] == 'unavailable':
            confirmation_source = 'visual_velocity_unavailable'
        elif prediction['measurement_is_visual']:
            confirmation_source = 'measured_rgbd_moving_target'
        elif prediction['velocity_source'] in ('vision', 'nero7_xy_kalman'):
            confirmation_source = 'real_visual_history_during_occlusion'
        else:
            confirmation_source = 'nominal_motion_prior'
        self._target.update(position=current[:3].copy(), pose=current.copy(),
                            observed_pose=prediction['observed_pose'].copy(),
                            timestamp=float(now), observed_timestamp=observed_stamp,
                            history=history, estimated_velocity=velocity.copy(), velocity=velocity.copy(),
                            ee_grasp_pose=predicted.copy(), predicted_ee_grasp_pose=predicted.copy(),
                            predicted_grasp_pose=predicted.copy(),
                            estimated_grasp_time=float(future_time),
                            visible=prediction['visible'], track_observation_age=age,
                            prediction=prediction, grasp_window=window,
                            prediction_valid=prediction['valid'],
                            confidence=prediction['prediction_confidence'],
                            velocity_source=prediction['velocity_source'],
                            measurement_is_visual=prediction['measurement_is_visual'],
                            confirmation_source=confirmation_source)
        if self._planned_time is not None:
            self._target['planned_grasp_time'] = float(self._planned_time)
            self._target['planned_grasp_pose'] = self._planned_pose.copy()
        if self._last_observation is not None:
            self._last_observation['predictions'] = {self._target['object_id']: prediction}
            self._last_observation['objects'] = [] if self._released else [self._target.copy()]
        return prediction

    def observe(self, state, timestamp, *, joint_velocity=None):
        observation = super().observe(state, timestamp, joint_velocity=joint_velocity)
        monitor = getattr(self, 'full_plan_scene_monitor', None)
        if monitor is not None:
            monitor.publish_state(state, timestamp)
        self._last_observation = observation
        if self._capture_pose is None:
            snapshot = self._snapshot(timestamp)
            if snapshot is None:
                prediction = dict(self._target.get('prediction', {}))
                prediction.update(valid=False, reason=self._perception_error or 'prediction_unavailable')
                self._target.update(prediction=prediction, prediction_valid=False)
                observation['predictions'] = {self._target['object_id']: prediction}
                observation['objects'] = [self._target.copy()]
                observation['moving_perception_diagnostics'] = {'available': False,
                    'reason': prediction['reason']}
                observation.update(belt_velocity=None, belt_speed=None,
                                   belt_velocity_valid=False, motion_source='unavailable')
                return observation
            self._motion_snapshot = snapshot
            self._perception_error = None
            # No source measurement timestamp is replaced with this tick's time.
            # target.timestamp labels the current *estimate*; observed_timestamp
            # and history retain the actual RGB-D capture times independently.
            for name in ('current_grasp', 'observed_grasp', 'grasp'):
                _readonly(getattr(self._motion_snapshot, name), (4, 4))
            velocity = self._motion_velocity()
            self._refresh_prediction(timestamp)
            velocity_available = snapshot.velocity_source != 'unavailable'
            observation['belt_velocity'] = velocity.copy() if velocity_available else None
            observation['belt_speed'] = float(velocity @ self.belt_direction) if velocity_available else None
            observation['belt_velocity_valid'] = bool(velocity_available and snapshot.valid)
            observation['motion_source'] = str(self._motion_snapshot.velocity_source)
            observation['nominal_belt_speed_m_s'] = self.nominal_speed_m_s
            observation['moving_perception_diagnostics'] = {
                'available': True, 'valid': bool(snapshot.valid),
                'reason': str(getattr(snapshot, 'reason', 'ok')),
                'velocity_source': str(snapshot.velocity_source),
                'velocity_available': velocity_available,
                'velocity_measurement': self._target['prediction']['velocity_measurement'],
                'measurement_is_visual': bool(getattr(snapshot, 'measured', True)),
                'observation_age_s': float(timestamp + self._policy_clock_origin - snapshot.observed_timestamp)}
            self._live_collision = self._live_target_snapshot(timestamp, self._dispatch_dt)
        else:
            self._target['status'] = ('RELEASED' if self._released else
                'RELEASE_PENDING' if self._release_time is not None else 'GRASPED')
            observation['objects'] = [] if self._released else [self._target.copy()]
            observation['belt_velocity'] = self._last_visual_velocity.copy()
            observation['belt_speed'] = float(self._last_visual_velocity @ self.belt_direction)
            self._live_collision = (self._payload_collision_snapshot() if self._release_time is None
                                    else self._released_collision_snapshot())
        return observation

    def select_target(self, observation):
        if self._reserved or self._released or not self.is_target_feasible(self._target['object_id'], observation):
            return None
        self._reserved = True
        self._target['status'] = 'PLANNING'
        self._planned_time = float(self._target['estimated_grasp_time'])
        self._planned_pose = np.asarray(self._target['predicted_grasp_pose']).copy()
        self._refresh_prediction(float(observation['timestamp']))
        return self._target.copy()

    def get_predicted_grasp_pose(self, object_id, observation=None):
        if object_id != self._target['object_id']:
            return None
        predictions = (observation or self._last_observation or {}).get('predictions', {})
        value = predictions.get(object_id)
        return None if value is None or not value.get('valid', False) else np.asarray(value['predicted_grasp_pose']).copy()

    predict_grasp_pose = get_predicted_grasp_pose

    def is_target_feasible(self, object_id, observation=None):
        if object_id != self._target['object_id'] or self._released:
            return False
        prediction = self._target.get('prediction', {})
        window = prediction.get('grasp_window', {})
        if self.policy.config.planning_mode == 'full_preplanned':
            # The short display/preview forecast is not an appointment.  A
            # moving part upstream can have a valid future intercept window
            # even while that arbitrary forecast remains outside it. The full
            # planner chooses an actual contact time and checks its deadline.
            return bool(prediction.get('valid', False) and window.get('feasible', False))
        pose = np.asarray(prediction.get('predicted_grasp_pose', self.grasp_pose))
        return bool(prediction.get('valid', False) and window.get('feasible', False)
                    and self.s_range[0] <= self._coordinate(pose[:3]) <= self.s_range[1])

    is_grasp_window_feasible = is_target_feasible

    def is_target_graspable(self, object_id, observation=None):
        return bool(object_id == self._target['object_id'] and self._target.get('grasp_window', {}).get('current_in_window', False))

    is_in_grasp_window = is_target_graspable

    def preview_fixed_high_grasp_plan(self, object_id, high_pose, remaining_high_motion,
                                     descent_duration, *, capture_high=False, capture_time=None):
        """Predict crossing of a fixed high point from the current real track.

        This is a proposal only: it never moves the high waypoint, changes a
        capture timestamp, or refreshes the age of a camera measurement.
        """
        if (object_id != self._target['object_id'] or not self._reserved
                or self._motion_snapshot is None or not self._motion_snapshot.valid):
            return {'accepted': False, 'reason': 'fixed_high_visual_unavailable'}
        now = float(self.data.time)
        current = _pose(self._motion_snapshot.current_grasp)
        velocity = self._motion_velocity()
        speed2 = float(velocity[:2] @ velocity[:2])
        # Only a truly zero measured XY vector uses the stationary branch;
        # no velocity deadzone or fabricated belt-speed prior is introduced.
        crossing = float((np.asarray(high_pose)[:2]-current[:2]) @ velocity[:2])/speed2 if speed2 > 0. else 0.
        descent_start = crossing - (descent_duration if capture_high else 0.)
        if capture_time is None:
            capture_time = now + max(float(remaining_high_motion), descent_start, 0.) + descent_duration
        elif float(capture_time) < now:
            return {'accepted': False, 'reason': 'fixed_high_checked_capture_elapsed'}
        capture_time = float(capture_time)
        predicted = current.copy()
        predicted[:3] += velocity*(capture_time-now)
        window = self._window(current, velocity, now)
        if (capture_time > window['end_time']-self.policy.config.intercept_deadline_margin
                or not self.s_range[0] <= self._coordinate(predicted[:3]) <= self.s_range[1]):
            return {'accepted': False, 'reason': 'fixed_high_capture_outside_window'}
        return {'accepted': True, 'reason': 'ok', 'estimated_grasp_time': capture_time,
                'predicted_grasp_pose': predicted, 'crossing_time': now+crossing}

    def refine_selected_grasp_plan(self, object_id, required_motion_time, deadline_margin=.25, *,
                                   allow_retime=True, final_tracking_correction_limit=None,
                                   requested_time=None, requested_pose=None):
        now = float(self.data.time)
        motion, margin = float(required_motion_time), float(deadline_margin)
        if (object_id != self._target['object_id'] or not self._reserved
                or not np.isfinite([motion, margin]).all() or min(motion, margin) < 0
                or self._motion_snapshot is None or not self._motion_snapshot.valid):
            return {'accepted': False, 'reason': 'invalid_intercept_timing'}
        current = _pose(self._motion_snapshot.current_grasp)
        velocity = self._motion_velocity()
        window = self._window(current, velocity, now)
        earliest = max(now + motion, window['start_time'])
        latest = window['end_time'] - margin
        if earliest > latest + 1e-9:
            return {'accepted': False, 'reason': 'insufficient_intercept_time',
                    'required_grasp_time': earliest, 'latest_grasp_time': latest,
                    'window_end': window['end_time']}
        exact = requested_time is not None or requested_pose is not None
        if exact:
            try:
                proposed_time = float(requested_time)
                proposed_pose = _readonly(requested_pose, (7,)).copy()
            except (TypeError, ValueError):
                return {'accepted': False, 'reason': 'invalid_requested_intercept'}
            if not np.isfinite(proposed_time) or not earliest - 1e-9 <= proposed_time <= latest + 1e-9:
                return {'accepted': False, 'reason': 'insufficient_intercept_time'}
            predicted = current.copy(); predicted[:3] += velocity * max(0., proposed_time - now)
            # A high waypoint supplies XY only. Preserve the latest observed
            # grasp's height above the plane at those XY, not at the unrelated
            # constant-velocity forecast XY. Never copy the high waypoint Z.
            if (self._motion_snapshot.velocity_source == 'nero7_xy_kalman'
                    and self._measured_plane_normal is not None):
                displacement = belt_plane_motion_from_xy(
                    np.r_[proposed_pose[:2]-current[:2], 0.], self._measured_plane_normal)
                predicted[:3] = current[:3]+displacement
            predicted[:2] = proposed_pose[:2]
            if self.moving_approach_mode == 'continuous':
                # A continuous request certifies the complete contact pose,
                # while a high request historically supplies only XY. Reject
                # a changed height/orientation BEFORE mutating reservations.
                from ..controllers._math import quat_distance
                if (np.linalg.norm(predicted[:3]-proposed_pose[:3])
                        > self.policy.config.trajectory_replan_position_tolerance
                        or quat_distance(predicted[3:], proposed_pose[3:])
                        > self.policy.config.trajectory_replan_orientation_tolerance):
                    return {'accepted': False, 'reason': 'certified_contact_pose_changed',
                            'predicted_grasp_pose': predicted.copy()}
        else:
            keep = self._planned_time is not None and earliest - 1e-9 <= self._planned_time <= latest + 1e-9
            proposed_time = self._planned_time if keep else earliest
            predicted = current.copy(); predicted[:3] += velocity * max(0., proposed_time - now)
            if not allow_retime and not keep:
                try:
                    correction_limit = float(final_tracking_correction_limit)
                except (TypeError, ValueError):
                    correction_limit = 0.
                correction = float('inf') if self._planned_pose is None else float(np.linalg.norm(predicted[:3]-self._planned_pose[:3]))
                if correction_limit <= 0 or correction > correction_limit + 1e-9:
                    return {'accepted': False, 'reason': 'insufficient_intercept_time'}
        # Float64 projection onto the belt axis may round an exact boundary
        # by a few ulps; retain the workspace, allowing only numerical slack.
        coordinate = self._coordinate(predicted[:3])
        if not self.s_range[0]-1e-9 <= coordinate <= self.s_range[1]+1e-9:
            return {'accepted': False, 'reason': 'trajectory_outside_reachable_workspace'}
        changed = self._planned_time is None or abs(self._planned_time-proposed_time) > 1e-9
        self._planned_time = float(proposed_time)
        self._planned_pose = predicted.copy()
        record = {'timestamp': now, 'estimated_grasp_time': self._planned_time,
                  'predicted_grasp_pose': predicted.tolist(), 'requested_exact': exact}
        self._intercept_updates.append(record)
        self._refresh_prediction(now)
        # Exact request coordinates are an already certified endpoint, not an
        # excuse to change the visual velocity or invent a fresh measurement.
        if exact:
            self._planned_pose = predicted.copy()
            for target in (self._target, *self._last_observation.get('objects', [])):
                target['ee_grasp_pose'] = predicted.copy()
                target['predicted_ee_grasp_pose'] = predicted.copy()
                target['predicted_grasp_pose'] = predicted.copy()
                target['planned_grasp_pose'] = predicted.copy()
            prediction = self._last_observation['predictions'][object_id]
            prediction['predicted_grasp_pose'] = predicted.copy()
            prediction['planned_grasp_pose'] = predicted.copy()
        return {'accepted': True, 'reason': 'ok', 'changed': changed,
                'estimated_grasp_time': self._planned_time,
                'predicted_grasp_pose': predicted.copy(),
                'predicted_ee_grasp_pose': predicted.copy(), 'window_end': window['end_time'],
                'deadline_slack': float(latest-self._planned_time)}

    retime_selected_grasp_plan = refine_selected_grasp_plan

    def validate_selected_grasp_plan(self, object_id, required_motion_time, deadline_margin=.25,
                                     final_tracking_correction_limit=.012):
        return self.refine_selected_grasp_plan(object_id, required_motion_time, deadline_margin,
            allow_retime=False, final_tracking_correction_limit=final_tracking_correction_limit)

    def _capture(self):
        if self._capture_pose is not None:
            return
        held = self.policy._secured_hold_pose
        if held is None:
            raise PickRejected('没有抓牢后的实测TCP，不能建立放回位置')
        self._capture_pose = _readonly(held, (7,))
        self._capture_time = float(self.data.time)
        self._capture_width = float(self._last_observation['gripper']['opening_width'])
        tcp = _matrix(held)
        visible_points = self._points_at(_pose(self._motion_snapshot.current_grasp))
        self._payload_local = _readonly((visible_points-tcp[:3, 3]) @ tcp[:3, :3])
        self._payload_grasp_local = _readonly(np.linalg.inv(tcp) @ self._motion_snapshot.current_grasp, (4, 4))
        self._last_visual_velocity = self._motion_velocity()
        self._last_measured_velocity = np.asarray(self._motion_snapshot.velocity).copy()
        self.grasp_pose = np.asarray(held).copy()
        self.grasp_transform = tcp.copy()
        self.lift_pose = self.grasp_pose.copy(); self.lift_pose[2] += self.settings.lift_m
        self.wait_pose = self.grasp_pose.copy(); self.wait_pose[2] += .10
        self.policy._wait_pose = self.wait_pose.copy()
        self.policy.config = replace(self.policy.config, safe_transfer_height=float(self.lift_pose[2]))
        self._target['status'] = 'GRASPED'
        self._live_collision = self._payload_collision_snapshot()

    def get_lift_pose(self, observation=None):
        if self.policy._secured_hold_pose is not None:
            self._capture()
        elif self._capture_pose is None:
            # Speculative route queries precede physical grasp confirmation.
            # They may inspect the current estimate but cannot latch a capture.
            pose = np.asarray(self._target.get('predicted_grasp_pose', self.grasp_pose)).copy()
            pose[2] += self.settings.lift_m
            return pose
        return self.lift_pose.copy()

    def get_bin_drop_pose(self, bin_index):
        if self._capture_pose is None:
            return np.asarray(self._target.get('predicted_grasp_pose', self.grasp_pose)).copy()
        return self.grasp_pose.copy()

    def get_release_pose(self, observation=None):
        """Use the shared RELEASING servo to clear the moving placed part.

        A motor-force drop together with an actual wider jaw opening is a
        release proxy, not a pad-contact measurement. Put-back starts at the
        fixed measured capture position; after opening starts, following the
        belt while raising keeps the next moving surface away from the fingers.
        """
        observation = observation or self._last_observation
        if self._capture_pose is None or observation is None:
            return self.grasp_pose.copy()
        jaw = observation['gripper']
        if self._release_time is None:
            opened = float(jaw['opening_width']) > self._capture_width
            unloaded = float(jaw['motor_force_magnitude_n']) < self.settings.min_force_n
            if not (opened and unloaded):
                return self.grasp_pose.copy()
            tcp = _matrix(np.asarray(observation['ee_pose']))
            self._released_pose = _readonly(_pose(tcp @ self._payload_grasp_local))
            self._release_time = float(observation['timestamp'])
            self._release_tcp = _readonly(np.asarray(observation['ee_pose']), (7,))
            self._live_collision = self._released_collision_snapshot()
        pose = self._release_tcp.copy()
        elapsed = max(0., float(observation['timestamp'])-self._release_time)
        speed = self.policy.config.cartesian_linear_speed
        # This nominal speed estimates the retreat arrival horizon; it is
        # not a command cap when local_tcp_linear_speed_limit_enabled=False.
        # The retreat Z is a fixed world height. Only the XY belt component
        # enters this estimate; tilted belt Z belongs to the moving part.
        horizontal_speed = float(np.linalg.norm(self._last_visual_velocity[:2]))
        vertical_speed = float(np.sqrt(max(0., speed*speed-horizontal_speed*horizontal_speed)))
        remaining_height = max(0., float(self.wait_pose[2])-float(observation['ee_pose'][2]))
        # Use one arrival horizon for belt displacement AND withdrawal. A
        # 100 mm Z goal with only 40 ms of lateral feed-forward incorrectly
        # normalizes almost all Cartesian speed onto Z. Conversely a short
        # measured-state Z increment can stall behind joint profiling. Keep
        # the actual retreat goal and predict its simultaneous belt position.
        horizon = self.policy.config.tracking_velocity_feedforward_time
        if vertical_speed > 0.:
            horizon = max(horizon, remaining_height/vertical_speed)
        pose[:2] += self._last_visual_velocity[:2]*(elapsed+horizon)
        pose[2] = float(self.wait_pose[2])
        return pose

    def is_release_clear(self, observation=None):
        """Hand an already detached tool to the shared fixed-position return.

        The 100 mm height is the final return target, not a requirement to
        chase the belt all the way to that height. Once both measured finger
        meshes are above the released visible surface and its existing object
        collision margin, RETURNING can finish that target on its own route.
        """
        observation = observation or self._last_observation
        if observation is None or self._release_time is None:
            return False
        collision = self._released_collision_snapshot()
        if not len(collision.fingers.points):
            return False
        part_top = float(np.max(collision.fingers.points[:, 2]))
        finger_bottoms = []
        for geom in self.geometry.fingers:
            vertices = (self.geometry.mesh_vertices[geom]
                @ self.data.geom_xmat[geom].reshape(3, 3).T
                + self.data.geom_xpos[geom])
            finger_bottoms.append(float(np.min(vertices[:, 2])))
        # Reuse the same 4 mm visible-object margin as _cloud_safe(). This
        # geometric handoff is not an abort condition or another timed dwell.
        margin = .004
        self._release_clearance = {
            'policy_time_s': float(observation['timestamp']),
            'released_visible_cloud_top_z_m': part_top,
            'measured_finger_bottom_z_m': finger_bottoms,
            'existing_object_collision_margin_m': margin,
            'clear': bool(finger_bottoms and min(finger_bottoms) >= part_top + margin),
            'final_fixed_retreat_target_z_m': float(self.wait_pose[2])}
        return self._release_clearance['clear']

    def on_policy_event(self, event):
        if event.name == 'lift_started':
            self._capture()
        super().on_policy_event(event)
        if event.name == 'release_complete' and self._release_time is None:
            tcp = _matrix(np.asarray(self._last_observation['ee_pose']))
            self._released_pose = _readonly(_pose(tcp @ self._payload_grasp_local))
            self._release_time = float(self.data.time)
            self._release_tcp = _readonly(np.asarray(self._last_observation['ee_pose']), (7,))
            self._live_collision = self._released_collision_snapshot()

    def _points_at(self, pose, local=None):
        matrix = _matrix(pose)
        points = self._template_local if local is None else local
        return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]

    def _live_target_snapshot(self, now, dt):
        pose = _pose(self._motion_snapshot.current_grasp)
        velocity = self._motion_velocity()
        key = (float(now), float(dt), pose.tobytes(), velocity.tobytes(), id(self._template_local))
        if getattr(self, '_live_target_snapshot_cache', (None,))[0] == key:
            return self._live_target_snapshot_cache[1]
        next_pose = pose.copy(); next_pose[:3] += velocity * max(0., dt)
        count = max(1, int(np.ceil(np.linalg.norm(next_pose[:3]-pose[:3])/.004)))
        cloud = _Cloud.build(np.concatenate([self._points_at(np.r_[pose[:3] + velocity*dt*u, pose[3:]])
                    for u in np.linspace(0., 1., count+1)]))
        # The two-millimetre sweep-sampling allowance preserves the original
        # four-millimetre measured-object collision envelope.
        result = _CollisionSnapshot('tracking', cloud, cloud, (_readonly(pose),),
                                    non_finger_margin=.006)
        self._live_target_snapshot_cache = (key, result)
        return result

    def _payload_collision_snapshot(self):
        return _CollisionSnapshot('payload', self._empty_cloud, self._empty_cloud,
            (), self._payload_local, _readonly(self.grasp_pose))

    def _released_collision_snapshot(self):
        pose = self.grasp_pose.copy() if self._released_pose is None else self._released_pose.copy()
        reference = pose.copy() if self._release_tcp is None else self._release_tcp.copy()
        if self._release_time is not None:
            offset = self._last_visual_velocity * max(0., float(self.data.time)-self._release_time)
            pose[:3] += offset
            reference[:3] += offset
        cloud = _Cloud.build(self._points_at(pose))
        # The real hold was confirmed at the measured TCP, whose visual part
        # center may retain a residual offset. Opening contact uses that actual
        # contact frame, rather than reimposing the pre-grasp visual-center
        # tolerance immediately after the jaws start opening.
        return _CollisionSnapshot('released', cloud, cloud, (_readonly(reference),))

    def _request_collision_snapshot(self, request):
        self._record_planning_request(request, source='submitted')
        if self._capture_pose is not None and self._release_time is None:
            return self._payload_collision_snapshot()
        if self._release_time is not None:
            return self._released_collision_snapshot()
        # The global engine has no per-sample absolute execution times. Freeze
        # requested capture geometry only: an additional permanent copy at the
        # current position can falsely block the future open finger. Actual
        # moving-object interaction is checked on each dispatched short edge.
        return _planning_target_snapshot(request, self._template_local, self.policy.config.approach_height)

    def _record_planning_request(self, request, *, source):
        self._planner_requests.append({'request_id': getattr(request, 'request_id', None),
            'moving_approach_mode': self.moving_approach_mode,
            'continuous_intercept': bool(getattr(request, 'continuous_intercept', False)),
            'stage': getattr(request, 'stage', None),
            'planner_seed': getattr(request, 'planner_seed', None),
            'planned_target_time_s': getattr(request, 'planned_target_time_s', None),
            'request_source': source,
            'request_snapshot_schema_version': 1,
            'request_snapshot': (asdict(request) if is_dataclass(request) else None),
            'continuation_target_poses': [np.asarray(pose).tolist() for pose in
                (getattr(request, 'continuation_target_poses', None) or ())]})
        emit = getattr(self, 'diagnostic_emit', None)
        if source == 'submitted' and callable(emit):
            emit('planning_progress', {'status': 'requested',
                'request_id': request.request_id, 'stage': request.stage,
                'moving_approach_mode': self.moving_approach_mode,
                'kind': ('continuous_intercept' if getattr(request, 'continuous_intercept', False)
                         else 'descent_check' if getattr(request, 'fixed_high_continuation', False)
                         else 'route'),
                'requested_policy_time_s': float(self.data.time)})

    def _record_effective_planning_request(self, request):
        self._record_planning_request(request, source='worker_effective_intercept')

    def _record_planning_result(self, result, request):
        path = getattr(result, 'path', None)
        self._planner_results.append({
            'request_id': result.request_id,
            'moving_approach_mode': self.moving_approach_mode,
            'continuous_intercept': bool(getattr(request, 'continuous_intercept', False)),
            'stage': getattr(result, 'stage', getattr(request, 'stage', None)),
            'received_policy_time_s': float(self.data.time),
            'planned_target_time_s': getattr(request, 'planned_target_time_s', None),
            'cost': getattr(result, 'cost', None),
            'planning_time_s': getattr(result, 'planning_time_s', None),
            'start_joints_rad': None if not hasattr(result, 'start_q') else np.asarray(result.start_q).tolist(),
            'goal_joints_rad': None if not path else np.asarray(path[-1]).tolist(),
            'target_pose7': None if not hasattr(result, 'target_pose') else np.asarray(result.target_pose).tolist(),
            'edge_durations_s': getattr(result, 'edge_durations', None),
            'path_fully_validated': getattr(result, 'path_fully_validated', None),
            'continuation_fully_validated': getattr(result, 'continuation_fully_validated', None),
            'continuation_path': getattr(result, 'continuation_path', None),
            'continuation_trajectory_duration_s': getattr(
                getattr(result, 'continuation_trajectory', None), 'duration', None),
            'continuation_trajectory_backend': getattr(
                getattr(result, 'continuation_trajectory', None), 'backend', None),
            'continuation_target_pose': getattr(result, 'continuation_target_pose', None),
            'full_pick_schedule': (None if getattr(result, 'full_pick_plan', None) is None else
                {'start_s': result.full_pick_plan.start, 'high_s': result.full_pick_plan.high_time,
                 'contact_s': result.full_pick_plan.contact_time, 'lift_start_s': result.full_pick_plan.lift_time,
                 'end_s': result.full_pick_plan.end, 'certified_samples': result.full_pick_plan.checked_samples,
                 'certified_command_samples': result.full_pick_plan.checked_command_samples,
                 'response_compensation_applied': result.full_pick_plan.command_curves is not None}),
            'diagnostics': getattr(result, 'diagnostics', None),
            'error': getattr(result, 'error', None), 'cancelled': getattr(result, 'cancelled', None)})
        emit = getattr(self, 'diagnostic_emit', None)
        if callable(emit):
            emit('planning_progress', {'status': 'returned',
                'request_id': result.request_id,
                'stage': getattr(result, 'stage', getattr(request, 'stage', None)),
                'moving_approach_mode': self.moving_approach_mode,
                'kind': ('continuous_intercept' if getattr(request, 'continuous_intercept', False)
                         else 'descent_check' if getattr(request, 'fixed_high_continuation', False)
                         else 'route'),
                'received_policy_time_s': float(self.data.time),
                'planned_target_time_s': getattr(request, 'planned_target_time_s', None),
                'planning_time_s': getattr(result, 'planning_time_s', None),
                'error': getattr(result, 'error', None),
                'cancelled': getattr(result, 'cancelled', None)})

    def _cloud_safe(self, geometry, collision, *, worker_request=None, rejection_details=None):
        if collision is None:
            return True
        if collision.kind == 'payload':
            tcp = geometry.tcp()
            points = collision.payload_local @ tcp[:3, :3].T + tcp[:3, 3]
            cloud = _Cloud.build(points)
            non_finger = cloud
            if len(points) and len(geometry.scene):
                distance, index = geometry.tree.query(points)
                close = distance < .008
                if close.any():
                    # Belt support is intentional only at the fixed put-back
                    # corridor during initial lift/final lowering. Raised
                    # obstacles and target contact elsewhere remain rejected.
                    support = geometry.scene_region_plane_to_base
                    if support is None:
                        if rejection_details is not None:
                            rejection_details.update(reason='payload_scene_without_support_plane')
                        return False
                    normal, origin = support[:3, 2], support[:3, 3]
                    nearby = geometry.scene[index[close]]
                    on_support = np.abs((nearby-origin) @ normal) <= .005
                    drop = collision.drop_pose
                    near_drop = (np.linalg.norm(tcp[:2, 3]-drop[:2]) <= .010
                                 and abs(tcp[2, 3]-drop[2]) <= .015)
                    if not near_drop or not on_support.all():
                        if rejection_details is not None:
                            rejection_details.update(reason='payload_scene_collision',
                                near_drop=bool(near_drop), on_support=bool(on_support.all()))
                        return False
        else:
            non_finger = collision.non_finger
        broadphase = getattr(geometry, 'target_broadphase_candidates', None)
        target_candidates = (broadphase(non_finger.points, collision.non_finger_margin)
                             if broadphase is not None and len(non_finger.points) else None)
        for geom, (_, radius) in geometry.meshes.items():
            finger = geom in geometry.fingers
            # All publications here belong to the one selected workpiece.
            # Allow its contact with both fingers, independent of uncertain
            # TCP/visual alignment. The parent checker still checks self and
            # scene collisions; non-finger links still check this target.
            if finger:
                continue
            if target_candidates is not None and geom not in target_candidates:
                continue
            cloud = non_finger
            if not len(cloud.points):
                continue
            margin = collision.non_finger_margin
            # Collision radii cover an 8 mm padded hull; our 4/6 mm envelopes
            # are strictly inside it, including extended convex corners.
            ids = cloud.tree.query_ball_point(geometry.data.geom_xpos[geom], geometry.collision_radii[geom]+1e-9)
            if ids and geometry.inside_geom(cloud.points[ids], geom, margin).any():
                collision_rejection = {
                    'kind': collision.kind, 'geom': geometry.model.geom(geom).name,
                    'policy_time_s': float(self.data.time) if worker_request is None else None,
                    'planned_target_time_s': getattr(worker_request, 'planned_target_time_s', None),
                    'validation_source': 'live_short_edge' if worker_request is None else 'frozen_worker_request',
                    'tcp_pose': _pose(geometry.tcp()).tolist(),
                    'target_contact_poses': [pose.tolist() for pose in collision.contact_poses],
                    'point_count_near_geom': len(ids), 'margin_m': margin}
                self._collision_rejection = collision_rejection
                if rejection_details is not None:
                    rejection_details.update(reason='selected_target_nonfinger_collision',
                                             details=collision_rejection)
                return False
        return True

    def _geometry_valid(self, geometry, q, width, *, allow_start_region=None,
                        collision_snapshot=None, worker_request=None, rejection_details=None,
                        scene_prevalidated=False):
        # The parent checker owns native self-collision and static scene tests.
        # Dynamic target geometry lives in a separate immutable publication.
        geometry.target = self._empty_cloud.points
        if not super()._geometry_valid(geometry, q, width, allow_start_region=allow_start_region,
                                      rejection_details=rejection_details,
                                      scene_prevalidated=scene_prevalidated):
            return False
        return self._cloud_safe(geometry, self._live_collision if collision_snapshot is None else collision_snapshot,
                                worker_request=worker_request, rejection_details=rejection_details)

    def _batch_cloud_safe(self, geometry, states, width, collision, request, rejected):
        """Final non-finger hull tests for one immutable worker publication.

        Payload support contact remains under _cloud_safe's original per-pose
        rules. Planning/released clouds are unchanged world point arrays, so
        their exact KD-tree/hull tests can share a batch of already computed FK.
        """
        if collision.kind not in {'planning', 'released'}:
            return None
        shared = getattr(geometry, '_state_evaluator', None)
        if shared is None:
            return None
        geoms = tuple(g for g in geometry.meshes if g not in geometry.fingers)
        if not geoms or not len(collision.non_finger.points):
            return True
        full = request.full_qpos.copy()
        full[geometry.model.joint('gripper_joint1').qposadr] = width/2
        full[geometry.model.joint('gripper_joint2').qposadr] = -width/2
        features = []
        for q in states:
            full[geometry.indices] = q
            feature = shared._states.get(full.tobytes())
            if feature is None:
                return None
            features.append(feature)
        from .gpu_scene_collision import ObservedPointCollisionIndex
        cached = getattr(geometry, '_worker_target_batch_index', None)
        if cached is None or cached[0] is not collision:
            index = ObservedPointCollisionIndex(collision.non_finger.points,
                {g: geometry.meshes[g][0] for g in geoms},
                backend='cpu', tree=collision.non_finger.tree)
            geometry._worker_target_batch_index = (collision, index)
        else:
            index = cached[1]
        positions = np.array([feature.poses['geom_xpos'][list(geoms)] for feature in features])
        rotations = np.array([feature.poses['geom_xmat'][list(geoms)].reshape(-1, 3, 3)
                              for feature in features])
        margin = collision.non_finger_margin
        bounds = np.array([geometry.mesh_bounds[g, float(margin)] for g in geoms])
        found = index.query_batch(positions, rotations,
            np.array([geometry.collision_radii[g]+1e-9 for g in geoms]),
            np.full(len(geoms), margin), bounds)
        for state_index, collisions in enumerate(found):
            for geom in geoms:
                indices = collisions.get(geom)
                if indices is None or not len(indices):
                    continue
                geometry.set_state(states[state_index], width)
                details = {'kind': collision.kind, 'geom': geometry.model.geom(geom).name,
                    'policy_time_s': None,
                    'planned_target_time_s': getattr(request, 'planned_target_time_s', None),
                    'validation_source': 'frozen_worker_request',
                    'tcp_pose': _pose(geometry.tcp()).tolist(),
                    'target_contact_poses': [pose.tolist() for pose in collision.contact_poses],
                    'point_count_near_geom': len(index.tree.query_ball_point(
                        positions[state_index, geoms.index(geom)], geometry.collision_radii[geom]+1e-9)),
                    'margin_m': margin,
                    'validation': 'batched_native_points_full_convex_hulls'}
                self._collision_rejection = details
                rejected.update(reason='selected_target_nonfinger_collision', details=details)
                return False
        return True

    def prepare_motion_snapshot(self, observation):
        """Sweep until the next geometry check without changing the send clock.

        A streamed joint edge can send more often than its full geometry
        review. That review supplies its own horizon; ordinary callers keep
        the existing one-dispatch-interval sweep. The target still starts at
        the current estimate made from its original capture timestamp.
        """
        self._dispatch_dt = float(observation.get('control_dt', 1./self.settings.control_hz))
        horizon = self._dispatch_dt
        try:
            requested = float(observation.get('geometry_validation_horizon_s', horizon))
        except (TypeError, ValueError):
            requested = horizon
        if np.isfinite(requested) and requested >= self._dispatch_dt:
            horizon = requested
        self.geometry_validation_stats.update(
            moving_target_sweep_horizon_s=None,
            moving_target_sweep_start_policy_time_s=None,
        )
        if self._capture_pose is None and self._motion_snapshot is not None:
            now = float(self.data.time)
            self._live_collision = self._live_target_snapshot(now, horizon)
            self.geometry_validation_stats.update(
                moving_target_sweep_horizon_s=horizon,
                moving_target_sweep_start_policy_time_s=now,
            )

    def validate_command(self, state, command):
        self.prepare_motion_snapshot(self._last_observation)
        return super().validate_command(state, command)

    def is_arm_configuration_safe(self, q, **kwargs):
        kwargs.setdefault('minimum_downward_alignment', self.policy._local_downward_alignment_limit())
        return super().is_arm_configuration_safe(q, **kwargs)

    def can_reuse_planning_certificate(self, request, result, observation):
        # Static geometry and jaw configuration use the shared immutable
        # certificate. The moving target publication is separately checked on
        # the actual short command; it does not trigger full-route revalidation.
        return super().can_reuse_planning_certificate(request, result, observation)

    def target_state_record(self, now):
        """Small trajectory/evidence record without inventing camera samples."""
        now = float(now)
        if self._capture_pose is None:
            if self._motion_snapshot is None:
                return None
            grasp = np.asarray(self._motion_snapshot.current_grasp).copy()
            velocity = np.asarray(self._motion_snapshot.velocity).copy()
            observed = float(self._motion_snapshot.observed_timestamp)-self._policy_clock_origin
            source = str(self._motion_snapshot.velocity_source)
            phase = 'belt'
        elif self._release_time is None:
            tcp = _matrix(np.asarray(self._last_observation['ee_pose']))
            grasp = tcp @ self._payload_grasp_local
            velocity = self._last_measured_velocity.copy()
            observed = self._capture_time
            source, phase = 'captured_relative_to_measured_tcp', 'held'
        else:
            grasp = _matrix(self._released_pose)
            grasp[:3, 3] += self._last_visual_velocity*max(0., now-self._release_time)
            velocity = self._last_measured_velocity.copy()
            observed = self._release_time
            source, phase = 'released_last_visual_velocity_extrapolation', 'released'
        return {'policy_time_s': now, 'grasp_tcp': grasp.tolist(),
                'intercept_tcp': None if self._planned_pose is None else _matrix(self._planned_pose).tolist(),
                'velocity_m_s': velocity.tolist() if source != 'unavailable' else None,
                'prediction_velocity_m_s': (self._motion_velocity().tolist()
                    if phase == 'belt' and source != 'unavailable' else
                    self._last_visual_velocity.tolist() if phase != 'belt' else None),
                'prediction_velocity_model': self._target.get('prediction', {}).get('prediction_velocity_model'),
                'observed_policy_time_s': observed,
                'observation_age_s': max(0., now-observed), 'velocity_source': source,
                'phase': phase,
                'acquisition_control': self.policy.acquisition_diagnostics()}

    def create_global_planning_worker(self):
        geometry = _copy_geometry(self.geometry)
        self._scene_indexes.append(geometry.scene_collision_index())
        self._scene_indexes[-1].prepare()
        geometry.target = self._empty_cloud.points
        snapshots = {}
        lock = threading.Lock()
        # These inputs are immutable throughout this worker's lifetime. The
        # validator never reads the live target, clock, or policy dictionaries.
        template_local = _readonly(self._template_local)
        approach_height = float(self.policy.config.approach_height)
        def collision_for(request):
            with lock:
                publication = snapshots.get(request.request_id)
            if publication is None:
                return None
            cached_request, collision = publication
            if (collision.kind == 'planning' or getattr(request, 'planning_mode', 'legacy') == 'full_preplanned') and cached_request is not request:
                collision = _planning_target_snapshot(request, template_local, approach_height)
                with lock:
                    snapshots[request.request_id] = (request, collision)
            return collision
        def validator(request, q):
            validator.last_rejection = {}
            collision = collision_for(request)
            if collision is None:
                validator.last_rejection = {'reason': 'missing_request_collision_snapshot'}
                return False
            width = float(request.full_qpos[geometry.model.joint('gripper_joint1').qposadr][0])*2
            return self._geometry_valid(geometry, q, width,
                allow_start_region=request.stage == 'pregrasp', collision_snapshot=collision,
                worker_request=request, rejection_details=validator.last_rejection)
        def batch(request, states):
            validator.last_rejection = {}
            collision = collision_for(request)
            if collision is None:
                validator.last_rejection = {'reason': 'missing_request_collision_snapshot'}
                return False
            width = float(request.full_qpos[geometry.model.joint('gripper_joint1').qposadr][0])*2
            # For a frozen point cloud, original native/self/scene checks and
            # the selected-target narrow phase can each consume the same FK
            # batch. Payload support/drop semantics use the original checker.
            if collision.kind in {'planning', 'released'}:
                target_ok = self._batch_cloud_safe(geometry, states, width, collision,
                                                   request, validator.last_rejection)
                if target_ok is not None:
                    if not target_ok:
                        return False
                    return self._batch_scene_validation(geometry, states,
                        lambda q: NeroPolicyContext._geometry_valid(self, geometry, q, width,
                            allow_start_region=request.stage == 'pregrasp',
                            scene_prevalidated=True, rejection_details=validator.last_rejection),
                        validator.last_rejection)
            target_prepare = getattr(geometry, 'prepare_target_broadphase', None)
            if callable(target_prepare):
                target_prepare(states, width, collision.non_finger.points,
                               collision.non_finger_margin)
            return self._batch_scene_validation(geometry, states,
                lambda q: self._geometry_valid(geometry, q, width,
                    allow_start_region=request.stage == 'pregrasp', collision_snapshot=collision,
                    worker_request=request, scene_prevalidated=True,
                    rejection_details=validator.last_rejection), validator.last_rejection)
        validator.batch = batch
        def bind(evaluator):
            geometry.model, geometry.data = evaluator.model, evaluator.data
            geometry._state_evaluator = evaluator
        validator.bind_state_evaluator = bind
        validator.geometry = geometry
        validator.segment_validation_semantics = 'nero_observed_convex_v1'
        validator.collision_for = collision_for
        bodies = tuple({int(self.model.geom_bodyid[g]) for g in self.geometry.meshes})
        engine = MuJoCoGlobalPlanningEngine(self.model, joint_names=self.geometry.names,
            ee_site_name='ee_site', ik_config=self.ik.config,
            robot_collision_body_ids=bodies, state_validator=validator,
            candidate_path_backend=self._make_candidate_path_backend(),
            warm_roadmap_backend=self._make_warm_roadmap_backend(),
            rank_goal_routes=self.moving_approach_mode == 'high',
            shared_state_evaluation=getattr(self.settings, 'shared_state_evaluation', True),
            minimal_state_evaluation=getattr(self.settings, 'minimal_state_evaluation', True),
            adaptive_segment_validation=getattr(self.settings, 'adaptive_segment_validation', True))
        if self.joint_limit_report['expanded']:
            engine.ik.lower_limits = geometry.nominal_command_lower.copy()
            engine.ik.upper_limits = geometry.nominal_command_upper.copy()
            engine.ik.center_positions = .5*(engine.ik.lower_limits+engine.ik.upper_limits)
        worker = _MovingPlanningWorker(engine, self._request_collision_snapshot, snapshots, lock,
            on_effective_request=self._record_effective_planning_request,
            on_planning_result=self._record_planning_result)
        self._workers.append(worker)
        return worker

    def planning_report(self):
        source = getattr(self.snapshot_provider, '__self__', None)
        diagnostics = getattr(source, 'diagnostics', None)
        return {'mode': 'visually_tracked_moving_belt',
                'moving_approach_mode': self.moving_approach_mode,
                'planning_acceleration': self.planning_acceleration_report(),
                'pregrasp_execution_timing': dict(self.policy._pregrasp_execution_timing),
                'timed_descent_events': list(self.policy._descent_trace),
                'full_preplanned': self.policy._full_pick_info(),
                'full_plan_scene_monitor': (None if not hasattr(self, 'full_plan_scene_monitor')
                    else self.full_plan_scene_monitor.snapshot()),
                'continuous_intercept': self.policy._continuous_intercept_info(),
                'candidate_plan_handoff': self._candidate_plan_handoff,
                'control_source': 'oct09_visual_acquisition_shared',
                'network_grasp_source': self._network_grasp_source,
                'visual_velocity_source_diagnostics': diagnostics() if callable(diagnostics) else None,
                'velocity_measurement': self._target.get('prediction', {}).get('velocity_measurement'),
                'prediction_velocity_m_s': (None if self._motion_snapshot is None
                    or self._motion_snapshot.velocity_source == 'unavailable' else self._motion_velocity().tolist()),
                'prediction_velocity_model': self._target.get('prediction', {}).get('prediction_velocity_model'),
                'height_model_normal_base': None if self._measured_plane_normal is None else self._measured_plane_normal.tolist(),
                'acquisition_control': self.policy.acquisition_diagnostics(),
                'velocity_source': None if self._motion_snapshot is None else str(self._motion_snapshot.velocity_source),
                'measurement_is_visual': None if self._motion_snapshot is None else bool(getattr(self._motion_snapshot, 'measured', True)),
                'prediction_confidence': None if self._motion_snapshot is None else float(getattr(self._motion_snapshot, 'confidence', 0.)),
                'prediction_uncertainty_m': None if self._motion_snapshot is None else np.asarray(getattr(self._motion_snapshot, 'uncertainty', np.zeros(3))).tolist(),
                'intercept_prediction_uncertainty_m': (None
                    if self._target.get('prediction', {}).get('intercept_prediction_uncertainty') is None
                    else self._target['prediction']['intercept_prediction_uncertainty'].tolist()),
                'intercept_uncertainty_reason': self._target.get('prediction', {}).get('intercept_uncertainty_reason'),
                'nominal_speed_m_s': self.nominal_speed_m_s,
                'nominal_speed_is_encoder_feedback': False,
                'belt_direction_base': self.belt_direction.tolist(),
                'belt_origin_base_m': self.belt_origin.tolist(),
                's_range_m': list(self.s_range), 'tracking_s_range_m': list(self.tracking_s_range),
                'intercept_updates': list(self._intercept_updates),
                'planner_seeds': [request['planner_seed'] for request in self._planner_requests
                                  if request['planner_seed'] is not None],
                'planner_requests': list(self._planner_requests),
                'planner_results': list(self._planner_results),
                'last_dynamic_collision_rejection': self._collision_rejection,
                'capture_tcp_pose': None if self._capture_pose is None else self._capture_pose.tolist(),
                'capture_policy_timestamp': self._capture_time,
                'release_policy_timestamp': self._release_time,
                'release_pose_proxy': 'measured_wider_jaw_and_motor_force_drop',
                'release_handoff': self._release_clearance,
                'put_back_mode': 'fixed_world_position_at_measured_capture',
                'drop_tcp_pose': None if self._capture_pose is None else self.grasp_pose.tolist(),
                'retreat_tcp_pose': self.wait_pose.tolist(),
                'grasp_confirmation': 'fresh_motor_width_force_proxy_debounced_0.4_s',
                'target_finger_contact': 'allowed_for_selected_workpiece_only; no visual pose gate',
                'collision_coverage': 'native robot self collision and captured static scene; '
                    'worker frozen capture target geometry; '
                    'live predicted short-step target sweep for non-finger robot geometry; '
                    'carried visible target surface',
                'limitation': 'No measured encoder; prediction validity belongs to visual tracker. '
                    'The global worker does not certify an absolute-time dynamic swept volume; '
                    'visible surface is not a complete solid payload model.'}


def build_moving_policy_context(geometry, detection, initial_state, settings, **kwargs):
    return NeroMovingPolicyContext(geometry, detection, initial_state, settings, **kwargs)
