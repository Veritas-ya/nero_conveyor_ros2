"""Bind the migrated Nero7 tracker to one task without opening devices.

All host timestamps supplied here must already be mapped to one monotonic
clock. Device timestamps are retained only for Nero7 frame correspondence.
"""
from threading import RLock
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from time import monotonic
import numpy as np

from mujoco_conveyor_sorting.runtime.nero_moving_perception import MovingTargetTracker
from mujoco_conveyor_sorting.runtime.nero_graspgenx import translate_bound_network_grasp
from mujoco_conveyor_sorting.runtime.types import TargetIdentity


@dataclass(frozen=True)
class InstalledVisualScene:
    """Evidence tying an installed point-cloud scene to one accepted frame.

    This is an in-process guard, not a full safety or collision certificate.
    """
    identity: TargetIdentity
    capture_time: float
    sequence: int
    clock_epoch: int
    geometry: object
    scene_digest: bytes
    target_digest: bytes
    grasp_digest: bytes
    report: object
    install_wall_s: float
    capture_detection: object
    reference_frame: object


@dataclass(frozen=True)
class PreparedVisualScene:
    """Filtered same-frame scene ready while network inference is running."""
    identity: TargetIdentity
    capture_time: float
    sequence: int
    clock_epoch: int
    geometry: object
    scene_digest: bytes
    target_digest: bytes
    report: object
    install_wall_s: float
    detection: object
    reference_frame: object


def _array_digest(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.digest()


class MeasuredMotionSession:
    """Publish only measurements belonging to one confirmed task and epoch.

    The selected grasp remains relative to the workpiece as subsequent depth
    detections move it. The original tracker owns association and prediction.
    Cancelling permanently invalidates this instance, including old callbacks.
    """
    def __init__(self, identity, **tracker_options):
        if not isinstance(identity, TargetIdentity) or identity.clock_epoch is None:
            raise ValueError('confirmed task identity and clock epoch are required')
        if identity.track_id is None or identity.track_generation is None:
            raise ValueError('Nero7 track id and generation are required')
        options = dict(tracker_options)
        if options.pop('external_velocity_only', True) is not True:
            raise ValueError('only the measured Nero7 velocity source is supported')
        if options.pop('nominal_speed_m_s', None) is not None:
            raise ValueError('nominal belt speed is not a measurement')
        self.identity = identity
        self.tracker_options = dict(options)
        self.tracker = MovingTargetTracker(nominal_speed_m_s=None,
                                            external_velocity_only=True, **options)
        self.tracker.target_identity = identity
        self._lock = RLock()
        self._cancelled = False
        self._latest = None
        self._selected = None
        self._captures = OrderedDict()
        self._pinned_capture = None

    @staticmethod
    def _cloud_digest(detection):
        cloud = np.asarray(detection["object_points"], dtype=np.float64)
        if cloud.ndim != 2 or cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError("invalid measured target cloud")
        return hashlib.sha256(np.ascontiguousarray(cloud).tobytes()).digest()

    def _active(self, identity):
        if self._cancelled or identity != self.identity:
            raise ValueError('inactive or replaced task')

    def update(self, detection, *, identity, timestamp, sequence, clock_epoch,
               source_timestamp, velocity_measurement):
        with self._lock:
            self._active(identity)
            if clock_epoch != identity.clock_epoch:
                self.cancel()
                raise ValueError('camera clock epoch changed; start a new task')
            report = velocity_measurement
            if report is not None:
                if (report.get('track_id') != identity.track_id or
                    report.get('track_generation') != identity.track_generation):
                    raise ValueError('Nero7 track identity changed')
            candidate = dict(detection)
            candidate["object_points"] = np.asarray(detection["object_points"], dtype=float).copy()
            candidate["grasp"] = np.asarray(detection["grasp"], dtype=float).copy()
            if self._selected is not None:
                candidate = translate_bound_network_grasp(self._selected, candidate)
            if candidate.get('target_identity') != identity:
                raise ValueError('detection identity does not match task')
            accepted = self.tracker.update(candidate, timestamp=timestamp, sequence=sequence,
                clock_epoch=clock_epoch, source_timestamp=source_timestamp,
                velocity_measurement=report)
            if accepted:
                self._latest = (candidate, float(timestamp), sequence, clock_epoch)
                self._captures[(float(timestamp), sequence)] = self._cloud_digest(candidate)
                while len(self._captures) > 64:
                    self._captures.popitem(last=False)
            return accepted

    def pin_network_capture(self, *, identity, timestamp, sequence):
        """Retain one accepted exposure while its asynchronous NN job runs.

        Only the immutable digest and key are retained; camera arrays remain
        owned by the inference request. Cancel clears the pin. A different
        pending request cannot silently replace it.
        """
        with self._lock:
            self._active(identity)
            key = (float(timestamp), sequence)
            if key not in self._captures:
                raise ValueError('network exposure is not an accepted capture')
            if self._pinned_capture is not None and self._pinned_capture[0] != key:
                raise ValueError('another network exposure is already pending')
            self._pinned_capture = (key, self._captures[key])

    def release_network_capture(self, *, identity, timestamp, sequence):
        with self._lock:
            self._active(identity)
            if self._pinned_capture is not None and self._pinned_capture[0] == (float(timestamp), sequence):
                self._pinned_capture = None

    def bind_selected_grasp(self, selected, *, identity, capture_timestamp, capture_sequence):
        """Install a network pose using the last observed cloud and original offset.

        The result must still refer to the same task and to a capture no later
        than the latest accepted frame. Capture provenance must be supplied by
        the owner of the asynchronous inference request.
        """
        with self._lock:
            self._active(identity)
            if self._selected is not None:
                raise ValueError('network grasp already bound to this task')
            if selected.get('target_identity') != identity or self._latest is None:
                raise ValueError('network result does not belong to the active observation')
            latest, timestamp, sequence, epoch = self._latest
            key = (float(capture_timestamp), capture_sequence)
            expected = self._captures.get(key)
            if expected is None and self._pinned_capture is not None and self._pinned_capture[0] == key:
                expected = self._pinned_capture[1]
            if expected is None or expected != self._cloud_digest(selected):
                raise ValueError('network result does not match an accepted capture')
            if not selected.get('network_candidate') or 'network_capture_grasp_base' not in selected:
                raise ValueError('a selected network grasp with capture provenance is required')
            translated = translate_bound_network_grasp(selected, latest)
            if not self.tracker.replace_geometry(translated, timestamp=timestamp,
                                                  sequence=sequence, clock_epoch=epoch):
                raise RuntimeError('observation changed during network grasp binding')
            self._pinned_capture = None
            self._selected = {key: deepcopy(selected[key]) for key in (
                'width', 'network_candidate', 'network_provenance',
                'network_capture_center_base_m', 'network_capture_grasp_base')}
            self._latest = (translated, timestamp, sequence, epoch)
            return translated

    def snapshot(self, now, *, robot_arrival_time=0.):
        with self._lock:
            if self._cancelled:
                return None
            return self.tracker.snapshot(now, robot_arrival_time=robot_arrival_time)

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self.tracker.reset()
            self._captures.clear()
            self._pinned_capture = None
            self._latest = None
            self._selected = None

    def attach_scene_frame(self, detection, *, identity, timestamp, sequence, clock_epoch):
        """Attach an asynchronously built full scene to the same target frame.

        This does not publish another position or refresh its age. The target
        cloud must agree with the already accepted pixels within numerical
        projection precision. A newer frame or cancellation rejects the work.
        """
        with self._lock:
            self._active(identity)
            if self._latest is None or self._latest[1:] != (float(timestamp), sequence, clock_epoch):
                raise ValueError('scene belongs to an old or different observation')
            latest = self._latest[0]
            header = detection['frame'].header
            if (detection.get('target_identity') != identity or
                    (header.sample_time, header.sequence, header.clock_epoch) != self._latest[1:]):
                raise ValueError('scene frame identity does not match observation')
            a, b = np.asarray(latest['object_points']), np.asarray(detection['object_points'])
            if a.shape != b.shape or not np.allclose(a, b, rtol=0., atol=1e-10):
                raise ValueError('scene target pixels differ from the accepted observation')
            if 'scene_points' not in detection or 'scene_state' not in detection:
                raise ValueError('full scene points and capture-time robot state are required')
            updated = dict(latest)
            for name in ('frame', 'mask', 'scene_points', 'scene_pixel_indices', 'scene_state', 'belt_region'):
                if name in detection:
                    updated[name] = detection[name]
            self._latest = (updated, *self._latest[1:])

    def install_latest_scene(self, geometry, *, identity, scene_options=None):
        """Install the original scene filter using one accepted RGB-D frame.

        The caller owns the model exclusively during installation. It supplies
        points in robot-base coordinates obtained from this frame, its matching
        scene state and calibration. Later observations can proceed concurrently;
        a current scene monitor is mandatory before an old snapshot can execute.
        """
        with self._lock:
            self._active(identity)
            if self._selected is None or self._latest is None:
                raise ValueError('bind a network grasp before installing its scene')
            detection, stamp, sequence, epoch = self._latest
            frame = detection['frame']
            header = frame.header
            if (header.sample_time != stamp or header.sequence != sequence or
                    header.clock_epoch != epoch):
                raise ValueError('scene frame does not match the accepted observation')
            transform = np.asarray(frame.camera_to_world_cv).copy()
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('calibrated camera transform is required')
            points = np.asarray(detection['scene_points'], dtype=float).copy()
            target = np.asarray(detection['object_points'], dtype=float).copy()
            state = deepcopy(detection['scene_state'])
            options = dict(scene_options or {})
            options.setdefault('gripper_nearby_scene_ignore', True)
            background = getattr(geometry, 'scene_background_reference', None)
            if background is not None:
                options.setdefault('scene_background_reference', background)
            if detection.get('scene_pixel_indices') is not None:
                options.setdefault('scene_pixel_indices', detection['scene_pixel_indices'])
            mask = detection.get('mask')
            region = detection.get('belt_region')
            capture_detection = dict(detection)
            capture_detection['object_points'] = target.copy()
            capture_detection['grasp'] = np.asarray(detection['grasp'], dtype=float).copy()
        started = monotonic()
        report = geometry.install_scene(points, target, state,
            camera_origin=transform[:3, 3].copy(), belt_region=region,
            depth_frame=frame, target_mask=mask, **options)
        elapsed = monotonic() - started
        scene_digest = _array_digest(geometry.scene)
        target_digest = _array_digest(geometry.target)
        with self._lock:
            self._active(identity)
            if self._latest is None:
                raise ValueError('observation unavailable during scene installation')
            if self._latest[3] != epoch:
                raise ValueError('camera clock changed during scene installation')
            if not np.array_equal(np.asarray(geometry.target), target[::3]):
                raise ValueError('installed target differs from the accepted frame')
            return InstalledVisualScene(identity, stamp, sequence, epoch, geometry,
                                        scene_digest, target_digest, _array_digest(capture_detection['grasp']), report, elapsed,
                                        capture_detection, frame)

    def prepare_captured_scene(self, geometry, detection, *, identity, timestamp,
                               sequence, clock_epoch, scene_options=None):
        """Filter the immutable observed frame concurrently with NN inference.

        This produces scene geometry, not a grasp or motion certificate. The
        model is exclusively owned by the caller. Final binding requires the
        selected network grasp for this same task and rechecks the scene hashes.
        """
        stamp = float(timestamp)
        with self._lock:
            self._active(identity)
            key = (stamp, sequence)
            if (clock_epoch != identity.clock_epoch or detection.get('target_identity') != identity
                    or key not in self._captures or self._captures[key] != self._cloud_digest(detection)):
                raise ValueError('scene does not match an accepted target capture')
            frame = detection['frame']; header = frame.header
            if (header.sample_time, header.sequence, header.clock_epoch) != (stamp, sequence, clock_epoch):
                raise ValueError('scene frame does not match its capture')
            transform = np.asarray(frame.camera_to_world_cv, dtype=float).copy()
            if transform.shape != (4,4) or not np.isfinite(transform).all():
                raise ValueError('calibrated camera transform is required')
            target = np.asarray(detection['object_points'], dtype=float).copy()
            scene = np.asarray(detection['scene_points'], dtype=float).copy()
            state = deepcopy(detection['scene_state'])
            frozen = dict(detection); frozen['object_points'] = target.copy()
            frozen['grasp'] = np.asarray(detection['grasp'], dtype=float).copy()
            options = dict(scene_options or {})
            options.setdefault('gripper_nearby_scene_ignore', True)
            background = getattr(geometry, 'scene_background_reference', None)
            if background is not None: options.setdefault('scene_background_reference', background)
            if detection.get('scene_pixel_indices') is not None:
                options.setdefault('scene_pixel_indices', detection['scene_pixel_indices'])
            region, mask = detection.get('belt_region'), detection.get('mask')
        started = monotonic()
        report = geometry.install_scene(scene, target, state, camera_origin=transform[:3,3],
            belt_region=region, depth_frame=frame, target_mask=mask, **options)
        elapsed = monotonic()-started
        scene_digest, target_digest = _array_digest(geometry.scene), _array_digest(geometry.target)
        with self._lock:
            self._active(identity)
            if self._latest is None or self._latest[3] != clock_epoch:
                raise ValueError('camera clock changed during scene preparation')
            if not np.array_equal(np.asarray(geometry.target), target[::3]):
                raise ValueError('installed target differs from the accepted capture')
            return PreparedVisualScene(identity, stamp, sequence, clock_epoch, geometry,
                scene_digest, target_digest, report, elapsed, frozen, frame)

    def bind_prepared_scene(self, prepared, *, identity):
        """Bind an already filtered scene to the actual selected network TCP."""
        with self._lock:
            self._active(identity)
            if self._selected is None:
                raise ValueError('network grasp has not been bound')
            if not isinstance(prepared, PreparedVisualScene) or prepared.identity != identity:
                raise ValueError('prepared scene belongs to a different task')
            geometry = prepared.geometry
            if (prepared.scene_digest != _array_digest(geometry.scene) or
                    prepared.target_digest != _array_digest(geometry.target) or
                    not np.array_equal(geometry.target, prepared.detection['object_points'][::3])):
                raise ValueError('prepared scene geometry changed')
            bound = translate_bound_network_grasp(self._selected, prepared.detection)
            bound['object_points'] = np.asarray(prepared.detection['object_points'], dtype=float).copy()
            bound['grasp'] = np.asarray(bound['grasp'], dtype=float).copy()
            return InstalledVisualScene(identity, prepared.capture_time, prepared.sequence,
                prepared.clock_epoch, geometry, prepared.scene_digest, prepared.target_digest,
                _array_digest(bound['grasp']), prepared.report, prepared.install_wall_s,
                bound, prepared.reference_frame)

    def install_captured_scene(self, geometry, detection, *, identity, timestamp,
                               sequence, clock_epoch, scene_options=None):
        """Install an accepted immutable frame while later observations continue.

        The scene builder may finish after newer observations. Its capture must
        still match the remembered measured target; the selected network grasp
        is expressed at that capture, never at the latest frame's timestamp.
        A live scene monitor is required by create_full_context before motion.
        """
        stamp = float(timestamp)
        with self._lock:
            self._active(identity)
            if self._selected is None:
                raise ValueError('bind a network grasp before installing its scene')
            key = (stamp, sequence)
            if (clock_epoch != identity.clock_epoch or detection.get('target_identity') != identity
                    or key not in self._captures or self._captures[key] != self._cloud_digest(detection)):
                raise ValueError('scene does not match an accepted target capture')
            frame = detection['frame']
            header = frame.header
            if (header.sample_time, header.sequence, header.clock_epoch) != (stamp, sequence, clock_epoch):
                raise ValueError('scene frame does not match its capture')
            transform = np.asarray(frame.camera_to_world_cv, dtype=float).copy()
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('calibrated camera transform is required')
            target = np.asarray(detection['object_points'], dtype=float).copy()
            scene = np.asarray(detection['scene_points'], dtype=float).copy()
            state = deepcopy(detection['scene_state'])
            bound = translate_bound_network_grasp(self._selected, detection)
            bound['object_points'] = target
            bound['grasp'] = np.asarray(bound['grasp'], dtype=float).copy()
            options = dict(scene_options or {})
            options.setdefault('gripper_nearby_scene_ignore', True)
            background = getattr(geometry, 'scene_background_reference', None)
            if background is not None:
                options.setdefault('scene_background_reference', background)
            if detection.get('scene_pixel_indices') is not None:
                options.setdefault('scene_pixel_indices', detection['scene_pixel_indices'])
            region, mask = detection.get('belt_region'), detection.get('mask')
        started = monotonic()
        report = geometry.install_scene(scene, target, state, camera_origin=transform[:3,3],
            belt_region=region, depth_frame=frame, target_mask=mask, **options)
        elapsed = monotonic()-started
        scene_digest, target_digest = _array_digest(geometry.scene), _array_digest(geometry.target)
        with self._lock:
            self._active(identity)
            if self._latest is None or self._latest[3] != clock_epoch:
                raise ValueError('camera clock changed during scene installation')
            if not np.array_equal(np.asarray(geometry.target), target[::3]):
                raise ValueError('installed target differs from the accepted frame')
            return InstalledVisualScene(identity, stamp, sequence, clock_epoch, geometry,
                scene_digest, target_digest, _array_digest(bound['grasp']), report, elapsed, bound, frame)

    def create_full_context(self, geometry, initial_state, settings, *, identity,
                            scene_binding, policy_clock_origin, belt_direction, belt_origin,
                            s_range, tracking_s_range=None, scene_monitor_options=None):
        """Construct the original full planner context for the bound grasp.

        The frozen target and scene belong to the same capture. If perception
        has advanced, require the original scene monitor before returning.
        No execution or hardware access starts.
        """
        from mujoco_conveyor_sorting.runtime.nero_moving_context import build_moving_policy_context
        if scene_monitor_options is None:
            raise ValueError('a live scene monitor is required for full planning')
        with self._lock:
            self._active(identity)
            if self._selected is None or self._latest is None:
                raise ValueError('bind a network grasp before creating the context')
            if settings.planning_mode != 'full_preplanned' or settings.moving_approach_mode != 'continuous':
                raise ValueError('full preplanned continuous approach is required')
            latest, latest_stamp, latest_sequence, latest_epoch = self._latest
            if (not isinstance(scene_binding, InstalledVisualScene) or
                    scene_binding.geometry is not geometry or scene_binding.identity != identity or
                    scene_binding.clock_epoch != latest_epoch or
                    scene_binding.scene_digest != _array_digest(geometry.scene) or
                    scene_binding.target_digest != _array_digest(geometry.target)):
                raise ValueError('installed scene is stale or does not match this observation')
            detection = scene_binding.capture_detection
            if scene_binding.grasp_digest != _array_digest(detection['grasp']):
                raise ValueError('captured network grasp changed after scene installation')
            if not np.array_equal(np.asarray(geometry.target), detection['object_points'][::3]):
                raise ValueError('installed target geometry does not match its capture')
            snapshot = self.tracker.snapshot(max(float(policy_clock_origin), latest_stamp))
            if (snapshot is None or not snapshot.valid or
                snapshot.velocity_source != 'nero7_xy_kalman' or
                snapshot.target_identity != identity):
                raise ValueError('a valid matching measured Nero7 snapshot is required')
        context = build_moving_policy_context(geometry, detection, initial_state, settings,
            snapshot_provider=self.snapshot, belt_direction=belt_direction,
            belt_origin=belt_origin, s_range=s_range, tracking_s_range=tracking_s_range,
            moving_approach_mode='continuous')
        try:
            context.set_policy_clock_origin(policy_clock_origin)
            with self._lock:
                self._active(identity)
            if scene_monitor_options is not None:
                from mujoco_conveyor_sorting.runtime.full_plan_scene_monitor import FullPlanSceneMonitor
                options = dict(scene_monitor_options)
                if 'reference_frame' in options:
                    raise ValueError('scene monitor reference must be the installed scene frame')
                context.full_plan_scene_monitor = FullPlanSceneMonitor(context,
                    reference_frame=scene_binding.reference_frame, **options)
                context.full_plan_scene_monitor.start()
            with self._lock:
                self._active(identity)
                if self._latest is None or self._latest[3] != scene_binding.clock_epoch:
                    raise ValueError('observation or clock changed during planner preparation')
                if self._latest[1:] != (scene_binding.capture_time, scene_binding.sequence, scene_binding.clock_epoch):
                    if scene_monitor_options is None:
                        raise ValueError('scene monitor required after visual frame advances')
                    from mujoco_conveyor_sorting.runtime.full_plan_scene_monitor import FullPlanSceneMonitor
                    current_frame = self._latest[0].get('frame')
                    if (current_frame is None or
                            (current_frame.header.sample_time, current_frame.header.sequence,
                             current_frame.header.clock_epoch) != self._latest[1:] or
                            FullPlanSceneMonitor._frame_binding(current_frame) !=
                            FullPlanSceneMonitor._frame_binding(scene_binding.reference_frame)):
                        raise ValueError('camera binding changed since scene capture')
                if (scene_binding.scene_digest != _array_digest(geometry.scene) or
                        scene_binding.target_digest != _array_digest(geometry.target)):
                    raise ValueError('installed scene changed during planner preparation')
        except Exception:
            context.close()
            raise
        return context
