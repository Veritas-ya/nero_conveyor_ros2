"""Measured scene/target planning; MuJoCo is a private kinematic mirror only."""
from dataclasses import dataclass, field, replace
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from mujoco_conveyor_sorting.calibration.geometry import inverse
from mujoco_conveyor_sorting.controllers.safe_path_planner import ConstrainedRRTConnect, SafePathConfig
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, PickRejected
from .reference import (Intercept, predict_entry_time, DEFAULT_FREEZE_UPSTREAM_DISTANCE_M,
    APPROACH_DURATION_S)
from .trajectory import certify_curve, sampling_times, retime_samples, merge_collinear_waypoints, geometry_check_times
from .motion_profile import select_motion_profile
from .belt_collision_scene import BELT_SURFACE_ONLY, possible_belt_neighbors
from .feedback_gates import TRAJECTORY_ARRIVAL_TOLERANCE_RAD
from .workpiece_contact import recovery_contact_allowed


def _support_hull_separated(hull_vertices, rotation, position, origin, normal):
    """Conservative proof that the complete mesh is above the support plane."""
    height = np.min(hull_vertices @ (rotation.T @ normal)) + (position-origin) @ normal
    return bool(height > -1e-6 + 1e-9)


def calibrated_model(config):
    model = RobotModel()
    # Change only the calibrated task site; collision meshes stay physical
    # model assets. Resolve flange->site through the actual site parent body.
    model.set_state(config.standby, config.settings.open_width_m)
    parent = int(model.model.site_bodyid[model.tcp_id])
    parent_pose = np.eye(4)
    parent_pose[:3, :3] = model.data.xmat[parent].reshape(3, 3)
    parent_pose[:3, 3] = model.data.xpos[parent]
    local = inverse(parent_pose) @ model.flange() @ config.tcp
    model.model.site_pos[model.tcp_id] = local[:3, 3]
    model.model.site_quat[model.tcp_id] = Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]]
    # The compiler may have marked the original site orientation as identical
    # to its parent. A measured rotation invalidates that optimization.
    model.model.site_sameframe[model.tcp_id] = 0
    model.set_state(config.standby, config.settings.open_width_m)
    return model


@dataclass(frozen=True)
class InterceptPlan:
    path: object
    reference: Intercept
    target: object
    scene_revision: int
    selected_grasp: np.ndarray
    failures: tuple
    timings_s: dict = field(default_factory=dict)


class MeasuredPlanner:
    def __init__(self, config, model=None, *, clock=time.monotonic):
        self.config = config
        self.model = calibrated_model(config) if model is None else model
        self.clock = clock
        self.version = 0
        self.identity = None
        self.target = None
        self.payload_local = None
        self.scene_revision = 0
        self.support = None
        self.support_origin = None
        self.contact_reference = None
        self.target_tree = None
        self.payload_tree = None
        self.released_points = None
        self.released_tree = None
        self.stats = {'states': 0, 'payload_states': 0}
        self.progress = {'stage': 'idle', 'since_monotonic_s': self.clock()}
        self.attempt_diagnostics = []
        self.collision_sink = None
        self.collision_record_count = 0
        self.collision_diagnostic_errors = []
        self.geometry_context = {}
        self.scene_report = None
        self._support_hulls = {}

    def _progress(self, stage, **detail):
        # Atomic replacement of small immutable records; no I/O on the worker.
        self.progress = {'stage': stage, 'since_monotonic_s': self.clock(), **detail}

    def diagnostics(self):
        item = self.progress
        return {**item, 'stage_elapsed_s': max(0., self.clock()-item['since_monotonic_s']),
                'completed_attempts': list(self.attempt_diagnostics),
                'collision_scene': self.scene_report}

    def install(self, target, state):
        """Install this exposure once. Existing native scene filters are reused."""
        self.model.check_feedback(state)
        ob = target.detection
        from .belt_collision_scene import OBSERVED_SCENE, BELT_SURFACE_ONLY, install_belt_surface
        scope = self.config.raw.get('environment_collision_scope', OBSERVED_SCENE)
        if scope == BELT_SURFACE_ONLY:
            self.scene_report = install_belt_surface(self.model, target)
        elif scope == OBSERVED_SCENE:
            self.model.install_scene(ob['scene_points'], target.points, target.state,
                camera_origin=ob.get('camera_origin_base_m'), belt_region=ob.get('belt_region'),
                depth_frame=ob.get('frame'), target_mask=ob.get('mask'),
                scene_pixel_indices=ob.get('scene_pixel_indices'))
            self.scene_report = {'scope':scope, 'collision_points':len(self.model.scene)}
        else:
            raise PickRejected('未知环境碰撞范围；拒绝安装空场景')
        self.model.target = np.empty((0, 3))  # moving target checked separately
        self.identity = target.identity
        self.target = target
        self.target_tree = cKDTree(target.points)
        self.payload_local = None
        self.released_points = self.released_tree = None
        self.scene_revision += 1
        self.version += 1
        plane = np.asarray(ob['belt_region']['plane_to_base'])
        self.plane = plane.copy()
        self.belt_bounds = np.asarray(ob['belt_region']['bounds_xy_m'])
        self.support, self.support_origin = plane[:3, 2].copy(), plane[:3, 3].copy()

    def target_points_at(self, stamp):
        return self.target.points + self.target.velocity * (float(stamp) - self.target.capture_s)

    def valid(self, q, width, *, stamp=None, contact=False, payload=False, allow_belt_support=False):
        self.stats['states'] += 1
        self.model.check_state(q, width, allow_finger_contact=True)
        tcp = self.model.tcp()
        xyz = tcp[:3, 3]
        if np.any(xyz < self.config.workspace[:, 0]) or np.any(xyz > self.config.workspace[:, 1]):
            raise PickRejected('TCP 超出现场工作区', details={'kind':'tcp_workspace',
                'tcp_base_m':xyz.tolist(), 'workspace_base_m':self.config.workspace.tolist()})
        # Support height is checked on actual hull vertices rather than TCP.
        for geom in self.model.gripper_geometries:
            vertices = self.model.mesh_vertices[geom]
            rotation = self.model.data.geom_xmat[geom].reshape(3, 3)
            position = self.model.data.geom_xpos[geom]
            if geom not in self._support_hulls:
                # A linear height function attains its minimum on a convex
                # hull vertex. Only prove separation here; near the boundary
                # retain the original exact full-mesh/XY evaluation below.
                from scipy.spatial import ConvexHull
                self._support_hulls[geom] = vertices[ConvexHull(vertices).vertices].copy()
            normal = self.plane[:3, 2]
            if _support_hull_separated(self._support_hulls[geom], rotation, position, self.support_origin, normal):
                continue
            world = vertices @ rotation.T + position
            local = (world-self.support_origin) @ self.plane[:3, :3]
            over_belt = np.all((local[:, :2] >= self.belt_bounds[:, 0]) & (local[:, :2] <= self.belt_bounds[:, 1]), axis=1)
            if np.any(over_belt & (local[:, 2] < -1e-6)):
                below = np.flatnonzero(over_belt & (local[:, 2] < -1e-6))
                raise PickRejected('机器人几何低于实测带面', details={
                    'kind':'belt_plane_penetration', 'geometry':self.model.model.geom(geom).name,
                    'minimum_signed_height_m':float(np.min(local[below,2])),
                    'penetrating_vertex_count':int(len(below)),
                    'example_vertices_base_m':world[below[:8]].tolist()})
        # A cancelled lightweight part may already intersect a finger. Allow
        # retreat from it without rejecting the very first recovery sample.
        # This scope cannot affect intercepts, another target, native scene
        # geometry, or the payload-to-environment checks below.
        workpiece_margin = float(self.config.raw.get('workpiece_collision_margin_m', .004))
        allow_workpiece = recovery_contact_allowed(self)
        if allow_workpiece:
            self.stats['recovery_workpiece_checks_skipped'] = self.stats.get('recovery_workpiece_checks_skipped', 0)+1
        released = not payload and stamp is None and self.released_points is not None
        delta = None
        if payload:
            points = self.payload_local @ tcp[:3, :3].T + xyz
        elif stamp is not None:
            # The immutable target tree is in exposure coordinates. Translate
            # only broad-phase hits; materialize the full predicted cloud on a
            # rejection so diagnostic packets retain their original contents.
            points = None
            delta = self.target.velocity * (float(stamp)-self.target.capture_s)
        else:
            points = self.released_points if released else np.empty((0, 3))
        if not len(self.target.points if points is None else points):
            return
        if points is None:
            # Scope: moving target during approach/follow only. Payload and
            # released clouds and all scene/self/belt checks remain unchanged.
            if allow_workpiece or not self.config.raw.get('grasp_workpiece_collision_check_enabled', True):
                self.stats['moving_workpiece_checks_skipped'] = self.stats.get('moving_workpiece_checks_skipped', 0) + 1
                return
            # The exposure tree is shared; hardware supplies its own margin.
            from .moving_target_collision import first_moving_target_hit
            hit=first_moving_target_hit(self.model,self.target.points,self.target_tree,delta,contact=contact,margin_m=workpiece_margin)
            if hit is not None:
                geom,hit_ids=hit
                detail={'geometry':self.model.model.geom(geom).name,
                        'collision_margin_m':workpiece_margin,'hit_count':len(hit_ids)}
                try:
                    from .collision_capture import capture_collision
                    detail=capture_collision(self,q,width,self.target_points_at(stamp),hit_ids,geom,
                        stamp=stamp,contact=contact,payload=payload,released=released,
                        margin_m=workpiece_margin)
                except Exception as error:
                    message=f'{type(error).__name__}: {error}'
                    self.collision_diagnostic_errors.append(message)
                    detail['diagnostic_error']=message
                raise PickRejected('机器人与实测工件点云碰撞',details=detail)
            return
        for geom in (() if allow_workpiece else self.model.meshes):
            if geom in self.model.fingers and (contact or payload or released):
                continue
            center = self.model.data.geom_xpos[geom]
            if payload:
                local_center = (center-xyz) @ tcp[:3, :3]
                ids = self.payload_tree.query_ball_point(local_center, self.model.collision_radii[geom]+1e-9)
            elif not released:
                ids = self.target_tree.query_ball_point(center-delta, self.model.collision_radii[geom]+1e-9)
            else:
                ids = self.released_tree.query_ball_point(center, self.model.collision_radii[geom]+1e-9)
            if ids:
                nearby = self.target.points[ids]+delta if points is None else points[ids]
                inside = self.model.inside_geom(nearby, geom, workpiece_margin)
                if inside.any():
                    hit_ids = np.asarray(ids, dtype=np.int64)[inside]
                    detail = {'geometry': self.model.model.geom(geom).name,
                              'collision_margin_m': workpiece_margin, 'hit_count': len(hit_ids)}
                    try:
                        from .collision_capture import capture_collision
                        all_points = self.target_points_at(stamp) if points is None else points
                        detail = capture_collision(self, q, width, all_points, hit_ids, geom,
                            stamp=stamp, contact=contact, payload=payload, released=released,
                            margin_m=workpiece_margin)
                    except Exception as error:
                        # Diagnostics must never hide or relax this rejection.
                        message = f'{type(error).__name__}: {error}'
                        self.collision_diagnostic_errors.append(message)
                        detail['diagnostic_error'] = message
                    raise PickRejected('机器人与实测工件点云碰撞', details=detail)
        if payload:
            self.stats['payload_states'] += 1
            if len(self.model.scene):
                query_points = points
                belt_only = self.scene_report is not None and self.scene_report['scope'] == BELT_SURFACE_ONLY
                clearance = float(self.config.raw.get('payload_belt_clearance_m', .008)) if belt_only else .008
                if belt_only:
                    # Every installed belt point lies within the fixed support
                    # band. Points farther than that band plus the unchanged
                    # clearance cannot be neighbours, regardless of XY.
                    possible = possible_belt_neighbors(points, self.support_origin, self.support, clearance)
                    query_points = points[possible]
                distance, indices = self.model.tree.query(query_points)
                close = distance < clearance
                if close.any():
                    near_support = np.abs((self.model.scene[indices[close]]-self.support_origin) @ self.support) <= .005
                    near_capture = (np.linalg.norm((xyz-self.capture_tcp[:3, 3])[:2]) <= .010 and
                                    abs(xyz[2]-self.capture_tcp[2, 3]) <= .015)
                    support_allowed = near_capture
                    if allow_belt_support:
                        local = (self.model.scene[indices[close]]-self.support_origin) @ self.plane[:3, :3]
                        over_belt = np.all((local[:, :2] >= self.belt_bounds[:, 0]) &
                                           (local[:, :2] <= self.belt_bounds[:, 1]), axis=1)
                        support_allowed = bool(over_belt.all() and
                            abs((xyz-self.capture_tcp[:3, 3])@self.support) <= .015)
                    if not support_allowed or not near_support.all():
                        bad_indices = np.flatnonzero(close)
                        raise PickRejected('载物点云与实测场景碰撞', details={
                            'kind':'payload_scene_collision', 'margin_m':clearance,
                            'hit_count':int(len(bad_indices)),
                            'minimum_distance_m':float(np.min(distance[bad_indices])),
                            'payload_points_base_m':query_points[bad_indices[:8]].tolist(),
                            'nearest_scene_points_base_m':self.model.scene[indices[bad_indices[:8]]].tolist(),
                            'support_allowed':bool(support_allowed),
                            'near_support_all':bool(near_support.all())})

    def check_path(self, path, width, *, origin=None, contact_time=None, reference=None,
                   payload=False, cancelled=lambda: False):
        self.geometry_context = {'trajectory_stage': path.stage,
            'origin_monotonic_s': origin, 'contact_monotonic_s': contact_time,
            'trajectory_version': path.version,
            **(getattr(self, 'candidate_context', {}) if 'intercept' in path.stage else {})}
        self._progress('sampled_geometry', trajectory_stage=path.stage,
            trajectory_duration_s=path.duration)
        previous = path.position(0)
        previous_time = 0.
        hz = self.config.raw.get('geometry_check_hz', 2.*self.config.settings.control_hz)
        phases = () if reference is None else tuple(
            value for value in (getattr(reference,'rotate_first_s',0.),getattr(reference,'prealign_s',0.), getattr(reference,'duration_s',path.duration),
                                getattr(reference,'follow_end_s',path.duration))
            if 0. <= value <= path.duration)
        samples = geometry_check_times(path.duration,hz,*phases)
        max_position_error = 0.
        max_orientation_error = 0.
        for sample_index, t in enumerate(samples):
            self._progress('sampled_geometry', trajectory_stage=path.stage,
                trajectory_duration_s=path.duration, sample_index=sample_index,
                sample_count=len(samples), trajectory_time_s=float(t))
            if cancelled():
                raise PickRejected('规划已取消')
            q = path.position(t)
            count = max(1, int(np.ceil(np.max(np.abs(q-previous))/.008)))
            # The preceding segment already checked this start at the same
            # q, width, time and contact/payload phase. Keep every endpoint
            # and every extra .008-rad subdivision, without checking it twice.
            for u in np.linspace(0., 1., count+1)[1:]:
                if cancelled():
                    raise PickRejected('规划已取消')
                stamp = None if origin is None else origin+previous_time+(t-previous_time)*u
                self.geometry_context = {**self.geometry_context,
                    'sample_index': sample_index, 'sample_count': len(samples),
                    'sample_elapsed_s': float(previous_time+(t-previous_time)*u),
                    'sample_monotonic_s': stamp}
                brake = reference is not None and hasattr(reference, 'follow_end_s') and (
                    previous_time+(t-previous_time)*u >= reference.follow_end_s)
                self.valid(previous+(q-previous)*u, width,
                    stamp=None if brake else stamp,
                    contact=contact_time is not None and stamp >= contact_time-1e-8,
                    payload=payload or brake, allow_belt_support=brake)
            if reference is not None:
                actual = self.model.tcp()
                xyz, quat = reference.sample(t)
                angle = Rotation.from_matrix(Rotation.from_quat(quat[[1,2,3,0]]).as_matrix().T@actual[:3,:3]).magnitude()
                position_error = float(np.linalg.norm(actual[:3,3]-xyz))
                max_position_error = max(max_position_error, position_error)
                max_orientation_error = max(max_orientation_error, float(angle))
                if (position_error > self.model.ik.config.position_tolerance or
                    angle > self.model.ik.config.orientation_tolerance):
                    raise PickRejected('关节曲线偏离指定 TCP 参考/方向', details={'elapsed_s':float(t),
                        'position_error_m': position_error, 'orientation_error_rad': float(angle)})
            previous = q
            previous_time = t
        if reference is not None:
            return {'samples':len(samples), 'configured_frequency_hz':hz, 'max_position_error_m': max_position_error,
                    'max_orientation_error_rad': max_orientation_error,
                    'position_tolerance_m': self.model.ik.config.position_tolerance,
                    'orientation_tolerance_rad': self.model.ik.config.orientation_tolerance}

    def intercept(self, target, state, *, cancelled=lambda: False):
        if self.config.raw.get('trajectory_profile') == 'smooth':
            from .smooth_planning import plan_smooth_intercept
            if self.config.raw.get('planning_reserve_mode', 'fixed') == 'measured':
                from .adaptive_planning import plan_with_measured_timing
                return plan_with_measured_timing(self, target, state, cancelled=cancelled,
                                                 planning_fn=plan_smooth_intercept)
            return plan_smooth_intercept(self, target, state, cancelled=cancelled)
        begun = self.clock()
        self.attempt_diagnostics = []
        self._progress('entry_selection', identity=target.identity)
        direction = self.config.direction
        entry_position = getattr(target, 'entry_position', target.center)
        entry = predict_entry_time(target.capture_s, entry_position @ direction,
            target.velocity @ direction, self.config.follow_bounds[0], begun,
            self.config.raw['planning_reserve_s'], maximum_upstream_distance_m=
            self.config.raw.get('freeze_upstream_distance_m', DEFAULT_FREEZE_UPSTREAM_DISTANCE_M))
        self._progress('initial_feedback_check', identity=target.identity)
        self.model.check_feedback(state)
        timings = {'initial_feedback_check': self.clock()-begun, 'scene_prepare': 0.}
        scene_prepared = False
        start = self.model.tcp().copy()
        deadline = min(begun+self.config.raw['planning_budget_s'],
                       entry-self.config.raw['command_advance_s'])
        def abort():
            return cancelled() or self.clock() >= deadline
        follow = self.config.settings.full_plan_close_window_s
        failures = []
        for candidate_index, grasp in enumerate(sorted(target.grasps[:2], key=lambda g: Rotation.from_matrix(start[:3, :3].T@g[:3, :3]).magnitude())):
            if abort():
                raise PickRejected('抓取候选规划预算耗尽', details={'failures': failures})
            ref = Intercept(target.capture_s, entry, APPROACH_DURATION_S, grasp[:3, 3], target.velocity,
                start[:3, 3], Rotation.from_matrix(start[:3, :3]).as_quat()[[3, 0, 1, 2]],
                Rotation.from_matrix(grasp[:3, :3]).as_quat()[[3, 0, 1, 2]])
            try:
                preparing_scene = False
                attempt_started = self.clock()
                self._progress('candidate_bounds', candidate_index=candidate_index)
                begin, end = ref.contact_xyz @ direction, (ref.contact_xyz+target.velocity*follow) @ direction
                if min(begin, end) < self.config.follow_bounds[0] or max(begin, end) > self.config.follow_bounds[1]:
                    raise PickRejected('接触及闭合跟随段超出现场抓取区间')
                follow_end = ref.contact_xyz+target.velocity*follow
                endpoints = np.stack((ref.contact_xyz, follow_end))
                if (np.any(endpoints < self.config.workspace[:, 0]) or
                        np.any(endpoints > self.config.workspace[:, 1])):
                    raise PickRejected('预测接触点或闭合跟随终点超出工作区', details={
                        'kind': 'predicted_contact_outside_workspace',
                        'contact_tcp_base_m': ref.contact_xyz.tolist(),
                        'follow_end_tcp_base_m': follow_end.tolist(),
                        'workspace_base_m': self.config.workspace.tolist(),
                        'prediction_horizon_s': ref.contact_s-target.capture_s,
                        'velocity_base_m_s': target.velocity.tolist()})
                dt = 1./self.config.settings.control_hz
                times = sampling_times(2.+follow, dt, 2.)
                joints = [np.asarray(state['joints']).copy()]
                for waypoint_index, t in enumerate(times[1:], start=1):
                    if abort():
                        raise PickRejected('抓取规划已取消或超出预算')
                    xyz, quat = ref.sample(t)
                    self._progress('intercept_ik', candidate_index=candidate_index,
                        waypoint_index=waypoint_index, waypoint_count=len(times)-1,
                        trajectory_time_s=float(t), candidate_wall_s=self.clock()-attempt_started,
                        remaining_budget_s=max(0., deadline-self.clock()))
                    solved = self.model.solve_pose(xyz, quat, joints[-1], cancelled=abort)
                    if not solved.success:
                        returned = getattr(solved, 'joint_positions', None)
                        residual = {'elapsed_s': float(t), 'candidate_index': candidate_index,
                            'target_tcp_base_m': np.asarray(xyz).tolist(),
                            'target_quaternion_wxyz': np.asarray(quat).tolist(),
                            'seed_joints_rad': joints[-1].tolist(),
                            'joint_lower_rad': self.model.command_lower.tolist(),
                            'joint_upper_rad': self.model.command_upper.tolist(),
                            'position_tolerance_m': self.model.ik.config.position_tolerance,
                            'orientation_tolerance_rad': self.model.ik.config.orientation_tolerance,
                            'candidate_wall_s': self.clock()-attempt_started}
                        for name in ('position_error', 'orientation_error', 'iterations', 'message'):
                            value = getattr(solved, name, None)
                            if value is not None:
                                residual[name] = value.item() if isinstance(value, np.generic) else value
                        if returned is not None:
                            returned = np.asarray(returned)
                            residual['returned_joints_rad'] = returned.tolist()
                            residual['returned_joint_margin_rad'] = np.minimum(
                                returned-self.model.command_lower, self.model.command_upper-returned).tolist()
                        raise PickRejected('两秒直线路径逆解失败', details=residual)
                    joints.append(np.asarray(solved.joint_positions))
                timings['selected_candidate_ik'] = self.clock()-attempt_started
                check_started = self.clock()
                self._progress('polynomial_limits', candidate_index=candidate_index)
                path = certify_curve(times, joints, self.config, self.model.command_lower, self.model.command_upper,
                    identity=target.identity, version=self.version, stage='intercept_follow', planning_s=self.clock()-begun)
                timings['polynomial_limits'] = self.clock()-check_started
                # IK and polynomial limits need no scene point-cloud index.
                # Build THIS exposure's scene only once a candidate passes,
                # still before any geometry certification or execution.
                if not scene_prepared:
                    preparing_scene = True
                    self._progress('scene_prepare', identity=target.identity)
                    check_started = self.clock()
                    self.install(target, state)
                    timings['scene_prepare'] = self.clock()-check_started
                    scene_prepared = True
                    preparing_scene = False
                path = replace(path, version=self.version)
                check_started = self.clock()
                self.check_path(path, self.config.settings.open_width_m, origin=entry, contact_time=ref.contact_s,
                                reference=ref, cancelled=abort)
                timings['sampled_geometry'] = self.clock()-check_started
                if self.clock() >= entry-self.config.raw['command_advance_s']:
                    raise PickRejected('轨迹交付晚于提前发送开始时间')
                self.contact_reference = ref
                timings['full_planning_delivery'] = self.clock()-begun
                path = replace(path, planning_s=timings['full_planning_delivery'])
                self.attempt_diagnostics.append({'candidate_index': candidate_index,
                    'status': 'delivered', 'wall_s': self.clock()-attempt_started})
                self._progress('delivered', candidate_index=candidate_index, timings_s=dict(timings))
                return InterceptPlan(path, ref, target, self.scene_revision, grasp.copy(), tuple(failures), timings)
            except PickRejected as error:
                if preparing_scene:
                    raise  # A common scene failure is not a grasp alternative.
                failures.append({'reason': str(error), 'details': error.details})
                self.attempt_diagnostics.append({'candidate_index': candidate_index,
                    'status': 'rejected', 'wall_s': self.clock()-attempt_started,
                    'reason': str(error), 'details': error.details})
        self._progress('all_candidates_rejected', scene_prepared=scene_prepared,
            planning_wall_s=self.clock()-begun)
        raise PickRejected('所有 OBB 候选均未交付可执行曲线', details={'failures': failures,
            'scene_prepared': scene_prepared, 'planning_wall_s': self.clock()-begun})

    def check_smooth_intercept(self, path, reference, *, cancelled=lambda: False):
        """Precheck braking with a visual attachment; never label it measured."""
        previous = self.payload_local, self.payload_tree, getattr(self, 'capture_tcp', None)
        xyz, quat = reference.sample(reference.follow_end_s)
        tcp = np.eye(4)
        tcp[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        tcp[:3, 3] = xyz
        self.payload_local = (self.target_points_at(reference.start_s+reference.follow_end_s)-xyz)@tcp[:3, :3]
        self.payload_tree = cKDTree(self.payload_local)
        self.capture_tcp = tcp
        try:
            return self.check_path(path, self.config.settings.open_width_m, origin=reference.start_s,
                contact_time=reference.contact_s, reference=reference, cancelled=cancelled)
        finally:
            self.payload_local, self.payload_tree, self.capture_tcp = previous

    def attach_measured(self, state, stamp):
        self.model.check_feedback(state)
        self.capture_tcp = self.model.tcp().copy()
        from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
        joint_stamp = mapped_stamp(state, 'joint')
        if joint_stamp is not None:
            stamp = joint_stamp
        # Position is the frozen visual prediction; tool pose is actual arm
        # feedback. This is explicitly an estimate, not proof of attachment.
        self.payload_local = (self.target_points_at(stamp)-self.capture_tcp[:3, 3]) @ self.capture_tcp[:3, :3]
        self.payload_tree = cKDTree(self.payload_local)

    def reanchor_support(self, state):
        """Begin lifting at the stopped TCP; retain the grip-time attachment."""
        if self.payload_local is None:
            raise PickRejected('减速结束但没有夹持时刻的载物估计')
        self.model.check_feedback(state)
        self.capture_tcp = self.model.tcp().copy()

    def release_measured(self, state):
        self.model.check_feedback(state)
        tcp = self.model.tcp()
        # Same limitation as the existing measured moving context: a release
        # estimate is not a fresh visual measurement or settled-bin proof.
        self.released_points = self.payload_local @ tcp[:3,:3].T + tcp[:3,3]
        self.released_tree = cKDTree(self.released_points)
        self.payload_local = None

    def validate_release_opening(self, state):
        count = int(np.ceil(abs(self.config.settings.open_width_m-state['width_m'])/.01))
        for width in np.linspace(state['width_m'],self.config.settings.open_width_m,count+1):
            self.valid(state['joints'],width,payload=True)

    def route(self, state, *, stage, goal_pose=None, goal_joints=None, payload=False,
              cancelled=lambda: False, start_joints=None):
        self._progress('route_start', trajectory_stage=stage)
        begun = self.clock()
        motion = select_motion_profile(self.config, payload=payload)
        q0 = np.asarray(state['joints'] if start_joints is None else start_joints).copy()
        width = float(state['width_m'])
        from .route_diagnostics import RouteDiagnostics
        evidence = RouteDiagnostics(stage=stage, identity=self.identity, payload=payload,
            start=q0, measured=state['joints'], width=width, goal_joints=goal_joints,
            goal_pose=goal_pose, scene_revision=self.scene_revision, scene_report=self.scene_report,
            budget_s=self.config.raw['planning_budget_s'])
        def report(outcome):
            return evidence.snapshot(outcome=outcome, elapsed_s=self.clock()-begun,
                cancelled=cancelled(), timed_out=self.clock()-begun > self.config.raw['planning_budget_s'])
        if payload and self.payload_local is None:
            raise PickRejected('未建立实测点云载物估计',
                details={'route_diagnostics':report('payload_estimate_missing')})
        if np.max(abs(q0-np.asarray(state['joints']))) > TRAJECTORY_ARRIVAL_TOLERANCE_RAD:
            raise PickRejected('后续路径的指令起点与实际反馈偏差过大',
                details={'route_diagnostics':report('start_feedback_mismatch')})
        smooth = self.config.raw.get('trajectory_profile') == 'smooth'
        if smooth:
            from .precise_ik import PrecisePoseSolver
            solver = PrecisePoseSolver(self.model)
        rng = np.random.default_rng(41)
        current_goal = None
        def abort():
            return cancelled() or self.clock()-begun > self.config.raw['planning_budget_s']
        def valid(q):
            role = ('start' if np.array_equal(q, q0) else
                    'goal' if current_goal is not None and np.array_equal(q, current_goal) else 'path')
            self.geometry_context = {'trajectory_stage':stage, 'route_state_role':role,
                                     'scene_revision':self.scene_revision}
            try:
                self.valid(q, width, payload=payload)
                evidence.checked(role, q)
                return True
            except PickRejected as error:
                evidence.checked(role, q, error)
                return False
        def edge(a, b):
            n = max(1, int(np.ceil(np.max(np.abs(b-a))/.008)))
            return all(not abort() and valid(a+(b-a)*u) for u in np.linspace(0, 1, n+1))
        planner = ConstrainedRRTConnect(self.model.command_lower, self.model.command_upper, valid,
            config=SafePathConfig(normalized_step_size=.12, execution_joint_step=.08,
                                  max_iterations=1500, shortcut_attempts=40),
            seed=41, cancelled=abort, segment_valid=edge)
        def goals():
            if goal_joints is not None:
                yield np.asarray(goal_joints)
            if goal_pose is not None:
                pose = np.asarray(goal_pose)
                quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
                for seed in [q0, *rng.uniform(self.model.command_lower, self.model.command_upper, size=(8, 7))]:
                    if abort():
                        return
                    evidence.ik_attempts += 1
                    ik_started = self.clock()
                    try:
                        solved = (solver.solve(pose[:3, 3], quaternion, seed, cancelled=abort) if smooth else
                                  self.model.solve_pose(pose[:3, 3], quaternion, seed, cancelled=abort))
                    except PickRejected as error:
                        raise PickRejected(str(error), details={**error.details,
                            'route_diagnostics':report('goal_ik_error')}) from error
                    finally:
                        evidence.times['goal_ik'] += self.clock()-ik_started
                    if not solved.success:
                        evidence.ik_failures += 1
                        if len(evidence.ik_examples) < 3:
                            evidence.ik_examples.append({'seed_joints_rad':np.asarray(seed).tolist(),
                                'position_error_m':getattr(solved,'position_error',None),
                                'orientation_error_rad':getattr(solved,'orientation_error',None),
                                'message':getattr(solved,'message',None)})
                    if solved.success:
                        # Attempt this complete route before paying for more IK
                        # branches. All later branches retain the same deadline.
                        yield solved.joint_positions
        for goal in goals():
            current_goal = np.asarray(goal).copy()
            self._progress('route_search', trajectory_stage=stage,
                goal_index=len(evidence.goals), goal_joints_rad=current_goal.tolist())
            before = evidence.rejection_counts.copy()
            search_started = self.clock()
            try:
                path = planner.plan(q0, goal)
            finally:
                search_wall = self.clock()-search_started
                evidence.times['path_search'] += search_wall
            if path is None:
                changed = evidence.rejection_counts-before
                role = ('start_rejected' if any(k.startswith('start | ') for k in changed) else
                        'goal_rejected' if any(k.startswith('goal | ') for k in changed) else
                        'cancelled_or_budget' if abort() else 'search_exhausted')
                evidence.goals.append({'goal_joints_rad':current_goal.tolist(), 'outcome':role,
                    'search_wall_s':search_wall})
                continue
            evidence.goals.append({'goal_joints_rad':current_goal.tolist(), 'outcome':'path_found',
                'search_wall_s':search_wall, 'waypoints':len(path)})
            if len(path) == 1:
                path = [path[0], path[0].copy()]
            original_waypoints = len(path)
            if smooth:
                path = merge_collinear_waypoints(path)
                if len(path) == 1:
                    path = [path[0], path[0].copy()]
            # Each checked chord uses a quintic with zero q'/q'' at corners.
            # Cubic interpolation is then fully rechecked, including extrema
            # and scene geometry; no unverified RRT shortcut is executed.
            timing_started = self.clock()
            times, qs = [0.], [q0.copy()]
            for a, b in zip(path, path[1:]):
                delta = abs(b-a)
                seconds = max(float(np.max(1.875*delta/motion.velocity_limits)),
                              float(np.max(np.sqrt(5.78*delta/motion.acceleration_limits))), .1)
                n = max(4, int(np.ceil(seconds*self.config.settings.control_hz)))
                for i in range(1, n+1):
                    u = i/n
                    qs.append(a+(b-a)*(10*u**3-15*u**4+6*u**5))
                    times.append(times[-1]+seconds/n)
            self.version += 1
            try:
                try:
                    if smooth:
                        curve = retime_samples(times, qs, motion, self.model.command_lower,
                            self.model.command_upper, identity=self.identity, version=self.version, stage=stage)
                        curve = replace(curve, diagnostics={**curve.diagnostics,
                            'original_waypoints': original_waypoints, 'waypoints': len(path),
                            **motion.diagnostics()})
                    else:
                        curve = certify_curve(times, qs, motion, self.model.command_lower, self.model.command_upper,
                            identity=self.identity, version=self.version, stage=stage, planning_s=self.clock()-begun)
                finally:
                    evidence.times['time_parameterization'] += self.clock()-timing_started
                geometry_started = self.clock()
                try:
                    self.check_path(curve, width, payload=payload, cancelled=abort)
                finally:
                    evidence.times['curve_geometry'] += self.clock()-geometry_started
                if abort():
                    raise PickRejected('搬运路线预算耗尽')
                return replace(curve, planning_s=self.clock()-begun,
                    diagnostics={**curve.diagnostics, 'route_diagnostics':report('delivered')})
            except PickRejected as error:
                if len(evidence.curve_failures) < 6:
                    evidence.curve_failures.append({'reason':str(error), 'details':error.details})
                if abort():
                    break
        outcome = ('cancelled' if cancelled() else 'planning_timeout' if abort() else
                   'no_ik_solution' if goal_pose is not None and not evidence.goals else
                   'no_certified_route')
        raise PickRejected(stage+' 未找到通过载物/场景检查的路线',
                           details={'route_diagnostics':report(outcome)})
