#!/usr/bin/env python3
"""Wall-clock MuJoCo contact comparison using the production shared policy.

The default input is the original catalog grasp and delayed simulator-history
observations. It is an explicit control/physics diagnostic, not a new camera or
GraspGen-X experiment. Physics runs on its own thread with its own model/data;
the policy and its original global worker use a separate planning mirror. Only
the mirror receives copied states. Only the original factory spawns and recycles terminal objects; an in-flight
object is never repositioned, welded or marked held by this harness.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import zipfile

os.environ.setdefault('MUJOCO_GL', 'egl')
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'scripts')]

import mujoco
import numpy as np

from mujoco_conveyor_sorting import ConveyorSortingEnv
from mujoco_conveyor_sorting.controllers.visual_acquisition import VisualAcquisitionSortingPolicy
from nero_core.sim_common import make_policy
from experiments.continuous_intercept_sim.metrics import contact_summary
from experiments.continuous_intercept_sim.full_preplanned import FullPickMirrorMixin,FullPickPhysicalMixin,completed_lift_hold,finish_contact_lift_experiment


def serializable(value):
    if is_dataclass(value):
        return serializable(asdict(value))
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False,
        indent=2, allow_nan=False)+'\n', encoding='utf-8')


def trace_policy_info(policy):
    """Per-cycle changes; immutable certificates/events have dedicated files."""
    info=dict(policy.get_info())
    if 'full_preplanned' in info:
        full=dict(info['full_preplanned'])
        full['recorded_event_count']=len(full.pop('events',()))
        full['recorded_prediction_count']=len(full.pop('prediction_diagnostics',()))
        static={'attempts','time_refinements','response_redundancy_candidates','response_projection_candidates','response','curve_checks',
            'phase_sweep_batch_passes','phase_sweep_time_fallbacks'}
        full['metrics']={k:v for k,v in full.get('metrics',{}).items() if k not in static}
        full['complete_diagnostics_files']=['policy_info.json','planning_results.json','worker_events.json']
        info['full_preplanned']=full
    return deepcopy(info)


def action_record(action):
    record={k:deepcopy(v) for k,v in action.items() if not k.startswith('_full_pick_')}
    bundle=action.get('_full_pick_bundle')
    if bundle is not None:record['full_pick_request_id']=bundle.request.request_id
    return record


def overrides(speed, seconds, spawn_x, cycles=1):
    return {'scene': {
        'spawning': {'x': spawn_x, 'interval_seconds': seconds+100.},
        'episode': {'object_count': cycles, 'initial_objects': 1, 'max_time': seconds+1.},
        'randomization': {'enabled': False},
        'conveyor': {'default_speed': speed, 'speed_jitter_std': 0.},
        'workspace': {'dynamic_intercept': {'enabled': True}},
        'debug': {'enabled': False}}}


def make_environment(seed, speed, seconds, spawn_x, cls=ConveyorSortingEnv, *,
                     category_id=None,spawn_y=None,spawn_yaw=None,cycles=1,random_objects=False):
    if random_objects or any(x is not None for x in (category_id,spawn_y,spawn_yaw)):
        parent=cls
        selected_category=category_id
        class ExplicitSpawn(parent):
            def spawn_object(self,category_id=None,*,x=None,y=None,yaw=None):
                if random_objects and category_id is None:
                    # Reuse the environment's original seeded category deck.
                    # It varies classes independently of control/recycle timing.
                    deck=self._automatic_category_manifest
                    category_id=int(deck[self.metrics.total_objects%len(deck)])
                return super().spawn_object(selected_category if category_id is None else category_id,
                    x=x,y=spawn_y if y is None else y,yaw=spawn_yaw if yaw is None else yaw)
        cls=ExplicitSpawn
    env = cls(render_mode=None, action_type='joint_position',
        enable_motion_compensation=True, grasp_assist=False,
        render_observations=False, perception_mode='simulator_history',
        grasp_event_source='physical_contacts', seed=seed,
        config_overrides=overrides(speed, seconds, spawn_x,cycles))
    env.reset(seed=seed, options={'belt_speed': speed, 'initial_objects': 1})
    return env


class PlanningMirror(ConveyorSortingEnv):
    """Read-only physical state copies plus the original policy environment.

    This model is never stepped and has no device. The original geometry checks
    and isolated global planner work on it while the physical model progresses.
    Its lifecycle state is policy bookkeeping, not physical grasp evidence.
    """
    def adopt_snapshot(self, snap):
        self._physical_grasp_confirmations = dict(snap['physical_grasp_confirmations'])
        self._physical_object_state_by_id=dict(snap['physical_object_states'])
        self.data.qpos[:] = snap['qpos']
        self.data.qvel[:] = snap['qvel']
        self.data.ctrl[:] = snap['ctrl']
        self.data.time = snap['sim_s']
        self._current_belt_speed = snap['belt_speed']
        self._last_gripper_command = float(snap['gripper']['command'])
        self.gripper_controller._target_open_fraction = float(snap['gripper']['target_open_fraction'])
        self.gripper_controller._last_action = self._last_gripper_command
        # Payload geometry and lifecycle feedback must come from the physical
        # contact evaluator, not the unstepped mirror's stale policy state.
        # Mirror the physical factory's logical identity and lifecycle. No reset
        # or physical pose write occurs here; only this unstepped copy changes.
        records=snap.get('physical_object_records')
        if records is not None:
            self._logical_to_slot.clear()
            for runtime,record in zip(self._objects,records):
                if runtime.logical_id != record['logical_id']:
                    runtime.logical_id=record['logical_id']
                    for name,value in record.items():
                        if name not in {'slot','metadata','body_mass','body_inertia'}:setattr(runtime,name,value)
                    runtime.metadata.pop('_supported_until',None)
                    runtime.metadata.pop('_lane_y',None)
                    runtime.metadata.update(record['metadata'])
                    for geom in runtime.geom_ids:
                        base=self._base_geom_properties[geom]
                        self.model.geom_contype[geom]=base['contype'] if runtime.logical_id is not None else 0
                        self.model.geom_conaffinity[geom]=base['conaffinity'] if runtime.logical_id is not None else 0
                        self.model.geom_rgba[geom]=base['rgba']
                        if runtime.logical_id is None:self.model.geom_rgba[geom,3]=0.
                    self.model.body_mass[runtime.body_id]=record['body_mass']
                    self.model.body_inertia[runtime.body_id]=record['body_inertia']
                if runtime.logical_id is not None:
                    self._logical_to_slot[int(runtime.logical_id)]=runtime
        for runtime in self._objects:
            identifier = runtime.logical_id
            physical_state = snap['physical_object_states'].get(identifier)
            if physical_state is not None and (physical_state == 'GRASPED'
                    or runtime.state == 'GRASPED'):
                runtime.state = physical_state
        mujoco.mj_forward(self.model, self.data)
        self.robot_controller.adopt_joint_dispatch_history(
            snap['joint_target'], snap['joint_target_velocity'])
        self._record_belt_motion()
        self._record_truth_state()
        self._capture_cameras()
        obs = self.get_observation(include_camera_images=False)
        # These quantities come from the physical model's actual contact and
        # controller history, not mirror contact reconstruction or target state.
        obs['gripper'] = deepcopy(snap['gripper'])
        obs['gripper_opening'] = float(snap['gripper']['opening_width'])
        obs['control_dt'] = snap['control_dt']
        obs['dispatch_clock'] = {'scheduled_dispatch_policy_time_s': snap['sim_s']}
        for target in obs['objects']:
            prediction = obs['predictions'].get(target['object_id'])
            if prediction is None:
                continue
            # Generic simulator predictions describe the object's support
            # origin. Supply an explicitly named TCP pose; do not relabel the
            # original object pose as an end-effector pose.
            grasp = self.policy._grasp_target_pose(target, obs, remember_orientation=False)
            if grasp is not None:
                grasp = np.asarray(grasp).copy()
                grasp[:3] = np.asarray(prediction['current_estimated_pose'])[:3]
                grasp[2] += float(target.get('grasp_height', self.policy.config.grasp_clearance))
                prediction['current_ee_grasp_pose'] = grasp
        self._comparison_observation = obs
        return obs

    def is_object_grasped(self, object_id):
        return bool(getattr(self,'_physical_grasp_confirmations',{}).get(int(object_id),False))

    is_target_grasped = is_object_grasped

    def preview_fixed_high_grasp_plan(self, object_id, high_pose, remaining_high_motion,
            descent_duration, *, capture_high=False, capture_time=None):
        """The live fixed-high projection, adapted to this diagnostic sensor.

        Same current-target constant-velocity projection and reachable window;
        only the simulated observation transport differs from the live adapter.
        """
        state = next((x for x in self._comparison_observation['objects']
                      if x['object_id'] == object_id), None)
        if state is None:
            return {'accepted': False, 'reason': 'diagnostic_target_unavailable'}
        now = float(self.data.time)
        velocity = np.asarray(state.get('estimated_velocity', state['linear_velocity']))
        current = np.r_[state['position'], self.policy._grasp_target_pose(
            state, self._comparison_observation, remember_orientation=False)[3:]]
        current[2] += float(state.get('grasp_height', self.policy.config.grasp_clearance))
        speed2 = float(velocity @ velocity)
        crossing = float((np.asarray(high_pose)[:3]-current[:3]) @ velocity)/speed2 if speed2 > 0 else 0.
        start = crossing-(descent_duration if capture_high else 0.)
        if capture_time is None:
            capture_time = now+max(float(remaining_high_motion), start, 0.)+descent_duration
        if capture_time < now:
            return {'accepted': False, 'reason': 'fixed_high_checked_capture_elapsed'}
        predicted = current.copy()
        predicted[:3] += velocity*(capture_time-now)
        window_end = float(state.get('grasp_window_end', float('inf')))
        if (capture_time > window_end-self.policy.config.intercept_deadline_margin
                or not self.grasp_region[0] <= predicted[0] <= self.grasp_region[1]):
            return {'accepted': False, 'reason': 'fixed_high_capture_outside_window'}
        return {'accepted': True, 'reason': 'ok', 'estimated_grasp_time': float(capture_time),
                'predicted_grasp_pose': predicted, 'crossing_time': now+crossing}


class PhysicsRunner:
    """One owner for actual physics; policy computation never holds its lock."""
    def __init__(self, env, output, seconds, *, video=False,evaluate_sorting=False,speed_profile=None):
        self.env, self.output, self.seconds = env, Path(output), float(seconds)
        self.video = video
        self.speed_profile = speed_profile
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.thread = None
        self.pending = None
        self.feed_pending = False
        self.spawn_records=[]
        self.last = None
        self.rows = []
        self.commands = []
        self.error = None
        self.finished = False
        self.wall_origin = None
        self.sim_origin = float(env.data.time)
        self.sorting_evaluator=None
        if evaluate_sorting:
            from experiments.graspgenx_sim.continuous import ContinuousSortingEvaluator
            self.sorting_evaluator=ContinuousSortingEvaluator(env,retain_sorted_until_evaluated=True)
        self.runtime = next(x for x in env._objects if x.logical_id is not None)
        self.object_geoms = set(self.runtime.geom_ids)
        self.belt_geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, 'conveyor_surface')
        self._publish(0.)

    def submit(self, action, stage, requested_at):
        # Certificates are immutable. Avoid copying an entire trajectory at
        # every tick; all mutable action and diagnostic values remain owned.
        value = {k:(v if k=='_full_pick_bundle' else deepcopy(v)) for k,v in action.items()}
        with self.lock:
            self.pending = (value, stage, requested_at)

    def request_next_object(self):
        with self.lock:self.feed_pending=True

    def snapshot(self,*,after_sim=None):
        with self.lock:
            if after_sim is not None and self.last['sim_s']<=after_sim:return None
            return deepcopy(self.last)

    def sim_time(self):
        with self.lock:return float(self.last['sim_s'])

    def _publish(self, wall):
        env = self.env
        if self.sorting_evaluator is not None:self.sorting_evaluator.update()
        grip = env._gripper_observation()
        active=next((r for r in env._objects if r.logical_id is not None),None)
        if active is not None:
            self.runtime=active;self.object_geoms=set(active.geom_ids)
        runtime = self.runtime
        pose = env.data.qpos[runtime.qpos_adr:runtime.qpos_adr+7].copy()
        vel = env.data.qvel[runtime.qvel_adr:runtime.qvel_adr+6].copy()
        object_forces = env._finger_object_contact_forces(runtime)
        belt_contact = any(self.belt_geom in (c.geom1, c.geom2)
            and bool({int(c.geom1), int(c.geom2)} & self.object_geoms)
            for c in env.data.contact[:env.data.ncon])
        row = {'sim_s': float(env.data.time), 'wall_s': float(wall),
            'belt_speed_m_s': float(env._current_belt_speed),
            'joint_q': env.data.qpos[env.arm_qpos_adrs].copy(),
            'joint_v': env.data.qvel[env.arm_dof_adrs].copy(),
            'joint_command': env.robot_controller.commanded_joint_positions.copy(),
            'command_velocity': env.robot_controller.commanded_joint_velocities.copy(),
            'tcp_pose': env._ee_pose().copy(), 'object_pose': pose, 'object_velocity': vel,
            'gripper_width_m': float(grip['opening_width']),
            'gripper_command': float(grip['command']),
            'left_normal_force_n': float(object_forces[0]),
            'right_normal_force_n': float(object_forces[1]),
            'force_scope': 'each_finger_on_selected_object_only',
            'object_id': -1 if runtime.logical_id is None else int(runtime.logical_id),
            'object_active': runtime.logical_id is not None,
            'object_belt_contact': belt_contact, 'physical_lifecycle': runtime.state,
            'active_assist_weld_count': int(sum(env.data.eq_active[i] for i in env._assist_eq_ids.values())),
            'eq_active_count': int(np.count_nonzero(env.data.eq_active))}
        self.rows.append(row)
        snap = {'sim_s': float(env.data.time), 'wall_s': float(wall),
            'qpos': env.data.qpos.copy(), 'qvel': env.data.qvel.copy(), 'ctrl': env.data.ctrl.copy(),
            'joint_target': row['joint_command'].copy(),
            'joint_target_velocity': row['command_velocity'].copy(),
            'belt_speed': float(env._current_belt_speed), 'gripper': deepcopy(grip),
            'physical_grasp_confirmations': {int(r.logical_id): env.is_object_grasped(r.logical_id)
                for r in env._objects if r.logical_id is not None},
            'physical_object_states': {int(r.logical_id): r.state
                for r in env._objects if r.logical_id is not None},
            'control_dt': env.control_dt, 'object_pose': pose, 'object_velocity': vel}
        snap['geom_rgba']=env.model.geom_rgba.copy()
        snap['physical_object_records']=[dict(slot=r.slot,logical_id=r.logical_id,
            body_mass=float(env.model.body_mass[r.body_id]),body_inertia=env.model.body_inertia[r.body_id].copy(),
            state=r.state,spawned_at=r.spawned_at,state_changed_at=r.state_changed_at,
            **{name:deepcopy(getattr(r,name)) for name in ('selected_at','grasped_at','released_at',
                'lost_contact_since','stable_contact_steps','contact_gap_steps','prediction_at_grasp',
                'actual_grasp_pose','planned_grasp_pose','planned_grasp_time','compensation_enabled_at_selection','failure_reason')},
            metadata={k:r.metadata[k] for k in ('_lane_y',) if k in r.metadata})
            for r in env._objects]
        snap['full_pick_dispatch_fault']=getattr(env,'full_pick_dispatch_fault',None)
        snap['last_actuator_application_sim_s']=getattr(env,'last_actuator_application_sim_s',None)
        if self.sorting_evaluator is not None:
            snap['independent_sorted_ids']=[identifier for identifier,row in self.sorting_evaluator.records.items() if row['success']]
        with self.lock:
            self.last = snap

    def start(self):
        self.thread = threading.Thread(target=self._loop, name='independent-contact-physics', daemon=True)
        self.thread.start()

    def _loop(self):
        writer = None
        try:
            if self.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(str(self.output/'physical.mp4'), fps=10,
                    codec='libx264', quality=7, macro_block_size=1)
                writer.append_data(self.env._render_camera('policy_external_camera', (640,480), depth=False))
            self.wall_origin = time.monotonic()
            next_video = .1
            while not self.done.is_set() and self.env.data.time-self.sim_origin < self.seconds:
                elapsed = time.monotonic()-self.wall_origin
                if self.env.data.time-self.sim_origin > elapsed:
                    self.done.wait(min(.002, self.env.data.time-self.sim_origin-elapsed))
                    continue
                with self.lock:
                    request, self.pending = self.pending, None
                action = None if request is None else request[0]
                if request is not None:
                    self.commands.append({'physics_receive_sim_s': float(self.env.data.time),
                        'physics_receive_wall_s': elapsed, 'stage': request[1],
                        'policy_requested_sim_s': request[2], 'action': action_record(action)})
                if self.speed_profile is not None:
                    values=np.asarray(self.speed_profile,dtype=float)
                    self.env.set_belt_speed(float(np.interp(self.env.data.time,values[:,0],values[:,1])))
                self.env.step(action)
                with self.lock:feed=self.feed_pending
                if feed:
                    # Only the environment's original terminal recycle/spawn
                    # methods may reposition a body; the arm is never reset.
                    identifier=self.env.spawn_object(x=self.env.spawn_x)
                    if identifier is not None:
                        self.spawn_records.append(dict(object_id=int(identifier),spawned_at_s=float(self.env.data.time),
                            category_id=int(self.env._logical_to_slot[identifier].category_id)))
                        with self.lock:self.feed_pending=False
                self._publish(time.monotonic()-self.wall_origin)
                if writer is not None and self.env.data.time >= next_video:
                    writer.append_data(self.env._render_camera('policy_external_camera',(640,480),depth=False))
                    next_video = float(self.env.data.time)+.1
        except BaseException as exc:
            self.error = f'{type(exc).__name__}: {exc}'
        finally:
            if writer is not None:
                writer.close()
            self.finished = True

    def stop(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=10.)
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError('physics owner did not terminate')


def run_case(args, mode, output):
    if getattr(args, 'planning_mode', None) != 'full_preplanned':
        raise ValueError('only full_preplanned is permitted')
    output.mkdir(parents=True, exist_ok=False)
    physical = mirror = runner = policy = None
    live_viewer = None
    began_wall = time.monotonic()
    def progress(message):
        print(f'[物理仿真 +{time.monotonic()-began_wall:6.1f}s] {message}', flush=True)
    progress(f'模式={mode}/{args.planning_mode}；输出={output.resolve()}')
    policy_rows, events = [], []
    summary = {'mode': mode, 'seed': args.seed, 'belt_speed_m_s': args.belt_speed,
        'input_source': 'original_catalog_grasp_and_simulator_history_diagnostic',
        'network_executed': False, 'camera_pipeline_executed': False,
        'hardware_connections': 0, 'grasp_assist': False, 'harness_object_pose_writes_after_initialization': 0,
        'physical_model': 'original_ConveyorSortingEnv_actuators_objects_contacts_gravity',
        'timing': 'physics has separate model and wall-clock thread; control or worker calculation does not pause it',
        'known_limitations': ['Diagnostic simulator-history perception, not a new GraspGen-X or camera test.',
            'This simulated belt/base layout differs from the saved 181031 real scene.',
            'Wall scheduling varies; same seeds do not imply bit-identical timing.'],
        'status': 'starting'}
    write_json(output/'summary.json', summary)
    try:
        progress('正在加载物理场景和独立规划模型…')
        full=getattr(args,'planning_mode','legacy')=='full_preplanned'
        through_bin=bool(getattr(args,'through_bin',False))
        cycles=int(getattr(args,'cycles',1))
        from experiments.continuous_intercept_sim.sorting_cycles import SortingCycles
        cycle_book=SortingCycles(cycles)
        summary['execution_scope']=('single_pick_through_lift_then_fixed_hold' if full else 'original_shared_sorting_policy')
        if through_bin:summary['execution_scope']='sequential_pick_correct_bin_return'
        summary['requested_cycles']=cycles
        summary['random_objects']=bool(getattr(args,'random_objects',False))
        summary['object_pool_policy']='original terminal recycle then original factory spawn; no arm reset; objects do not accumulate in bin'
        class FullPhysical(FullPickPhysicalMixin,ConveyorSortingEnv):pass
        class FullMirror(FullPickMirrorMixin,PlanningMirror):pass
        mirror_class=FullMirror if full else PlanningMirror
        if getattr(args,'network_grasp_replay',None) is not None:
            from experiments.continuous_intercept_sim.saved_network_grasp import SavedNetworkGraspMirrorMixin
            class ReplayMirror(SavedNetworkGraspMirrorMixin,FullMirror):pass
            mirror_class=ReplayMirror
        physical = make_environment(args.seed,args.belt_speed,args.seconds,args.spawn_x,
                                    FullPhysical if full else ConveyorSortingEnv,
                                    category_id=args.category_id,spawn_y=args.spawn_y,spawn_yaw=args.spawn_yaw,cycles=cycles,random_objects=getattr(args,'random_objects',False))
        mirror = make_environment(args.seed,args.belt_speed,args.seconds,args.spawn_x,
                                  mirror_class,
                                  category_id=args.category_id,spawn_y=args.spawn_y,spawn_yaw=args.spawn_yaw,cycles=cycles,random_objects=getattr(args,'random_objects',False))
        if getattr(args,'network_grasp_replay',None) is not None:
            mirror.configure_saved_network_grasp(args.network_grasp_replay,args.network_candidate_index)
            summary['input_source']='saved_GraspGenX_GraspMoE_pose_with_current_simulator_history_diagnostic'
            summary['saved_network_grasp']=mirror.saved_network_provenance
            summary['fresh_vision_or_network_executed']=False
        mirror.simulation_command_advance_s = float(args.simulation_command_advance_s)
        summary['simulation_command_advance'] = {'requested_s': mirror.simulation_command_advance_s,
            'simulation_only': True, 'measured_response_model': False,
            'requires_original_command_limits_and_complete_geometry_certificate': True,
            'fallback_to_unadvanced_trajectory': False}
        mirror.contact_orientation_preparation_s=float(getattr(args,'orientation_preparation_s',0.))
        summary['orientation_preparation_s']=mirror.contact_orientation_preparation_s
        mirror.contact_orientation_tracking_entry=bool(getattr(args,'prepare_tracking_entry',False))
        summary['orientation_preparation_tracking_entry']=mirror.contact_orientation_tracking_entry
        mirror.contact_acceleration_m_s2=float(getattr(args,'contact_acceleration',0.))
        summary['contact_acceleration_m_s2']=mirror.contact_acceleration_m_s2
        mirror.contact_finger_entry_height_m=float(getattr(args,'finger_entry_height',0.))
        mirror.contact_finger_entry_lead_s=float(getattr(args,'finger_entry_lead_s',.8))
        summary['finger_entry_shape']={'height_increment_m':mirror.contact_finger_entry_height_m,
            'lead_s':mirror.contact_finger_entry_lead_s,'arrival_gate':False}
        mirror.contact_response_projection_fraction=float(getattr(args,'response_projection_fraction',.25))
        summary['response_projection_fraction']=mirror.contact_response_projection_fraction
        mirror.contact_approach_response_enabled=not bool(getattr(args,'no_approach_response',False))
        summary['approach_response_enabled']=mirror.contact_approach_response_enabled
        if args.contact_station_x is not None:
            if not mirror.grasp_region[0]<=args.contact_station_x<=mirror.grasp_region[1]:
                raise ValueError('contact station must be inside the original grasp region')
            mirror.contact_station_x_m=args.contact_station_x
            summary['diagnostic_contact_station_x_m']=args.contact_station_x
        if getattr(args,'intercept_x_min',None) is not None:
            for env in (physical,mirror):
                if not env.dynamic_intercept_x_range[0]<=args.intercept_x_min<env.dynamic_intercept_x_range[1]:
                    raise ValueError('candidate intercept minimum must tighten the original interval')
                env.dynamic_intercept_x_range=(args.intercept_x_min,env.dynamic_intercept_x_range[1])
            summary['diagnostic_intercept_x_range_m']=mirror.dynamic_intercept_x_range
        from experiments.continuous_intercept_sim.response_diagnostics import response_model_status
        response_model = None
        if getattr(args,'joint_response_model',None) is not None:
            from mujoco_conveyor_sorting.controllers.joint_response import JointResponseModel
            model_path=args.joint_response_model
            model=JointResponseModel.from_dict(json.loads(model_path.read_text()))
            if not model.metadata.get('simulation_only',False):
                raise ValueError('physical simulation requires an explicit simulation_only response model')
            mirror.joint_response_model=model
            response_model = model
            summary['joint_response_model']={'path':str(model_path),
                'sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'metadata':model.metadata}
        response_status = response_model_status(response_model,
            path=getattr(args, 'joint_response_model', None))
        summary['response_model_admission'] = response_status
        if not response_status['configured'] and mirror.simulation_command_advance_s:
            progress(f'仅仿真：请求提前{mirror.simulation_command_advance_s*1000:g}毫秒的命令曲线；须完整认证，未使用已测响应模型')
        elif not response_status['configured']:
            progress('未配置关节响应模型；完整轨迹沿用名义指令，响应补偿未启用')
        elif not response_status['eligible']:
            progress('关节响应模型未通过资格检查；沿用名义指令，逐轴原因保存在summary')
        else:
            progress('关节响应模型已加载并通过资格检查；补偿是否启用仍须由完整曲线认证决定')
        if through_bin:
            # Primary request-bound loaded planner owns the ordinary handoff.
            # The optional generic prefetch has no request-bound payload.
            mirror.create_prefetch_planning_worker=lambda: None
            summary['speculative_prefetch']=False
        base = make_policy(mirror,True)
        config = base.config
        base.close()
        policy = VisualAcquisitionSortingPolicy(config, env=mirror,
            acquisition_translation_step_m=.05, acquisition_command_integrator=True,
            visual_acquisition_tracking=True, visual_servo_bias_compensation=True)
        summary['control_policy_class'] = type(policy).__name__
        summary['control_adapter'] = 'same VisualAcquisitionSortingPolicy as live; original simulation config limits retained'
        fields = policy.config.__dataclass_fields__
        if mode == 'continuous' and 'moving_approach_mode' not in fields:
            raise RuntimeError('shared moving_approach_mode implementation is not available yet')
        changes = {'pregrasp_fixed_high_enabled': mode == 'high', 'timed_descent_enabled': mode == 'high',
            'dynamic_intercept_tracking': True, 'enable_speculative_prefetch': False}
        changes = {k:v for k,v in changes.items() if k in fields}
        if 'moving_approach_mode' in fields:
            changes['moving_approach_mode'] = mode
        policy.config = replace(policy.config,**changes)
        if full:
            policy.config=replace(policy.config,planning_mode='full_preplanned',
                selection_policy='first_safe',replan_policy='event_triggered',
                full_plan_replanning_enabled=not getattr(args,'no_full_replanning',False),
                full_plan_compute_budget_s=args.full_plan_compute_budget_s,
                full_plan_contact_handoff_enabled=getattr(args,'contact_handoff',False))
            if getattr(args,'minimum_approach_s',None) is not None:
                policy.config=replace(policy.config,descent_lead_time=args.minimum_approach_s)
            from mujoco_conveyor_sorting.controllers.full_pick_execution import validate_full_pick_settings
            validate_full_pick_settings(policy.config)
        if through_bin:
            policy.config=replace(policy.config,transfer_timeout=30.,return_timeout=30.,attempt_timeout=90.)
            summary['full_sort_time_budget_reason']='full outward transfer and return at the existing model-derived slower speed; lift_timeout remains unchanged'
        mirror.policy = policy
        summary['policy_config'] = asdict(policy.config)
        summary['scene_overrides'] = overrides(args.belt_speed,args.seconds,args.spawn_x,cycles)
        summary['initial_qpos_sha256'] = hashlib.sha256(physical.data.qpos.tobytes()).hexdigest()
        summary['initial_qpos_matches_mirror'] = bool(np.array_equal(physical.data.qpos,mirror.data.qpos))
        runtime = next(x for x in physical._objects if x.logical_id is not None)
        summary['object_category_id'] = int(runtime.category_id)
        summary['category_manifest']=physical._automatic_category_manifest.tolist()
        summary['randomization_scope']='original seeded category deck only; original dimensions, mass, friction and belt speed retained'
        summary['object_body_mass_kg']=float(physical.model.body_mass[runtime.body_id])
        summary['object_mass_metadata_kg']=runtime.metadata.get('mass')
        summary['object_initial_pose'] = physical.data.qpos[runtime.qpos_adr:runtime.qpos_adr+7].copy()
        summary['model_signature'] = {k:hashlib.sha256(np.asarray(getattr(physical.model,k)).tobytes()).hexdigest()
            for k in ('body_mass','body_inertia','geom_friction','actuator_gainprm','actuator_biasprm','jnt_range',
                      'geom_size','geom_type','geom_dataid','mesh_vert','mesh_face')}
        # Preconstruct independent worker/model before starting the timed trial.
        progress('正在准备规划器…')
        policy._ensure_planning_worker()
        if full:
            geometry_begin=time.perf_counter();geometry_cpu=time.thread_time()
            physical.prepare_full_pick_geometry()
            summary['physical_geometry_preparation_wall_s']=time.perf_counter()-geometry_begin
            summary['physical_geometry_preparation_thread_cpu_s']=time.thread_time()-geometry_cpu
        profile=None
        if getattr(args,'belt_speed_profile',None) is not None:
            profile=json.loads(args.belt_speed_profile.read_text())
            values=np.asarray(profile,dtype=float)
            if (values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all()
                    or values[0,0]!=0. or np.any(np.diff(values[:,0])<=0.)
                    or np.any(values[:,1]<physical.belt_speed_range[0])
                    or np.any(values[:,1]>physical.belt_speed_range[1])
                    or values[0,1]!=args.belt_speed):
                raise ValueError('speed profile must start at [0, initial speed], strictly increase time and retain original belt range')
            summary['controlled_physical_belt_speed_profile']=profile
            summary['controlled_profile_sha256']=hashlib.sha256(args.belt_speed_profile.read_bytes()).hexdigest()
        runner = PhysicsRunner(physical,output,args.seconds,video=not args.no_video,
            evaluate_sorting=through_bin,speed_profile=profile)
        mirror.full_pick_physics_clock=runner.sim_time
        mirror.full_pick_replan_worker_delay_s=float(getattr(args,'replan_worker_delay_s',0.))
        if not math.isfinite(mirror.full_pick_replan_worker_delay_s) or mirror.full_pick_replan_worker_delay_s<0.:
            raise ValueError('worker diagnostic delay must be finite and nonnegative')
        summary['replan_worker_diagnostic_delay_s']=mirror.full_pick_replan_worker_delay_s
        np.savez_compressed(output/'initial_physical_state.npz',qpos=physical.data.qpos.copy(),
            qvel=physical.data.qvel.copy(),ctrl=physical.data.ctrl.copy(),
            qacc_warmstart=physical.data.qacc_warmstart.copy(),act=physical.data.act.copy(),
            simulation_dt_s=physical.model.opt.timestep,control_dt_s=physical.control_dt,
            actuator_gainprm=physical.model.actuator_gainprm.copy(),actuator_biasprm=physical.model.actuator_biasprm.copy())
        if getattr(args, 'viewer', False):
            from experiments.continuous_intercept_sim.viewer import PhysicsViewer
            progress('正在打开实时窗口…')
            live_viewer = PhysicsViewer(runner)
            live_viewer.start()
            progress('实时窗口已打开；关闭窗口将结束本轮并保存记录')
        else:
            progress('后台运行；加--viewer可打开实时窗口')
        runner.start()
        progress(f'物理计时开始，计划运行{args.seconds:g}秒；带速{args.belt_speed*1000:g}毫米/秒')
        previous_sim = -1.
        last_stage = None
        returned_after_sort=False
        handoff_retired=False
        prefix_retirement_revision=0
        last_progress_wall = time.monotonic()
        while not runner.finished:
            snapshot = runner.snapshot(after_sim=previous_sim)
            if snapshot is None:
                time.sleep(.001)
                continue
            previous_sim = snapshot['sim_s']
            observation_began=time.monotonic()
            observation = mirror.adopt_snapshot(snapshot)
            if not policy._full_pick_contact_handed_off:
                handoff_retired=False
            if policy._full_pick_pre_motion_replans<prefix_retirement_revision:
                prefix_retirement_revision=0
            began = time.monotonic()
            fault=snapshot.get('full_pick_dispatch_fault') if full else None
            if full and policy._full_pick_finished:
                action=completed_lift_hold(policy)
            elif full and (fault or policy._full_pick_abort_reason):
                action=policy._full_pick_abort(observation,fault or policy._full_pick_abort_reason)
            else:
                action = policy.act(observation)
            if full and not through_bin and not policy._full_pick_finished:
                finish_contact_lift_experiment(policy,mirror)
            if full and policy.stage.value=='recovering' and not policy._full_pick_abort_reason:
                action=policy._full_pick_abort(observation,policy._last_reason or 'full_pick_task_failed')
            if full and policy._full_pick_finished:
                action=completed_lift_hold(policy)
            elif full and policy._full_pick_abort_reason:
                action=dict(action,_full_pick_stop_reason=policy._full_pick_abort_reason)
            elif full and policy._full_pick_contact_handed_off:
                action=dict(action,_full_pick_shared_config=policy.config)
                if not handoff_retired:
                    action['_full_pick_retire_bundle']=True;handoff_retired=True
            elif full and policy._full_pick_bundle is not None and not policy._full_pick_finished:
                action=dict(action,_full_pick_bundle=policy._full_pick_bundle,
                    _full_pick_lift_admitted=policy._full_pick_lift_admitted,
                    _full_pick_config=policy.config,_full_pick_offsets=mirror.full_pick_offsets)
            if full and policy._full_pick_pre_motion_replans>prefix_retirement_revision:
                action=dict(action,_full_pick_retire_bundle=True)
                prefix_retirement_revision=policy._full_pick_pre_motion_replans
            computed = time.monotonic()
            if full:
                action=dict(action,_full_pick_prepared_at_sim_s=float(observation['timestamp']),
                    _full_pick_sample_sim_s=getattr(policy,'_full_pick_sample_time_s',None),
                    _full_pick_policy_finished_monotonic_s=computed)
            runner.submit(action,policy.stage.value,float(observation['timestamp']))
            after = runner.snapshot()
            target = next((x for x in observation['objects'] if x['object_id']==policy.target_id),None)
            policy_rows.append({'sim_s': float(observation['timestamp']),
                'feedback_joint_q':snapshot['qpos'][mirror.arm_qpos_adrs].copy(),
                'feedback_joint_v':snapshot['qvel'][mirror.arm_dof_adrs].copy(),
                'feedback_tcp':observation.get('ee_pose'),
                'snapshot_wall_elapsed_s':snapshot['wall_s'],
                'observation_started_monotonic_s':observation_began,'policy_started_monotonic_s':began,
                'policy_finished_monotonic_s':computed,
                'physics_after_calculation_sim_s': after['sim_s'],
                'calculation_wall_s': computed-began, 'stage': policy.stage.value,
                'policy_info': trace_policy_info(policy),
                'target_position': None if target is None else target['position'].copy(),
                'target_velocity': None if target is None else np.asarray(target.get('estimated_velocity',target['linear_velocity'])).copy(),
                'target_predicted_pose': None if target is None else target.get('predicted_grasp_pose'),
                'action': action_record(action)})
            events.extend(serializable(e) for e in policy.drain_events())
            now_wall = time.monotonic()
            stage = policy.stage.value
            previous_stage=last_stage
            if through_bin:
                def feed_after_independent_placement():
                    if cycle_book.previous_target not in after.get('independent_sorted_ids',[]):
                        summary['sequence_failure_reason']='previous_object_not_independently_sorted'
                        runner.done.set()
                    else:runner.request_next_object()
                returned_after_sort=cycle_book.observe(stage,policy.target_id,after['sim_s'],feed_after_independent_placement,reason=policy._last_reason)
                if stage=='waiting' and last_stage=='returning':
                    progress(f'已返回轮数={len(cycle_book.returns)}/{cycles}；独立确认入框={len(after.get("independent_sorted_ids",[]))}')
            if stage != last_stage or now_wall-last_progress_wall >= 5.:
                reason = policy._full_pick_abort_reason if full else policy._last_reason
                progress(f'仿真时间={after["sim_s"]:.2f}/{args.seconds:g}秒；阶段={stage}'
                    + (f'；原因={reason}' if reason else ''))
                last_stage, last_progress_wall = stage, now_wall
            if runner.error:
                raise RuntimeError(runner.error)
            if through_bin and stage=='waiting' and previous_stage=='returning' and policy._last_reason!='wait_pose_reached':
                summary['sequence_failure_reason']=policy._last_reason
                runner.done.set()
            if through_bin and returned_after_sort:
                summary['sort_and_return_completed']=True
                runner.done.set()
        if runner.error:
            raise RuntimeError(runner.error)
        if live_viewer is not None and live_viewer.error:
            raise RuntimeError(live_viewer.error)
        summary['status'] = ('interrupted' if live_viewer is not None
            and live_viewer.closed_by_user else 'completed')
    except KeyboardInterrupt:
        summary['status'] = 'interrupted'
        progress('收到中断，正在停止仿真并保存记录…')
    except Exception as exc:
        summary['status'] = 'error'
        summary['error'] = f'{type(exc).__name__}: {exc}'
        import traceback
        (output/'error.txt').write_text(traceback.format_exc())
    finally:
        if runner is not None:
            runner.stop()
        if live_viewer is not None:
            live_viewer.stop()
            summary['viewer'] = live_viewer.summary()
        if runner is not None:
            progress('物理运行结束，正在保存轨迹、接触力和规划记录…')
            summary.update(contact_summary(runner.rows))
            summary['physics_steps'] = len(runner.rows)-1
            summary['max_physics_wall_lag_s'] = max((r['wall_s']-r['sim_s'] for r in runner.rows),default=0.)
            summary['eq_active_max'] = max((r['eq_active_count'] for r in runner.rows),default=0)
            summary['active_assist_weld_max'] = max((r['active_assist_weld_count'] for r in runner.rows),default=0)
            if runner.sorting_evaluator is not None:
                summary['sorting_evaluation']=runner.sorting_evaluator.summary(expected_objects=int(getattr(args,'cycles',1)))
                summary['cycle_returns']=cycle_book.returns
                summary['manual_spawn_records']=runner.spawn_records
                summary['per_object_contact']={str(identifier):contact_summary([r for r in runner.rows if r['object_id']==identifier])
                    for identifier in sorted({r['object_id'] for r in runner.rows if r['object_id']>=0})}
                runner.sorting_evaluator.close()
            summary['other_equalities'] = 'original finger symmetry and parked inactive objects'
            summary['stages'] = list(dict.fromkeys(x['stage'] for x in policy_rows))
            summary['full_pick_dispatch_fault']=getattr(physical,'full_pick_dispatch_fault',None)
            summary['full_pick_stop_reason']=getattr(physical,'full_pick_stop_reason',None)
            write_json(output/'full_pick_dispatch.json',getattr(physical,'full_pick_dispatch_rows',[]))
            write_json(output/'payload_geometry_bindings.json',getattr(mirror,'full_pick_payload_bindings',[]))
            summary['policy_calls'] = len(policy_rows)
            summary['physics_advanced_during_policy_calls_s'] = sum(max(0.,x['physics_after_calculation_sim_s']-x['sim_s']) for x in policy_rows)
            write_json(output/'physical_telemetry.json',runner.rows)
            write_json(output/'dispatched_actions.json',runner.commands)
            if runner.rows:
                np.savez_compressed(output/'physical_trace.npz',**{k:np.array([r[k] for r in runner.rows])
                    for k in ('sim_s','wall_s','belt_speed_m_s','joint_q','joint_v','joint_command','command_velocity','tcp_pose',
                        'object_pose','object_velocity','gripper_width_m','gripper_command',
                        'left_normal_force_n','right_normal_force_n','object_belt_contact','object_id')})
        if policy is not None:
            summary['final_policy_info'] = policy.get_info()
            summary['controller_lift_scope_completed']=bool(policy._full_pick_finished
                and not policy._full_pick_abort_reason and not summary.get('full_pick_dispatch_fault'))
            summary['acceptance_success']=bool(summary.get('physical_lift_and_hold')
                and summary['controller_lift_scope_completed'] and summary.get('status')=='completed')
            if getattr(args,'through_bin',False):
                summary['acceptance_success']=bool(summary.get('sort_and_return_completed')
                    and summary.get('sorting_evaluation',{}).get('successful_sorted_objects')==int(getattr(args,'cycles',1))
                    and summary.get('status')=='completed' and not policy._full_pick_abort_reason
                    and not summary.get('full_pick_dispatch_fault'))
            write_json(output/'policy_info.json',summary['final_policy_info'])
            if getattr(args,'planning_mode','legacy')=='full_preplanned' and mirror is not None:
                write_json(output/'submitted_requests.json',getattr(mirror,'full_pick_submissions',[]))
                results=[]
                for index,result in enumerate(getattr(mirror,'full_pick_results',[])):
                    bundle=result.full_pick_plan
                    row={k:getattr(result,k) for k in ('request_id','success','error','planning_time_s',
                        'effective_request','diagnostics','path_fully_validated','continuation_fully_validated')}
                    if bundle is not None:
                        row['schedule']={k:getattr(bundle,k) for k in ('start','high_time','contact_time','lift_time','end')}
                        stamps=np.array([t for phase,t,q in bundle.checked_samples])
                        values=[bundle.sample_full(t) for t in stamps]
                        np.savez_compressed(output/f'full_pick_curve_{index}.npz',
                            phase=np.array([phase for phase,t,q in bundle.checked_samples]),
                            time_s=stamps,q=np.array([v[0] for v in values]),
                            v=np.array([v[1] for v in values]),a=np.array([v[2] for v in values]))
                        command_samples=bundle.checked_command_samples
                        if command_samples:
                            command_times=np.array([t for _,t,_ in command_samples])
                            commands=[bundle.sample_command_full(t) for t in command_times]
                            np.savez_compressed(output/f'full_pick_command_curve_{index}.npz',
                                phase=np.array([p for p,_,_ in command_samples]),time_s=command_times,
                                q=np.array([v[0] for v in commands]),v=np.array([v[1] for v in commands]),
                                a=np.array([v[2] for v in commands]))
                    results.append(row)
                write_json(output/'planning_results.json',results)
            policy.close()
            if full:
                write_json(output/'worker_events.json',getattr(mirror,'full_pick_worker_events',[]))
        for env in (mirror,physical):
            if env is not None:
                env.close()
        with gzip.open(output/'policy_trace.json.gz','wt',encoding='utf-8') as stream:
            json.dump(serializable(policy_rows),stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        summary['policy_trace_file']='policy_trace.json.gz'
        summary['trace_schema']='dynamic_per_cycle_v2_full_certificates_and_events_in_dedicated_files'
        write_json(output/'events.json',events)
        write_json(output/'summary.json',summary)
    progress(f'记录已保存：{output.resolve()}/summary.json')
    print(json.dumps(serializable({k:summary.get(k) for k in ('mode','status','stages','closure_command_sent',
        'physical_lift_and_hold','max_lift_m','error')}),ensure_ascii=False),flush=True)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('high','continuous','both'),default='continuous')
    parser.add_argument('--seed',type=int,default=1543)
    parser.add_argument('--belt-speed',type=float,default=.04)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--viewer',action='store_true',help='打开只读实时MuJoCo窗口；关闭窗口结束本轮')
    parser.add_argument('--random-objects',action='store_true',help='每轮按原随机类别牌组补料，使用该类原目录抓姿')
    parser.add_argument('--cycles',type=int,default=1,help='同一会话连续完整抓放轮数；轮间不重置机械臂')
    parser.add_argument('--through-bin',action='store_true',help='继续原分类投放和返回；以独立实际入框评分，不在抬升后停止')
    parser.add_argument('--seconds',type=float,default=55.)
    parser.add_argument('--spawn-x',type=float,default=-1.28)
    parser.add_argument('--category-id',type=int,help='仅在原spawn_object初始化时指定原目录类别')
    parser.add_argument('--spawn-y',type=float,help='初始化的横向位置，单位米')
    parser.add_argument('--spawn-yaw',type=float,help='初始化的绕带面法向角度，单位弧度')
    parser.add_argument('--planning-mode',choices=('full_preplanned',),default='full_preplanned',
        help='仅允许 full_preplanned；旧控制器不再作为运行入口')
    parser.add_argument('--simulation-command-advance-s',type=float,default=.1,
        help='仅仿真：命令参考提前秒数；要求全轨迹重新认证，无回退')
    parser.add_argument('--no-full-replanning',action='store_true',help='物理A/B：关闭full执行中重规划，保留原安全判断')
    parser.add_argument('--belt-speed-profile',type=Path,help='物理A/B：[[仿真秒,带速m/s],...]，原传送带接口线性插值')
    parser.add_argument('--replan-worker-delay-s',type=float,default=0.,help='迟到反例：仅延迟后台接续任务，物理线程继续运行')
    parser.add_argument('--full-plan-compute-budget-s',type=float,default=3.)
    parser.add_argument('--minimum-approach-s',type=float,
        help='显式直接接近最短时间候选；改变预约后仍重新完整认证')
    parser.add_argument('--intercept-x-min',type=float,help='在原可抓取区内选择更靠下游的预约位置；不放宽原范围')
    parser.add_argument('--contact-handoff',action='store_true',help='显式实验：完整检查直接接近，预约接触后转原闭合确认与反馈抬升')
    parser.add_argument('--orientation-preparation-s',type=float,default=0.,help='显式在起点位置先转向的时间候选；完整曲线重认证，不等待高位')
    parser.add_argument('--prepare-tracking-entry',action='store_true',help='转向同时到沿带平滑加速的接近入口；Z取本次起点，完整重认证')
    parser.add_argument('--contact-acceleration',type=float,default=0.,help='接触时沿带加速度候选m/s²；速度仍匹配带速，完整重认证')
    parser.add_argument('--finger-entry-height',type=float,default=0.,help='连续曲线中延后夹指进入的高度增量m；不设中间到位门槛，完整重认证')
    parser.add_argument('--finger-entry-lead-s',type=float,default=.8,help='上述曲线形状接点距捕获的秒数')
    parser.add_argument('--response-projection-fraction',type=float,default=.25,help='模型补偿中横向/方向修正的投影比例候选；完整重认证')
    parser.add_argument('--no-approach-response',action='store_true',help='接近沿原名义曲线；保留响应模型供原载物限速')
    parser.add_argument('--contact-station-x',type=float,help='显式模拟低位抓取站X坐标；等待后执行原限时直接接近')
    parser.add_argument('--joint-response-model',type=Path,help='本物理仿真独立校准、标记simulation_only的响应模型')
    parser.add_argument('--network-grasp-replay',type=Path,help='明确的旧网络抓姿回放诊断；本轮不采图/运行网络，仍用模拟历史观测')
    parser.add_argument('--network-candidate-index',type=int,default=4,help='已保存网络输出的source_index')
    args=parser.parse_args(argv)
    if not np.isfinite(args.simulation_command_advance_s) or args.simulation_command_advance_s < 0:
        parser.error('simulation command advance must be finite and nonnegative')
    if args.simulation_command_advance_s and (args.contact_handoff or args.joint_response_model or args.mode != 'continuous'):
        parser.error('simulation command advance requires continuous full plan without contact handoff or response model')
    if args.orientation_preparation_s<0 or not np.isfinite(args.orientation_preparation_s):
        parser.error('orientation-preparation-s must be finite and nonnegative')
    if args.random_objects and (args.category_id is not None or args.network_grasp_replay is not None):
        parser.error('--random-objects cannot reuse a fixed category or another object saved grasp')
    if args.cycles<1 or (args.cycles>1 and not args.through_bin):
        parser.error('cycles>1 requires --through-bin; cycles must be positive')
    if args.network_grasp_replay and not (args.mode=='continuous' and args.planning_mode=='full_preplanned' and args.contact_handoff):
        parser.error('saved network replay requires continuous/full_preplanned/contact-handoff diagnostic')
    if not 0 <= args.belt_speed <= 1 or args.seconds <= 0:
        parser.error('belt-speed must be in [0,1] m/s; seconds must be positive')
    if args.viewer and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        parser.error('--viewer需要桌面显示，请在Ubuntu桌面终端运行（当前DISPLAY为空）')
    print(f'[物理仿真] 开始启动；实时窗口={args.viewer}；视频保存={not args.no_video}',flush=True)
    args.output.mkdir(parents=True,exist_ok=False)
    sources=[*list(Path(__file__).parent.glob('*.py')),
        *list((PROJECT/'mujoco_conveyor_sorting/controllers').glob('*.py')),
        *list((PROJECT/'mujoco_conveyor_sorting/envs').glob('*.py')),
        *list((PROJECT/'mujoco_conveyor_sorting/configs').glob('*.yaml')),
        *list((PROJECT/'experiments/graspgenx_sim').glob('*.py')),PROJECT/'nero_core/sim_common.py']
    before={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(args.output/'source_hashes.json',before)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path,str(path.relative_to(PROJECT)))
    modes=('high','continuous') if args.mode=='both' else (args.mode,)
    results=[]
    for mode in modes:
        result=run_case(args,mode,args.output/mode)
        results.append(result)
        if result['status']=='interrupted':
            break
    after={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(args.output/'comparison.json',{'runs':results,'paired_input_identity':
        len({r.get('initial_qpos_sha256') for r in results})==1,
        'new_network_inference':False,'hardware_execution':False,
        'source_changed_during_run':[p for p in before if before[p]!=after[p]],
        'source_after_sha256':after})
    return 1 if any(x['status'] in ('error','interrupted') for x in results) else 0


if __name__=='__main__':
    raise SystemExit(main())
