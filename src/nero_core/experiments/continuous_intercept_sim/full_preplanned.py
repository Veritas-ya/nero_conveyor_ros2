"""Full-pick adapters for the existing diagnostic MuJoCo contact harness.

Known simulator meshes and delayed simulator-history poses are used here;
this is not a camera/GraspGen-X or native-point-cloud collision experiment.
All hypothetical object placement is confined to private collision MjData.
"""
from dataclasses import replace
from copy import deepcopy
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK
from mujoco_conveyor_sorting.controllers.planning_worker import AsyncPlanningWorker, MuJoCoGlobalPlanningEngine
from mujoco_conveyor_sorting.controllers._math import quat_distance
from mujoco_conveyor_sorting.controllers.dynamic_intercept import intercept_lead


def matrix(pose):
    pose=np.asarray(pose)
    value=np.eye(4);value[:3,3]=pose[:3]
    value[:3,:3]=Rotation.from_quat(pose[3:][[1,2,3,0]]).as_matrix()
    return value


def pose(value):
    return np.r_[value[:3,3],Rotation.from_matrix(value[:3,:3]).as_quat()[[3,0,1,2]]]


def completed_lift_hold(policy):
    """The contact experiment ends at lift, then holds for independent scoring."""
    q=getattr(policy,'_full_pick_completed_hold_q',None)
    if q is None:q=policy._full_pick_bundle.lift.q1
    return dict(arm=q.copy(),
        gripper=policy._grasp_gripper_action,_full_pick_stop_reason='completed_lift_holding')


def finish_contact_lift_experiment(policy, mirror):
    """Stop after the ORIGINAL measured clearance condition, before transport.

    The policy sets clearance_complete only after its unchanged height, XY,
    orientation and measured joint-velocity tests. Independent physical hold
    scoring still happens later; this is not itself a grasp success flag.
    """
    if (not policy._full_pick_contact_handed_off or policy._full_pick_abort_reason
            or policy.stage.value not in {'lifting','transferring'}):
        return False
    if not (policy._lift_clearance_complete or policy.stage.value=='transferring'):
        return False
    policy._full_pick_completed_hold_q=mirror.robot_controller.commanded_joint_positions.copy()
    policy._full_pick_finished=True
    policy._full_pick_note('single_pick_lift_scope_completed',float(mirror.data.time),
        confirmation='original_shared_measured_clearance_or_lift_reached',
        independent_retention_score_pending=True,bin_transfer_not_sent=True)
    return True


def tcp_prediction(prediction, tcp_current):
    """Express the SAME exposure using the catalog TCP offset, never restamp."""
    result=dict(prediction)
    current=np.asarray(prediction['current_estimated_pose'])
    offset=np.linalg.inv(matrix(current))@matrix(tcp_current)
    result['object_observed_pose']=np.asarray(prediction.get('source_object_observed_pose',prediction['observed_pose'])).copy()
    result['object_current_estimated_pose']=np.asarray(prediction.get('source_object_current_estimated_pose',current)).copy()
    result['observed_pose']=pose(matrix(prediction['observed_pose'])@offset)
    result['current_ee_grasp_pose']=np.asarray(tcp_current).copy()
    result['predicted_ee_grasp_pose']=pose(matrix(prediction['predicted_grasp_pose'])@offset)
    result['tcp_to_object']=np.linalg.inv(matrix(tcp_current))@matrix(result['object_current_estimated_pose'])
    return result


def object_offset_at_capture(prediction,request):
    """Keep the measured object fixed when a different grasp yaw is proposed."""
    obj=np.array(prediction['object_observed_pose'],copy=True)
    obj[:3]+=np.asarray(request.continuation_velocity_m_s)*(
        request.planned_target_time_s-prediction['observed_timestamp'])
    return np.linalg.inv(matrix(request.target_pose))@matrix(obj)


class TargetGeometry:
    """Immutable-request target sweeps and conditional payload on known meshes."""
    # Use the already implemented exact-time edge fallback directly. A whole
    # 3 s sweep at every robot pose creates thousands of unrelated pose/time
    # combinations; per-edge sweeps keep all original curve samples/checks.
    prefer_timed_edges=True
    def __init__(self, env):
        self.model=deepcopy(env.model);self.data=mujoco.MjData(self.model)
        self.arm=np.asarray(env.arm_qpos_adrs)
        self.site=int(env.ee_site_id)
        self.robot=set(env._robot_collision_body_ids)
        # Entire moving fingers may contact the selected workpiece.
        self.fingers={mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,n)
                      for n in ('gripper_link1','gripper_link2')}
        self.targets={r.logical_id:(int(r.qpos_adr),set(r.geom_ids))
                      for r in env._objects if r.logical_id is not None}
        self.target_geom_masks={identifier:np.isin(np.arange(self.model.ngeom),tuple(geoms))
            for identifier,(_,geoms) in self.targets.items()}
        self.belt=int(env.belt_geom_id)
        self.last_rejection={}
        self.loaded_requests={}
        self.static_requests={}

    def register_targets(self,env):
        """New logical ids reuse original physical slots, never cached results."""
        for runtime in env._objects:
            identifier=runtime.logical_id
            if identifier is not None and identifier not in self.targets:
                spec=(int(runtime.qpos_adr),set(runtime.geom_ids))
                for geom in runtime.geom_ids:
                    self.model.geom_contype[geom]=env.model.geom_contype[geom]
                    self.model.geom_conaffinity[geom]=env.model.geom_conaffinity[geom]
                self.targets[identifier]=spec
                self.target_geom_masks[identifier]=np.isin(np.arange(self.model.ngeom),tuple(spec[1]))

    @staticmethod
    def request_key(request):
        return (request.generation,request.request_id,request.scene_epoch,request.target_id)

    def bind_loaded_request(self,request):
        """Freeze this diagnostic's measured held-body transform, per request."""
        address,_=self.targets[request.target_id]
        self.data.qpos[:]=request.full_qpos
        self.data.qpos[self.arm]=request.start_q
        self._update_geometry()
        tcp=np.eye(4);tcp[:3,3]=self.data.site_xpos[self.site]
        tcp[:3,:3]=self.data.site_xmat[self.site].reshape(3,3)
        body=matrix(request.full_qpos[address:address+7])
        offset=np.linalg.inv(tcp)@body
        self.loaded_requests={self.request_key(request):offset.copy()}
        return dict(request_id=request.request_id,source='current_simulated_joint_and_body_snapshot_after_physical_grasp_confirmation',
            object_pose=pose(body).tolist(),tcp_pose=pose(tcp).tolist(),tcp_to_object=offset.tolist())

    def bind_static_request(self,request):
        address,_=self.targets[request.target_id]
        body=np.array(request.full_qpos[address:address+7],copy=True)
        self.static_requests={self.request_key(request):body}
        return dict(request_id=request.request_id,source='released_terminal_object_in_request_snapshot',
            object_pose=body.tolist())

    def _update_geometry(self):
        # Private collision MjData only: no actuation, acceleration or constraint
        # solve is consumed here. Keep native rigid-body transforms/contact
        # generation; the physical environment still calls its full dynamics.
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_collision(self.model,self.data)

    def __call__(self, request, q):
        self.last_rejection={}
        spec=self.targets.get(request.target_id)
        if spec is None:
            self.last_rejection={'reason':'unknown_simulated_target'};return False
        loaded_offset=self.loaded_requests.get(self.request_key(request))
        static_pose=self.static_requests.get(self.request_key(request))
        loaded=request.full_phase=='lift' or loaded_offset is not None
        if request.planned_target_time_s is None and not loaded and static_pose is None:
            self.last_rejection={'reason':'missing_simulated_capture_time'};return False
        address,geoms=spec
        full=np.asarray(request.full_qpos).copy();full[self.arm]=q
        # The request snapshot carries the observed object-to-TCP transform.
        # snapshot's object pose is expressed at request observation time;
        # its selected TCP is stored independently by the mirror's snapshot.
        offset=(self.offsets[request.target_id] if loaded_offset is None else loaded_offset) if static_pose is None else None
        begin=request.full_sample_start_time_s
        end=request.full_sample_time_s
        if begin is None:begin=end=request.planned_target_time_s
        if begin is None:begin=end=request.intercept_request_time_s or 0.
        end=begin if end is None else end
        velocity=np.zeros(3) if loaded or static_pose is not None else np.asarray(request.continuation_velocity_m_s)
        steps=max(1,int(np.ceil(np.linalg.norm(velocity*(end-begin))/.004)))
        times=np.linspace(begin,end,steps+1)
        self.data.qpos[:]=full
        self._update_geometry()
        tcp=np.eye(4);tcp[:3,3]=self.data.site_xpos[self.site]
        tcp[:3,:3]=self.data.site_xmat[self.site].reshape(3,3)
        if not loaded and static_pose is None:
            contact_pose=(request.continuation_target_poses[0]
                if request.planning_mode!='full_preplanned' and request.continuation_target_poses
                else request.target_pose)
            base=matrix(contact_pose)@offset
            base_pose=pose(base)
        for stamp in times:
            if loaded:
                obj=tcp@offset
            elif static_pose is not None:
                obj=matrix(static_pose)
            else:
                obj=base.copy()
                obj[:3,3]+=velocity*(stamp-request.planned_target_time_s)
            obj_pose=pose(obj) if loaded or static_pose is not None else base_pose.copy()
            if not loaded and static_pose is None:
                obj_pose[:3]+=velocity*(stamp-request.planned_target_time_s)
            self.data.qpos[address:address+7]=obj_pose
            self._update_geometry()
            contacts=self.data.contact
            count=self.data.ncon
            flags=self.target_geom_masks[request.target_id]
            relevant=np.flatnonzero((contacts.dist[:count]<=1e-6)
                & (flags[contacts.geom1[:count]]|flags[contacts.geom2[:count]]))
            for index in relevant:
                c=contacts[int(index)]
                a,b=int(c.geom1),int(c.geom2)
                if float(c.dist)>1e-6 or not ({a,b}&geoms):continue
                other=b if a in geoms else a
                if other in geoms:continue
                body=int(self.model.geom_bodyid[other])
                if body in self.fingers:continue
                if body in self.robot:
                    self.last_rejection={'reason':'simulated_target_robot_contact',
                        'geom':self.model.geom(other).name,'phase':request.full_phase}
                    return False
                if loaded:
                    # Only the original support contact may persist at lift
                    # start. An erroneous TCP offset must not move that plane.
                    support_z=float(request.full_qpos[address+2])
                    supported=(other==self.belt and obj[2,3]>=support_z-request.intercept_position_tolerance_m)
                    if not supported:
                        self.last_rejection={'reason':'simulated_payload_world_contact',
                            'geom':self.model.geom(other).name,'phase':'lift'}
                        return False
        return True


class FullPickMirrorMixin:
    def _dynamic_intercept_prediction_lead(self, position, velocity, ee_position, **timing):
        policy=getattr(self,'policy',None)
        config=getattr(policy,'config',None)
        if (config is None or config.planning_mode!='full_preplanned'
                or config.moving_approach_mode!='continuous'):
            return super()._dynamic_intercept_prediction_lead(position,velocity,ee_position,**timing)
        # Match the actual full direct request, including the original catalog
        # TCP height. This changes when diagnostic observations become eligible
        # for planning, not the physical workspace or collision acceptance.
        reserve=config.full_plan_compute_budget_s+config.trajectory_minimum_duration+self.control_dt
        station=getattr(self,'contact_station_x_m',None)
        if station is not None and abs(float(velocity[0]))>1e-6:
            station_lead=(float(station)-float(position[0]))/float(velocity[0])
            target=np.asarray(position)+np.asarray(velocity)*station_lead+np.array([0.,0.,timing['grasp_height']])
            motion=max(config.descent_lead_time,float(np.linalg.norm(target-ee_position))/config.cartesian_linear_speed)
            return min(timing['maximum_horizon'],max(reserve+motion,station_lead))
        return intercept_lead(position,velocity,ee_position,
            grasp_height=timing['grasp_height'],approach_clearance=0.,
            cartesian_speed=config.cartesian_linear_speed,
            minimum_motion_time=config.descent_lead_time,latency=reserve,
            maximum_horizon=max(timing['maximum_horizon'],reserve+config.descent_lead_time))

    def adopt_snapshot(self,snap):
        obs=super().adopt_snapshot(snap)
        decorate=getattr(self,'adapt_grasp_observation',None)
        if callable(decorate):obs=decorate(obs,snap)
        for target in obs['objects']:
            identifier=target['object_id'];prediction=obs['predictions'].get(identifier)
            if prediction is None or 'current_ee_grasp_pose' not in prediction:continue
            chosen=getattr(self.policy,'_full_pick_catalog_quaternion',None)
            if chosen is not None:
                prediction['current_ee_grasp_pose']=np.asarray(prediction['current_ee_grasp_pose']).copy()
                prediction['current_ee_grasp_pose'][3:]=chosen
            adapted=tcp_prediction(prediction,prediction['current_ee_grasp_pose'])
            obs['predictions'][identifier]=adapted
            target['prediction']=adapted
            target['predicted_ee_grasp_pose']=adapted['predicted_ee_grasp_pose']
            self.full_plan_grasp_width_m=float(target['recommended_grip_width'])
        obs['dispatch_clock']['scheduled_dispatch_policy_time_s']=float(snap['sim_s'])+self.control_latency
        previous=snap.get('last_actuator_application_sim_s')
        if previous is not None:
            obs['control_dt']=max(self.control_dt,
                obs['dispatch_clock']['scheduled_dispatch_policy_time_s']-float(previous))
        obs['belt_velocity']=np.asarray(next(iter(obs['predictions'].values()),{}).get('estimated_velocity',np.zeros(3)))
        self._comparison_observation=obs
        return obs

    def create_global_planning_worker(self):
        validator=TargetGeometry(self)
        # Shape offsets are frozen per worker job by submit, not read from live
        # policy dictionaries on its background thread.
        engine=MuJoCoGlobalPlanningEngine(self.model,joint_names=self.arm_joint_names,
            ee_site_name=self.ee_site_name,ik_config=self.robot_config.get('ik',{}),
            robot_collision_body_ids=tuple(self._robot_collision_body_ids),state_validator=validator)
        owner=self
        advance = float(getattr(owner, 'simulation_command_advance_s', 0.))
        if advance:
            from experiments.continuous_intercept_sim.command_advance import install_simulation_advance
            install_simulation_advance(engine, advance)
        owner.full_pick_worker_events=[]
        owner.full_pick_payload_bindings=[]
        original_plan=engine.plan
        def observed_plan(request,cancelled):
            clock=getattr(owner,'full_pick_physics_clock',lambda:float(owner.data.time))
            began=time.perf_counter();cpu=time.thread_time();sim=clock();began_mono=time.monotonic()
            try:
                if request.full_parent_version is not None:
                    delay=float(getattr(owner,'full_pick_replan_worker_delay_s',0.))
                    if delay:time.sleep(delay)
                return original_plan(request,cancelled)
            finally:
                owner.full_pick_worker_events.append(dict(request_id=request.request_id,
                    started_sim_s=sim,ended_sim_s=clock(),
                    started_monotonic_s=began_mono,ended_monotonic_s=time.monotonic(),
                    parent_version=request.full_parent_version,
                    wall_s=time.perf_counter()-began,thread_cpu_s=time.thread_time()-cpu,
                    cancelled=cancelled(),diagnostics=deepcopy(engine._request_diagnostics)))
        engine.plan=observed_plan
        class Worker(AsyncPlanningWorker):
            def submit(self,request):
                # A single-pick harness submits no replacement while busy.
                if self.busy:return False
                validator.register_targets(owner)
                predictions=owner._comparison_observation['predictions']
                validator.offsets={key:np.array(p['tcp_to_object'],copy=True)
                    for key,p in predictions.items() if 'tcp_to_object' in p}
                selected=predictions.get(request.target_id)
                if (selected is not None and 'object_observed_pose' in selected
                        and request.planned_target_time_s is not None):
                    validator.offsets[request.target_id]=object_offset_at_capture(selected,request)
                owner.full_pick_offsets={k:v.copy() for k,v in validator.offsets.items()}
                if (request.stage in {'lifting','transferring','releasing'}
                        and owner._physical_grasp_confirmations.get(request.target_id,False)):
                    owner.full_pick_payload_bindings.append(validator.bind_loaded_request(request))
                elif (request.stage=='returning' and (request.target_id not in owner._physical_object_state_by_id
                        or owner._physical_object_state_by_id.get(request.target_id)
                        in {'SORTED','FAILED','MISSED','SKIPPED'})):
                    owner.full_pick_payload_bindings.append(validator.bind_static_request(request))
                accepted=super().submit(request)
                if accepted:owner.full_pick_submissions.append(request)
                return accepted
            def poll(self):
                result=super().poll()
                if result is not None:owner.full_pick_results.append(result)
                return result
        self.full_pick_submissions=[];self.full_pick_results=[]
        worker=Worker(engine);self._planning_workers.append(worker)
        self.ik=DampedLeastSquaresIK(self.model,mujoco.MjData(self.model),self.arm_joint_names,
                                    self.ee_site_name,self.robot_config.get('ik',{}))
        return worker


class FullPickPhysicalMixin:
    """Resample at actual actuator application; retain original controller limits."""
    def prepare_full_pick_geometry(self):
        # The same private checker was formerly first constructed inside the
        # physics application callback. Build it before the wall-clock trial.
        if not hasattr(self,'_full_pick_live_target'):
            self._full_pick_live_target=TargetGeometry(self)
        return self._full_pick_live_target

    def _queue_action(self,action):
        if action is not None and action.get('_full_pick_retire_bundle'):
            self._action_queue.clear()
        if action is not None and action.get('_full_pick_stop_reason'):
            # Invalidate every queued certificate on task failure. The fixed
            # hold is captured once; subsequent feedback cannot drag it along.
            self.full_pick_stop_reason=action['_full_pick_stop_reason']
            self._action_queue.clear()
            if not hasattr(self,'_full_pick_fault_hold'):
                self._full_pick_fault_hold=np.asarray(action['arm']).copy()
                self._full_pick_fault_gripper=action.get('gripper',1.)
        super()._queue_action(action)
        if action is not None and self._action_queue:
            normalized=self._action_queue[-1][1]
            for key in ('_full_pick_bundle','_full_pick_lift_admitted','_full_pick_config','_full_pick_offsets',
                        '_full_pick_shared_config','_full_pick_prepared_at_sim_s','_full_pick_sample_sim_s',
                        '_full_pick_policy_finished_monotonic_s'):
                if key in action:normalized[key]=action[key]

    def _apply_due_actions(self):
        due=[]
        while self._action_queue and self._action_queue[0][0]<=self.data.time+1e-12:
            due.append(self._action_queue.popleft()[1])
        if not due:return
        action=due[-1];bundle=action.get('_full_pick_bundle')
        now=float(self.data.time)
        previous=getattr(self,'last_actuator_application_sim_s',None)
        dispatch_dt=self.control_dt if previous is None else now-float(previous)
        if dispatch_dt<=0.:raise RuntimeError('non-increasing physical actuator application clock')
        fault=getattr(self,'full_pick_dispatch_fault',None) or getattr(self,'full_pick_stop_reason',None)
        if fault:
            action=dict(arm=self._full_pick_fault_hold.copy(),gripper=self._full_pick_fault_gripper)
        elif bundle is not None and self.data.time>=getattr(bundle,'execution_start',bundle.start):
            config=action['_full_pick_config']
            q,v,a=bundle.sample_command_full(now)
            current=self.data.qpos[self.arm_qpos_adrs].copy()
            previous=self.robot_controller.commanded_joint_positions
            prior_v=self.robot_controller.commanded_joint_velocities
            limited,*_=self.robot_controller.preview_joint_step(q,previous,prior_v,dispatch_dt)
            reason=None
            if now>bundle.end:reason='full_pick_dispatch_past_certificate'
            elif now>=bundle.lift_time and not action['_full_pick_lift_admitted']:
                reason='loaded_lift_not_admitted_before_dispatch'
            elif np.max(np.abs(limited-q))>1e-7:reason='controller_retimer_would_change_certified_clock'
            else:
                scratch=mujoco.MjData(self.model);scratch.qpos[:]=self.data.qpos
                scratch.qpos[self.arm_qpos_adrs]=q;mujoco.mj_forward(self.model,scratch)
                actual=self._ee_pose();planned=np.r_[scratch.site_xpos[self.ee_site_id],np.zeros(4)]
                mujoco.mju_mat2Quat(planned[3:],scratch.site_xmat[self.ee_site_id])
                envelope=config.acquisition_path_tracking_error_limit if now<bundle.approach_end_time else config.max_ik_joint_step
                if (np.max(np.abs(q-current))>envelope
                    or np.linalg.norm(planned[:3]-actual[:3])>config.max_translation_step
                    or quat_distance(planned[3:],actual[3:])>config.max_rotation_step
                    or not self.is_arm_edge_safe(current,q,
                        normalized_resolution=config.planner_edge_resolution,
                        approach_axis_local=config.approach_axis_local,
                        minimum_downward_alignment=bundle.request.minimum_downward_alignment,
                        minimum_tcp_height=bundle.minimum_tcp_height_at(now),
                        minimum_joint_margin=bundle.request.minimum_joint_margin,
                        maximum_jacobian_condition=config.singularity_hard_stop_condition)):
                    reason='full_pick_feedback_or_live_geometry_invalid'
                if reason is None:
                    if not hasattr(self,'_full_pick_live_target'):
                        self.prepare_full_pick_geometry()
                    validator=self._full_pick_live_target
                    validator.register_targets(self)
                    validator.offsets=action['_full_pick_offsets']
                    segment=bundle.segment_at(now) if callable(getattr(bundle,'segment_at',None)) else bundle
                    req=replace(segment.request,full_phase=segment.phase(now),
                        full_sample_start_time_s=now,full_sample_time_s=now)
                    steps=max(1,int(np.ceil(np.max(np.abs(q-current))/.008)))
                    if not all(validator(req,current+(q-current)*u) for u in np.linspace(0.,1.,steps+1)):
                        reason='full_pick_live_target_or_payload_invalid'
            if not hasattr(self,'full_pick_dispatch_rows'):self.full_pick_dispatch_rows=[]
            segment=bundle.segment_at(now) if callable(getattr(bundle,'segment_at',None)) else bundle
            def version_of(value):
                req=getattr(value,'request',None)
                return None if req is None else [req.generation,req.request_id]
            self.full_pick_dispatch_rows.append(dict(application_sim_s=now,reference_q=q.copy(),
                application_monotonic_s=time.monotonic(),
                prepared_feedback_sim_s=action.get('_full_pick_prepared_at_sim_s'),
                prepared_sample_sim_s=action.get('_full_pick_sample_sim_s'),
                prepared_finished_monotonic_s=action.get('_full_pick_policy_finished_monotonic_s'),
                trajectory_version=version_of(bundle),sampled_segment_version=version_of(segment),
                scheduled_splice_sim_s=bundle.start,parent_version=getattr(getattr(bundle,'request',None),'full_parent_version',None),
                reference_v=v.copy(),reference_a=a.copy(),actual_q=current,
                nominal_reference_q=bundle.sample_full(now)[0],
                same_time_feedback_translation_error_m=float(np.linalg.norm(
                    self.ik.forward_pose(bundle.sample_full(now)[0])[:3]-self._ee_pose()[:3])) if hasattr(self,'ik') else None,
                response_compensation_applied=bundle.command_curves is not None,
                application_dt_s=dispatch_dt,
                prepared_q=np.asarray(action['arm']).copy(),candidate_sent=reason is None,reason=reason))
            if reason:
                self.full_pick_dispatch_fault=reason;self._full_pick_fault_hold=current
                self._full_pick_fault_gripper=action.get('gripper',1.)
                action=dict(arm=current,gripper=self._full_pick_fault_gripper)
            else:action=dict(action,arm=q,arm_time_parameterized=True)
        if not fault and action.get('_full_pick_shared_config') is not None:
            config=action['_full_pick_shared_config']
            current=self.data.qpos[self.arm_qpos_adrs].copy()
            q,*_=self.robot_controller.preview_joint_step(np.asarray(action['arm']),
                self.robot_controller.commanded_joint_positions,
                self.robot_controller.commanded_joint_velocities,dispatch_dt)
            scratch=mujoco.MjData(self.model);scratch.qpos[:]=self.data.qpos
            scratch.qpos[self.arm_qpos_adrs]=q;mujoco.mj_forward(self.model,scratch)
            planned=np.r_[scratch.site_xpos[self.ee_site_id],np.zeros(4)]
            mujoco.mju_mat2Quat(planned[3:],scratch.site_xmat[self.ee_site_id])
            actual=self._ee_pose()
            distance=float(np.linalg.norm(planned[:3]-actual[:3]))
            rotation=float(quat_distance(planned[3:],actual[3:]))
            joint_error=float(np.max(np.abs(q-current)))
            if not hasattr(self,'full_pick_dispatch_rows'):self.full_pick_dispatch_rows=[]
            self.full_pick_dispatch_rows.append(dict(application_sim_s=now,
                phase='shared_feedback_after_contact_handoff',prepared_q=np.asarray(action['arm']).copy(),
                reference_q=q.copy(),actual_q=current.copy(),
                previous_command_q=self.robot_controller.commanded_joint_positions.copy(),
                previous_command_velocity=self.robot_controller.commanded_joint_velocities.copy(),
                application_dt_s=dispatch_dt,prepared_tcp_pose=planned.copy(),actual_tcp_pose=actual.copy(),
                feedback_translation_error_m=distance,feedback_rotation_error_rad=rotation,
                feedback_joint_error_rad=joint_error,
                feedback_translation_limit_m=config.max_translation_step,
                feedback_rotation_limit_rad=config.max_rotation_step,
                feedback_joint_limit_rad=config.max_ik_joint_step,
                arm_time_parameterized=bool(action.get('arm_time_parameterized',False)),
                candidate_sent=distance<=config.max_translation_step and rotation<=config.max_rotation_step
                    and joint_error<=config.max_ik_joint_step))
            if (np.linalg.norm(planned[:3]-actual[:3])>config.max_translation_step
                    or quat_distance(planned[3:],actual[3:])>config.max_rotation_step
                    or np.max(np.abs(q-current))>config.max_ik_joint_step):
                self.full_pick_dispatch_fault='contact_handoff_feedback_envelope_invalid'
                self._full_pick_fault_hold=current
                self._full_pick_fault_gripper=action.get('gripper',1.)
                action=dict(arm=current,gripper=self._full_pick_fault_gripper)
        self.robot_controller.apply_action(action,action_mode=self.action_type,dt=dispatch_dt)
        self.last_actuator_application_sim_s=now
        if (getattr(self,'full_pick_dispatch_rows',None)
                and self.full_pick_dispatch_rows[-1]['application_sim_s']==now):
            self.full_pick_dispatch_rows[-1]['dispatched_q']=self.robot_controller.commanded_joint_positions.copy()
            self.full_pick_dispatch_rows[-1]['dispatched_velocity']=self.robot_controller.commanded_joint_velocities.copy()
        if action.get('gripper') is not None:self._last_gripper_command=float(action['gripper'])
