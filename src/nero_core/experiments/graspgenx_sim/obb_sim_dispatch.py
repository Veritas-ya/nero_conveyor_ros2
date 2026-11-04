"""MuJoCo-only application-boundary checks for a visual full-pick command."""
from types import SimpleNamespace
import numpy as np

class SimulationDispatchRejected(RuntimeError):
    pass

class OBBSimulationDispatch:
    def __init__(self, env, context, clock_origin):
        self.env=env; self.context=context; self.origin=float(clock_origin)
        self.count=0; self.previous_width=None; self.terminated=False
        self.last_guard_diagnostics=None

    def prepare(self, action, observation, state, *, application_sim_s):
        """Validate at actuator application using feedback sampled at that instant."""
        ctx=self.context; policy=ctx.policy
        if self.terminated: raise SimulationDispatchRejected('dispatch_terminated')
        if policy.config.planning_mode != 'full_preplanned': raise SimulationDispatchRejected('wrong_controller')
        bundle=policy._full_pick_bundle
        if bundle is None: raise SimulationDispatchRejected('missing_certified_bundle')
        if policy._full_pick_abort_reason: raise SimulationDispatchRejected(policy._full_pick_abort_reason)
        clock=float(application_sim_s)-self.origin
        if not np.isfinite(clock) or clock < float(observation['timestamp'])-1e-9:
            raise SimulationDispatchRejected('invalid_dispatch_clock')
        candidate=dict(action)
        if clock >= bundle.execution_start:
            if clock > bundle.end: raise SimulationDispatchRejected('full_pick_dispatch_past_certificate')
            candidate['arm']=np.asarray(bundle.sample_command_full(clock)[0]).copy()
            candidate['arm_time_parameterized']=True
        close_intent='_dispatch_closure_intent' in candidate
        if close_intent:
            close_status=policy.dispatch_closure_status(candidate,observation)
            if close_status.get('ready'):
                requested_gripper=float(close_status['gripper'])
            else:
                requested_gripper=float(candidate['gripper'])
        else:
            requested_gripper=float(candidate['gripper'])
        controller_action={k:v for k,v in candidate.items() if k != '_dispatch_closure_intent'}
        command=ctx.robot_controller.apply_action(controller_action,dt=self.env.control_dt)
        if candidate.get('arm_time_parameterized') is True and not command.time_parameterized:
            raise SimulationDispatchRejected('controller_retimer_would_change_certified_clock')
        q=np.asarray(command.joint_target,dtype=float)
        reason=policy.full_pick_dispatch_reason(observation,q,clock)
        if reason: raise SimulationDispatchRejected(reason)
        if not ctx.validate_command(state,SimpleNamespace(joint_target=q)):
            raise SimulationDispatchRejected('live_motion_geometry: '+str(ctx.motion_validation_rejection))
        width=.5*(requested_gripper+1.)*ctx.settings.open_width_m
        if not np.isfinite(width) or not 0<=width<=ctx.settings.open_width_m:
            raise SimulationDispatchRejected('invalid_gripper_width')
        if self.previous_width is None or abs(width-self.previous_width)>.0001:
            if not ctx.validate_gripper_command(state,width):
                raise SimulationDispatchRejected('live_gripper_geometry: '+str(ctx.motion_validation_rejection))
            if not ctx.validate_gripper_command({**state,'joints':q},width):
                raise SimulationDispatchRejected('endpoint_gripper_geometry: '+str(ctx.motion_validation_rejection))
        actual=self.env.robot_controller
        limited,*_=actual.preview_joint_step(q,actual.commanded_joint_positions,actual.commanded_joint_velocities,self.env.control_dt)
        difference=np.asarray(limited)-q
        if np.max(np.abs(difference))>1e-7:
            self.last_guard_diagnostics={'reason':'simulation_retimer_would_change_certified_clock',
                'application_sim_s':float(application_sim_s), 'policy_time_s':clock,
                'phase':bundle.phase(clock), 'control_dt_s':float(self.env.control_dt),
                'maximum_joint_difference_rad':float(np.max(np.abs(difference))),
                'joint_difference_rad':difference.tolist(),
                'requested_joint_positions_rad':q.tolist(),
                'limited_joint_positions_rad':np.asarray(limited).tolist(),
                'commanded_joint_positions_rad':np.asarray(actual.commanded_joint_positions).tolist(),
                'commanded_joint_velocities_rad_s':np.asarray(actual.commanded_joint_velocities).tolist()}
            raise SimulationDispatchRejected('simulation_retimer_would_change_certified_clock')
        if close_intent:
            committed=policy.commit_dispatch_closure(candidate,observation)
            if committed:
                ctx.robot_controller.gripper.command(float(candidate['gripper']))
            elif requested_gripper != float(candidate['gripper']):
                raise SimulationDispatchRejected('closure_commit_changed')
        self.previous_width=width;self.count+=1
        return {'arm':q.copy(),'gripper':candidate['gripper'],'arm_time_parameterized':True}

    def stop(self):
        self.terminated=True
        self.env._action_queue.clear()
