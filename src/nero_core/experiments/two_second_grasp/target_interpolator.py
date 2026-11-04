"""Optional MuJoCo inner-loop target interpolation, not a factory model."""
import numpy as np


class SimulationTargetInterpolator:
    def __init__(self, env):
        self.env=env
        self.actuators=np.asarray(env.robot_controller.actuator_ids,dtype=int)
        self.output=env.data.ctrl[self.actuators].copy()
        self.target=env.robot_controller._last_joint_target.copy()
        self.start=self.output.copy()
        self.start_s=float(env.data.time)

    def apply(self):
        now=float(self.env.data.time)
        target=self.env.robot_controller._last_joint_target
        if not np.array_equal(target,self.target):
            self.start=self.output.copy()
            self.target=target.copy()
            self.start_s=now
        elapsed=now-self.start_s+float(self.env.model.opt.timestep)
        u=float(np.clip(elapsed/self.env.control_dt,0.,1.))
        self.output=self.start+(self.target-self.start)*u
        self.env.data.ctrl[self.actuators]=self.output

    def report(self):
        return {'model':'linear_interpolation_between_accepted_joint_targets',
                'period_s':float(self.env.control_dt),'hardware_identified':False}
