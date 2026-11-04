"""MuJoCo substep acceleration and contact diagnostics, simulation only."""
import numpy as np
import mujoco


class PhysicsAccelerationTrace:
    def __init__(self, env, phase, window=None):
        self.env, self.phase = env, phase
        self.window = window
        self.window_samples = []
        self.count = 0
        self.peaks = {'velocity_difference': [None]*7, 'post_step_qacc': [None]*7}
        self.phase_peaks = {}

    def __call__(self, model, data, before_qvel):
        acceleration = (data.qvel[self.env.arm_dof_adrs]-before_qvel)/model.opt.timestep
        qacc = data.qacc[self.env.arm_dof_adrs].copy()
        self.count += 1
        phase = self.phase()
        if self.window is not None and self.window[0] <= data.time <= self.window[1]:
            self.window_samples.append({'sim_s':float(data.time),'phase':phase,
                'qvel_before':before_qvel.copy(),'qvel_after':data.qvel[self.env.arm_dof_adrs].copy(),
                'velocity_difference':acceleration.copy(),'post_step_qacc':qacc.copy(),
                'actuator_force':data.qfrc_actuator[self.env.arm_dof_adrs].copy(),
                'constraint_force':data.qfrc_constraint[self.env.arm_dof_adrs].copy()})
        phase_peaks = self.phase_peaks.setdefault(phase, {'velocity_difference':[None]*7,'post_step_qacc':[None]*7})
        selected = [(name, i, float(value)) for name,values in
            (('velocity_difference',acceleration),('post_step_qacc',qacc))
            for i,value in enumerate(values)
            if (self.peaks[name][i] is None or abs(value)>abs(self.peaks[name][i]['value_rad_s2'])
                or phase_peaks[name][i] is None or abs(value)>abs(phase_peaks[name][i]['value_rad_s2']))]
        if not selected:
            return
        contacts=[]
        for i in range(data.ncon):
            contact=data.contact[i]
            if contact.dist > 1e-6:
                continue
            force=np.zeros(6)
            mujoco.mj_contactForce(model,data,i,force)
            if force[0] <= 0:
                continue
            contacts.append({'geom1':model.geom(int(contact.geom1)).name,
                'geom2':model.geom(int(contact.geom2)).name,'normal_force_n':float(force[0])})
        for name,i,value in selected:
            dof=self.env.arm_dof_adrs[i]
            item={'value_rad_s2':value,'sim_s':float(data.time),
                'phase':phase,'joint':i+1,'contacts':contacts,
                'qvel_before_rad_s':float(before_qvel[i]),'qvel_after_rad_s':float(data.qvel[dof]),
                'joint_actuator_force':float(data.qfrc_actuator[dof]),
                'joint_constraint_force':float(data.qfrc_constraint[dof]),
                'joint_bias_force':float(data.qfrc_bias[dof])}
            if self.peaks[name][i] is None or abs(value)>abs(self.peaks[name][i]['value_rad_s2']):
                self.peaks[name][i]=item
            if phase_peaks[name][i] is None or abs(value)>abs(phase_peaks[name][i]['value_rad_s2']):
                phase_peaks[name][i]=item

    def report(self):
        return {'physics_timestep_s':float(self.env.model.opt.timestep),
            'substeps':self.count,'methods':{
                'velocity_difference':'(qvel_after - qvel_before) / physics_timestep',
                'post_step_qacc':'MuJoCo qacc immediately after mj_step, before viewer updates'},
            'per_joint_absolute_peaks':self.peaks,'per_phase_joint_absolute_peaks':self.phase_peaks,
            'requested_window_s':self.window,'window_samples':self.window_samples}
