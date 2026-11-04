"""Request-local observed-surface checks using the field robot convex hulls.

Native MuJoCo still checks the simulated static world/self geometry. Only the
target surface comes from a captured visual cloud; no object CAD pose or live
physics lifecycle enters this validator. A single visible surface is not a
complete solid reconstruction.
"""
import numpy as np
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel,PickRejected


class ObservedTargetGeometry:
    prefer_timed_edges=True

    def __init__(self,points_world,capture_time_s,velocity_world_m_s,base_to_world):
        self.points=np.array(points_world,dtype=float,copy=True)
        self.velocity=np.array(velocity_world_m_s,dtype=float,copy=True)
        self.base_to_world=np.array(base_to_world,dtype=float,copy=True)
        self.capture_time=float(capture_time_s)
        if (self.points.ndim!=2 or self.points.shape[1]!=3 or not len(self.points)
                or self.velocity.shape!=(3,) or self.base_to_world.shape!=(4,4)
                or not np.isfinite(np.r_[self.points.ravel(),self.velocity,
                    self.base_to_world.ravel(),self.capture_time]).all()):
            raise ValueError('finite observed cloud, capture clock, measured velocity and calibration required')
        self.points.setflags(write=False);self.velocity.setflags(write=False)
        self.base_to_world.setflags(write=False)
        self.geometry=RobotModel()
        self.last_rejection={}
        self.calls=0

    def points_at(self,stamp):
        world=self.points+self.velocity*(float(stamp)-self.capture_time)
        return (world-self.base_to_world[:3,3])@self.base_to_world[:3,:3]

    def __call__(self,request,q):
        self.calls+=1;self.last_rejection={}
        if request.full_phase=='lift':
            # Payload geometry requires its own measured grasp transform.
            # This validator certifies the explicit unloaded prefix only.
            self.last_rejection={'reason':'observed_prefix_has_no_payload_certificate'}
            return False
        end=request.full_sample_time_s
        begin=request.full_sample_start_time_s
        if end is None:end=request.planned_target_time_s
        if end is None:
            self.last_rejection={'reason':'observed_target_capture_clock_missing'}
            return False
        begin=end if begin is None else begin
        steps=max(1,int(np.ceil(np.linalg.norm(self.velocity*(end-begin))/.004)))
        width=2.*float(request.full_qpos[self.geometry.model.joint('gripper_joint1').qposadr[0]])
        # Same semantics as NeroPolicyContext._allow_target_finger_contact
        # and TargetGeometry: selected workpiece contact with the fingers is
        # permitted. Static-world finger collisions remain native checks.
        allow_fingers=True
        for stamp in np.linspace(begin,end,steps+1):
            self.geometry.target=self.points_at(stamp)
            try:
                self.geometry.check_state(q,width,allow_finger_contact=allow_fingers,check_scene=False)
            except PickRejected as error:
                self.last_rejection=dict(reason='observed_target_geometry',message=str(error),
                    capture_time_s=self.capture_time,sample_time_s=float(stamp),width_m=width,
                    points=len(self.points),finger_contact_allowed=allow_fingers)
                return False
        return True
