"""Use actual deployed acquisition methods while retaining NN IK seed support."""
from mujoco_conveyor_sorting.controllers.visual_acquisition import VisualAcquisitionMixin
from mujoco_conveyor_sorting.controllers.servo_response import VisualFollowingLeadAdapter
from .policy import NetworkSeedSortingPolicy


SHARED_METHODS = (
    '_current_visual_acquisition', '_grasp_target_pose', '_tracking_command_target',
    '_guard_final_tracking_xy', '_cartesian_servo_pose', '_relative_visual_closure_intercept',
)


class Nero7NetworkSeedPolicy(NetworkSeedSortingPolicy):
    """Network seeds remain numerical guesses; shared full checks still run.

    Explicit function aliases prevent historical experiment overrides from
    shadowing the hardware VisualAcquisitionMixin. Python super() in these
    functions resolves after VisualAcquisitionMixin in this inherited MRO.
    """
    _current_visual_acquisition = VisualAcquisitionMixin._current_visual_acquisition
    _grasp_target_pose = VisualAcquisitionMixin._grasp_target_pose
    _tracking_command_target = VisualAcquisitionMixin._tracking_command_target
    _guard_final_tracking_xy = VisualAcquisitionMixin._guard_final_tracking_xy
    _cartesian_servo_pose = VisualAcquisitionMixin._cartesian_servo_pose
    _relative_visual_closure_intercept = VisualAcquisitionMixin._relative_visual_closure_intercept

    def __init__(self, *args, **kwargs):
        self._acquisition_tangent_priority = False
        super().__init__(*args, **kwargs)
        self._following_lead = VisualFollowingLeadAdapter(
            initial_horizon_s=self.config.tracking_velocity_feedforward_time,
            maximum_horizon_s=self.config.tracking_synchronization_horizon,
            filter_time_s=.2)

    def get_info(self):
        info = super().get_info()
        # A small diagnostic projection, without copying the full config on
        # every 100Hz control tick or changing any configured behavior.
        info['nero7_effective_policy_config'] = {
            name: getattr(self.config, name) for name in (
                'dynamic_intercept_tracking', 'control_timestep', 'descent_lead_time',
                'tracking_velocity_feedforward_time', 'closing_tracking_lead_time',
                'closure_contact_lead_time', 'closure_horizontal_tolerance',
                'closure_vertical_tolerance', 'closure_orientation_tolerance')}
        info['nero7_effective_policy_config']['acquisition_tangent_priority'] = self._acquisition_tangent_priority
        return info

    @classmethod
    def shared_method_identity(cls):
        names = SHARED_METHODS+('_cartesian_servo_reference', '_closure_pose_reached',
                              '_refresh_pending_motion_target')
        return {name: getattr(cls,name) is getattr(VisualAcquisitionMixin,name) for name in names}
