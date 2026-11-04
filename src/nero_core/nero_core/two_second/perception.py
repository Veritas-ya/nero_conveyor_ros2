"""One production/preview binding to the retained Nero7 perception chain."""
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from .measured_rgbd import green_belt_points


def make_meter(config, surface, *, previous=None):
    mode=config.raw.get('perception_detection_mode','rgb_depth')
    follow=getattr(config,'follow_bounds',None)
    meter=Nero7BeltVelocityMeter(surface,belt_direction_base=config.direction,
        maximum_measured_speed_m_s=config.raw['maximum_measured_speed_m_s'],
        detection_mode=mode,plane_mode=config.raw.get('perception_plane_mode','ransac3d'),
        green_belt_extractor=green_belt_points,
        tracking_entry_coordinate_m=None if follow is None else float(follow[0]),
        pointcloud_speed=config.raw.get('velocity_source')=='pointcloud_xy_kalman')
    meter.parameters['isolate_boundary_depth_gaps'] = config.raw.get('isolate_boundary_depth_gaps', False)
    if mode!='legacy_depth':
        # Only the upstream base-height modes interpret this as vertical metres.
        meter.parameters['block_height_max']=config.raw['maximum_target_height_m']
    from mujoco_conveyor_sorting.runtime.workpiece_admission import MeasuredWorkpieceAdmission
    meter.candidate_filter=MeasuredWorkpieceAdmission(surface,config.raw['maximum_target_height_m'],
        include_speed_position=meter.pointcloud_speed).filter
    if previous is not None:
        # A changed calibration can rebuild the adapter without changing the
        # device epoch. Its IDs must still differ from already consumed tasks.
        meter._generation=previous._generation+1
        meter._next_id=(previous._tracker._next_id if previous._tracker is not None else previous._next_id)
    return meter
