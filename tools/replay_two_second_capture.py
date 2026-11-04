"""Replay real saved RGB-D through the measured deployment input path, no devices."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from mujoco_conveyor_sorting.runtime.types import RGBDFrame, CameraIntrinsics, SampleHeader
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel, detect_target
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from nero_core.two_second.measured_rgbd import fit_plane, green_belt_points, depth_capture_timing
from nero_core.two_second.grasp import visible_belt_aligned_obb
from nero_core.two_second.main import json_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame-stem', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stem = args.frame_stem
    files = [Path(str(stem)+suffix) for suffix in ('.json', '_depth_m.npy', '_rgb.png')]
    meta = json.loads(files[0].read_text())
    depth = np.load(files[1], allow_pickle=False)
    rgb = cv2.cvtColor(cv2.imread(str(files[2])), cv2.COLOR_BGR2RGB)
    frame = RGBDFrame(header=SampleHeader(**meta['header']), depth_m=depth, rgb=rgb,
        depth_intrinsics=CameraIntrinsics(**meta['depth_intrinsics']),
        color_intrinsics=CameraIntrinsics(**meta['color_intrinsics']),
        camera_to_world_cv=np.array(meta['depth_to_base_cv']),
        serial_number=meta['serial_number'], calibration_id=meta['calibration_id'],
        source='saved_physical_rgbd', rgb_aligned_to_depth=False,
        timestamp_metadata=meta['timestamp_metadata'])
    d2c = np.array(meta['depth_to_color_cv'])
    result = {'mode': 'saved_physical_capture_offline', 'hardware_connections': 0, 'commands_sent': 0,
        'source_hashes': {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        'calibration_id': meta['calibration_id'], 'physical_grasp_verified': False}
    start = time.perf_counter()
    try:
        result['timing'] = depth_capture_timing(frame)
        model = RobotModel()
        model.check_feedback(meta['robot_state'])
        result['model_startup_wall_s'] = time.perf_counter()-start
        stage = time.perf_counter()
        detection = detect_target(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
            target_shape='simple',robot_model=model,robot_state=meta['robot_state'])
        result['detection_wall_s'] = time.perf_counter()-stage
        plane = detection['belt_region']['plane_to_base']
        local = (detection['object_points']-plane[:3,3])@plane[:3,:3]
        box = visible_belt_aligned_obb(local)
        result['visible_obb'] = box
        result['visible_obb_center_base_m'] = plane[:3,:3]@np.array(box['center_world_m'])+plane[:3,3]
        stage = time.perf_counter()
        model.install_scene(detection['scene_points'],detection['object_points'],meta['robot_state'],
            camera_origin=detection['camera_origin_base_m'],belt_region=detection['belt_region'],
            depth_frame=frame,target_mask=detection['mask'],scene_pixel_indices=detection['scene_pixel_indices'])
        result['scene_prepare_wall_s'] = time.perf_counter()-stage
        meter = Nero7BeltVelocityMeter(model.measured_belt_surface,maximum_measured_speed_m_s=.30)
        meter.update(frame,d2c)
        result.update(status='measured_geometry_and_scene_replayed',object_points=len(detection['object_points']),
            retained_scene_points=len(model.scene), qualified_speed_tracks=sum(t.speed_valid for t in meter.latest_snapshot.tracks),
            velocity_note='单个保存曝光不能提供多帧卡尔曼测速，不补固定带速',
            tcp_note='未提供现场TCP标定，未生成机器人抓姿或运动轨迹')
    except Exception as error:
        result.update(status='replay_rejected',error=str(error),details=getattr(error,'details',{}))
    result['total_wall_s'] = time.perf_counter()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(json_value(result),ensure_ascii=False,indent=2))
    print(json.dumps(json_value(result),ensure_ascii=False))
    return 0 if result['status']=='measured_geometry_and_scene_replayed' else 1


if __name__=='__main__':
    raise SystemExit(main())
