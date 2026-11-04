"""SDK binding for the original Nero7 RGB-D detector and belt calibration.

Only measured, eroded green-belt support is passed to the unchanged upstream
multi-frame calibrator. No nominal plane or legacy fallback is substituted.
"""
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.base_plane import (
    EmptyBeltCalibration, plane_context, check_plane_context)


class Nero7FusedPlane:
    def __init__(self, parameters, mode, green_extractor):
        if green_extractor is None:
            raise ValueError('融合测速必须提供原生绿色带面观测来源')
        self.parameters=parameters
        self.mode=mode
        self.green_extractor=green_extractor
        self.plane=self.context=None
        self.last_error=''
        self.reset()

    def reset(self):
        p=self.parameters
        self.collector=EmptyBeltCalibration(self.mode,p['plane_calibration_frames'],
            p['plane_ransac_distance'],p['base_plane_min_ratio'],p['plane_max_tilt_rad'],
            p['plane_max_flat_residual_m'],p['plane_max_height_spread_m'],p['plane_min_xy_span_m'],
            p['plane_calibration_max_points'],p['plane_max_3d_tilt_rad'])
        self.plane=self.context=None
        self.last_error=''

    def observe(self, frame, d2c, camera, color_to_base):
        if self.context is None:
            self.context=plane_context(camera,color_to_base,'robot_base')
        else:
            check_plane_context(self.context,camera,color_to_base,'robot_base')
        if self.plane is not None:return True
        try:
            world,mask,_=self.green_extractor(frame,d2c)
            points=np.asarray(world)[np.asarray(mask,dtype=bool)]
            if not self.collector.add(points,frame.header.source_time):return False
            self.plane=self.collector.fit()
            self.last_error=''
            return True
        except ValueError as error:
            # Start a new measured window; a failed fit never authorizes IDs.
            self.reset()
            self.last_error=str(error)
            return False

    def diagnostics(self):
        return {'ready':self.plane is not None,'mode':self.mode,
            'source':'upstream_EmptyBeltCalibration_measured_green_native_depth',
            'frames':len(self.collector.stamps),'required_frames':self.collector.frames,
            'error':self.last_error,
            'plane':None if self.plane is None else {'normal_base':self.plane.normal.tolist(),
                'offset_m':float(self.plane.offset),'diagnostics':self.plane.diagnostics},
            'height_units':'base_vertical_metres','thresholds_from':'nero7_upstream_perception.yaml'}
