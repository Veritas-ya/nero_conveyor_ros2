"""Same-exposure point-cloud Kalman speed alongside unchanged Nero7 IDs."""
import numpy as np
from vendor.nero7_speed_reference.nero7_perception.kalman_motion import KalmanMotion


class PointCloudSpeedTracker:
    def __init__(self, params, belt_direction):
        self.params=dict(params)
        self.belt_direction=np.asarray(belt_direction,float).copy()
        self.tracks={}

    def update(self, tracks, detections, stamp):
        stamp=float(stamp)
        active={int(track.id) for track in tracks}
        self.tracks={key:value for key,value in self.tracks.items() if key in active}
        for track in tracks:
            if track.misses or abs(float(track.stamp)-stamp)>1e-9:
                continue
            matches=[d for d in detections if d.get('speed_position_base') is not None and
                     np.linalg.norm(np.asarray(d['centroid_base'])-np.asarray(track.observed_centroid))<1e-7]
            if len(matches)!=1:
                continue
            point=np.asarray(matches[0]['speed_position_base'],float)
            if point.shape!=(3,) or not np.isfinite(point).all():
                continue
            entry=self.tracks.get(int(track.id))
            if entry is None:
                p=self.params
                keys=('measurement_std','accel_noise_std','cross_accel_noise_std',
                      'initial_velocity_std','innovation_gate','max_velocity_std',
                      'observation_max_age','qualification_gap')
                settings={key:p['kalman_'+key] for key in keys}
                settings.update(window_s=p['speed_window_s'],min_samples=p['speed_min_samples'],
                    min_span=p['speed_min_span'],belt_direction=self.belt_direction,
                    max_velocity=p.get('kalman_max_velocity_m_s',p['tracker_max_vel']))
                motion=KalmanMotion(point[:2],stamp,**settings)
                self.tracks[int(track.id)]={'motion':motion,'point':point.copy(),
                                           'point_count':matches[0].get('speed_position_point_count')}
            elif entry['motion'].update(point[:2],stamp):
                entry.update(point=point.copy(),point_count=matches[0].get('speed_position_point_count'))

    def get(self, track_id):
        return self.tracks.get(int(track_id))
