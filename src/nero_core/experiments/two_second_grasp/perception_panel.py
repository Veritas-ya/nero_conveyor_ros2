"""Read-only rendering of the same simulated RGB-D frame used by detection."""
import cv2
import numpy as np


def visual_speed_world(frame, base_to_world_rotation, *, track_id=None):
    """Read a live qualified Kalman estimate; never substitute simulator truth."""
    candidates=[]
    for identity,entry in ((frame.get('nero_capture') or {}).get('measurements') or {}).items():
        report=entry.get('velocity_measurement') or {}
        if track_id is not None and str(identity)!=str(track_id):
            continue
        if not report.get('speed_valid') or report.get('motion_source')!='nero7_xy_kalman':
            continue
        v=np.asarray(report.get('velocity_base_m_s'),dtype=float)
        rotation=np.asarray(base_to_world_rotation,dtype=float)
        if v.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(v).all() or not np.isfinite(rotation).all():
            continue
        candidates.append({'track_id':int(identity),'velocity_world_m_s':rotation@v,
                           'frame_time_s':float(frame['timestamp'])})
    return candidates[0] if len(candidates)==1 else None


def speed_hud_values(live, *, now_s, frozen_velocity, belt_speed, truth_speed=None, paused=False):
    """Separate live measurement, frozen control input and simulation truth."""
    if live is None or not 0 <= now_s-live['frame_time_s'] <= .25:
        live_text='Visual Kalman: -- (no fresh track)'
    else:
        live_text=f"{'Last' if paused else 'Live'} visual ID {live['track_id']}: {100*live['velocity_world_m_s'][0]:+.2f} cm/s"
    frozen_text=('Frozen for plan: --' if frozen_velocity is None else
                 f'Frozen for plan: {100*np.asarray(frozen_velocity)[0]:+.2f} cm/s')
    belt_text=f'Sim belt setting: {100*belt_speed:+.2f} cm/s'
    truth_text=('Sim object x velocity: --' if truth_speed is None else
                f'Sim object x velocity: {100*truth_speed:+.2f} cm/s')
    return live_text,frozen_text,belt_text,truth_text


def render_panel(frame, *, status, completed, requested, fixed_speed_m_s=None, speed_lines=(), visual_speed_source='kalman'):
    rgb = np.asarray(frame['top_rgb'])
    depth = np.asarray(frame['top_depth'])
    labels = np.asarray(frame['top_instance_labels'])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2] or labels.shape != depth.shape:
        raise ValueError('simulation RGB, depth and labels must share image dimensions')
    color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    valid = np.isfinite(depth) & (depth > 0.)
    norm = np.zeros(depth.shape, np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], [5., 95.])
        norm[valid] = np.clip((depth[valid]-lo)*255./max(hi-lo, 1e-6), 0, 255).astype(np.uint8)
    depth_color = cv2.applyColorMap(255-norm, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    meta = frame.get('nero_capture') or {}
    measurements = meta.get('measurements') or {}
    for track_id in measurements:
        mask = (labels == int(track_id)).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cv2.drawContours(color, contours, -1, (0, 255, 255), 2)
        x, y, w, h = cv2.boundingRect(mask)
        cv2.putText(color, f'ID {track_id}', (x, max(18, y-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
    view = np.concatenate((color, depth_color), axis=1)
    source = ('fixed prediction %.2f cm/s' % (fixed_speed_m_s*100.) if fixed_speed_m_s is not None
              else {'constant_fit': 'visual constant fit prediction', 'corrected_kalman': 'corrected visual Kalman prediction'}.get(visual_speed_source, 'visual Kalman prediction'))
    lines = (f'MuJoCo simulated RGB-D  |  {status}  |  completed {completed}/{requested}',
             f'frame {meta.get("sequence", "-")}  tracks {len(measurements)}  |  {source}',
             'Left: RGB + detected contour     Right: depth (frame-relative colors)', *speed_lines)
    font_scale=.9 if view.shape[1]>=1600 else .45
    line_height=34 if font_scale>.5 else 21
    cv2.rectangle(view, (0, 0), (view.shape[1], 8+line_height*len(lines)), (20, 20, 20), -1)
    for i, line in enumerate(lines):
        cv2.putText(view, line, (8, line_height-5+line_height*i), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), 2 if font_scale>.5 else 1, cv2.LINE_AA)
    return view
