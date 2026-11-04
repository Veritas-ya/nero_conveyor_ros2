"""Display-only Kalman telemetry. Never estimate or alter a control velocity."""
import json
import numpy as np


def live_speed_snapshot(frame, peer, report, direction, capture_s):
    if peer is None:return None
    diagnostics=(report or {}).get('diagnostics',{})
    details={d['id']:d for d in diagnostics.get('tracks',[])}
    rows=[]
    for track in peer.tracks:
        extra=details.get(track.track_id,{})
        velocity=track.velocity_base_m_s
        if velocity is None:velocity=extra.get('velocity_base_m_s')
        rows.append({'id':int(track.track_id),'position_base_m':list(track.observed_position_base),
            'velocity_base_m_s':None if velocity is None else list(velocity),
            'along_belt_m_s':None if velocity is None else float(np.asarray(velocity)@direction),
            'speed_valid':bool(peer.current and track.speed_valid and not track.misses),
            'confirmed':bool(track.confirmed),'misses':int(track.misses),
            'sample_count':int(track.sample_count),'span_s':float(track.span_s),
            'accepted_count':extra.get('accepted_count')})
    source=(report or {}).get('motion_source','nero7_xy_kalman')
    return {'source':source,'frame_sequence':int(frame.header.sequence),
        'control_velocity_source':(report or {}).get('control_velocity_source',source),
        'fixed_control_speed_m_s':(report or {}).get('fixed_control_speed_m_s'),
        'clock_epoch':int(peer.clock_epoch),'track_generation':int(peer.track_generation),
        'capture_s':float(capture_s),'current':bool(peer.current),'tracks':rows}


def frozen_speed_snapshot(target, direction, accepted_s):
    report=getattr(target,'speed_report',{})
    velocity=np.asarray(target.velocity)
    position=getattr(target,'entry_position',None)
    frame=target.detection.get('frame')
    return {'source':report.get('source'),'identity':list(target.identity),
        'configured_speed_m_s':report.get('configured_speed_m_s'),
        'calibration_id':None if frame is None else frame.calibration_id,
        'serial_number':None if frame is None else frame.serial_number,
        'frame_sequence':int(target.sequence),'clock_epoch':int(target.epoch),
        'capture_s':float(target.capture_s),'accepted_s':float(accepted_s),
        'velocity_base_m_s':velocity.tolist(),'along_belt_m_s':float(velocity@direction),
        'position_base_m':None if position is None else np.asarray(position).tolist(),
        'sample_count':report.get('sample_count'),'span_s':report.get('span_s'),
        'speed_std_m_s':report.get('speed_std_m_s')}


def frozen_from_events(path, identity, direction):
    """Recover only values that older runs actually logged, without modifying them."""
    result=None
    for line in path.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue  # journal may be mid-append
        data=row.get('data',{})
        if row.get('event')=='qualified_target' and data.get('identity')==identity:
            report=data.get('speed_report',{});velocity=report.get('prediction_velocity_base_m_s') if report.get('source')=='operator_fixed_belt_speed' else report.get('measured_velocity_base_m_s')
            result={'source':report.get('source'),'identity':identity,'from_saved_events':True,
                'configured_speed_m_s':report.get('configured_speed_m_s'),
                'frame_sequence':data.get('sequence'),'clock_epoch':identity[0],
                'capture_s':data.get('capture_monotonic_s'),'accepted_s':row['monotonic_s'],
                'velocity_base_m_s':velocity,'position_base_m':None,
                'along_belt_m_s':None if velocity is None else float(np.asarray(velocity)@direction),
                'sample_count':report.get('sample_count'),'span_s':report.get('span_s')}
        if row.get('event')=='execution_event' and result is not None:
            record=data.get('record',{})
            if record.get('event')=='planning_submitted' and record['data'].get('identity')==identity:
                result['position_base_m']=record['data'].get('entry_observed_position_base_m')
    return result


def speed_lines(live, frozen, *, now, maximum_age_s, historical=False):
    """Text and RGB colors, shared by fixed-size HUDs and Qt status labels."""
    rows=[] if live is None else [t for t in live['tracks'] if t.get('along_belt_m_s') is not None]
    rows.sort(key=lambda t:(not t['speed_valid'],not t['confirmed'],t['misses'],-t['sample_count']))
    prefix='历史卡尔曼' if historical else '实时卡尔曼'
    if live is not None and not historical and now-live['capture_s']>maximum_age_s:
        prefix='卡尔曼末值（未更新）'
    if live is not None and live.get('control_velocity_source')=='operator_fixed_belt_speed':
        prefix+='参考（不用于本次控制）'
    if rows:
        t=rows[0];qualifier='' if t['speed_valid'] else '（待确认）'
        extra='' if len(rows)==1 else f'，另{len(rows)-1}条轨迹'
        first=f'{prefix} ID{t["id"]}：{t["along_belt_m_s"]*100:.2f} cm/s{qualifier}{extra}'
    else:first=f'{prefix}：等待有效观测'
    second='冻结速度：尚未冻结'
    if frozen and frozen.get('along_belt_m_s') is not None:
        if frozen.get('source')=='operator_fixed_belt_speed':
            second=f'用户设定带速 ID{frozen["identity"][2]}：{frozen["configured_speed_m_s"]*100:.2f} cm/s'
        else:
            second=f'冻结速度 ID{frozen["identity"][2]}：{frozen["along_belt_m_s"]*100:.2f} cm/s'
    return [(first,(90,235,255)),(second,(255,224,80))]


def paired_track_labels(live, frame, direction):
    if (not live or live['frame_sequence']!=frame.header.sequence or
            live['clock_epoch']!=frame.header.clock_epoch):return []
    return [{'id':t['id'],'position':t['position_base_m'],'velocity':t['velocity_base_m_s'],
        'reasons':[] if t['speed_valid'] else ['滤波待确认'],
        's_m':float(np.asarray(t['position_base_m'])@direction)}
        for t in live['tracks'] if not t['misses']]
