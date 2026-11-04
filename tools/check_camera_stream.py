#!/usr/bin/env python3
"""Audit camera delivery/exposure and Nero7 speed; never connects to a robot."""
import argparse
import json
from pathlib import Path
import time
import numpy as np


def summarize_archive(directory):
    """Archive stride is intentional; compare SDK index increments to received sequence."""
    records=[]
    for path in sorted((Path(directory)/'rgbd_frames').glob('*.npz')):
        with np.load(path,allow_pickle=False) as packet:
            m=json.loads(str(packet['metadata']))
        h=m['header'];timing=m.get('timestamp_metadata',{})
        d=timing.get('depth') or {};c=timing.get('color') or {}
        # Historical color exposure_us was mislabeled. Reinterpret only this
        # report, from raw metadata and the verified Gemini335L sensor units.
        records.append({'sequence':h['sequence'],'epoch':h['clock_epoch'],
            'device_s':d.get('device_time_s'),'device_index':d.get('frame_index'),
            'depth_exposure_us':d.get('frame_metadata',{}).get('EXPOSURE'),
            'color_exposure_us':(None if c.get('frame_metadata',{}).get('EXPOSURE') is None
                                 else c['frame_metadata']['EXPOSURE']*100)})
    if not records:
        raise ValueError(f'No archived RGB-D frames in {directory}')
    pairs=[]
    for a,b in zip(records,records[1:]):
        if a['epoch']!=b['epoch'] or any(r[k] is None for r in (a,b) for k in ('device_s','device_index')):continue
        dt=b['device_s']-a['device_s'];di=b['device_index']-a['device_index'];dq=b['sequence']-a['sequence']
        if dt>0 and di>=0 and dq>0:pairs.append({'device_counter_fps':di/dt,'device_index_increment':di,
            'received_sequence_increment':dq,'unreceived_device_frames':max(0,di-dq)})
    processing=[]
    events=Path(directory)/'events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            row=json.loads(line)
            if row.get('event')!='runtime_progress':continue
            perception=(row.get('data',{}).get('vision') or {}).get('perception',{})
            value=perception.get('processing_ms')
            if value is not None:processing.append(float(value))
    return {'mode':'offline_archive','source':str(Path(directory).resolve()),'camera_connections':0,
        'robot_commands':0,'samples':records,'pairs':pairs,
        'exposure_units_assumption':'Gemini335L: depth raw=1us, color raw=100us; historical archives lack product ID.',
        'processing_sample_count':len(processing),
        'processing_sample_median_ms':float(np.median(processing)) if processing else None,
        'processing_sample_max_ms':max(processing) if processing else None,
        'recording_note':'Archive is rate limited; gaps between saved frames are not themselves camera drops.',
        'unreceived_device_frames_in_saved_intervals':sum(p['unreceived_device_frames'] for p in pairs),
        'median_device_counter_fps':float(np.median([p['device_counter_fps'] for p in pairs])) if pairs else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'))
    parser.add_argument('--offline-run',type=Path)
    parser.add_argument('--seconds',type=float,default=20.)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds<=0:parser.error('seconds must be finite and positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if args.offline_run:
        result=summarize_archive(args.offline_run)
    else:
        from nero_core.two_second.config import CellConfig
        from nero_core.two_second.planning import calibrated_model
        from nero_core.two_second.vision import LiveVision
        from nero_core.two_second.diagnostics import json_value
        from mujoco_conveyor_sorting.runtime.backends.orbbec import OrbbecCameraBackend
        from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
        config=CellConfig.load(args.config);cal=load_runtime_depth_calibration(config.raw['camera_calibration_file'])
        cfg=dict(config.raw['camera']);cfg.update(enable_color=True,require_color=True,timestamp_source='device',
            serial_number=cal['serial_number'],calibration_id=cal['calibration_id'],camera_to_world_cv=cal['depth_to_base_cv'])
        model=calibrated_model(config);camera=OrbbecCameraBackend(cfg);vision=None;rows=[]
        result={'mode':'live_camera_only','robot_connections':0,'robot_commands':0,'status':'starting'}
        def speed_sink(frame,d2c,peer,report,*,capture_s):
            rows.append({'sequence':frame.header.sequence,'capture_s':capture_s,
                'clock_epoch':frame.header.clock_epoch,
                'track_generation':None if peer is None else peer.track_generation,
                'source_s':frame.header.source_time,'metadata':frame.timestamp_metadata,
                'processing_ms':(report or {}).get('diagnostics',{}).get('processing_ms'),
                'tracks':[] if peer is None else [{'id':t.track_id,'speed_valid':t.speed_valid,
                    'velocity_m_s':t.velocity_base_m_s,'samples':t.sample_count,'span_s':t.span_s}
                    for t in peer.tracks]})
        try:
            camera.start();vision=LiveVision(camera,cal,config,model,lambda:None)
            vision.speed_sink=speed_sink;vision.start();began=time.monotonic()
            while time.monotonic()-began<args.seconds:
                if vision.error:raise vision.error
                time.sleep(.05)
            result.update(status='captured',vision=vision.diagnostics())
        except KeyboardInterrupt:result['status']='cancelled'
        except Exception as error:result.update(status='failed',error=str(error))
        finally:
            if vision is not None:vision.close()
            result['capture']=camera.get_capture_diagnostics();camera.close()
        # Nothing large is serialized on the measurement callback.
        with (args.output/'frames.jsonl').open('w') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        result['processed_measurement_frames']=len(rows)
        result['frame_exposure_us']={name:sorted({r['metadata'][name]['exposure_us'] for r in rows
            if (r['metadata'].get(name) or {}).get('exposure_us') is not None}) for name in ('depth','color')}
        result['speed_by_track']=summarize_speeds(rows)
        result=json_value(result)
    (args.output/'report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','pairs','vision')},ensure_ascii=False,indent=2))
    return 1 if result.get('status')=='failed' else 0


def summarize_speeds(rows):
    """Describe valid measured speeds per identity; never pool different objects."""
    speeds={}
    for row in rows:
        for track in row['tracks']:
            if not track['speed_valid'] or track['velocity_m_s'] is None:continue
            # Identity includes clock and tracker generation across resets.
            key=f"{row.get('clock_epoch',0)}:{row.get('track_generation',0)}:{track['id']}"
            speeds.setdefault(key,[]).append(float(np.linalg.norm(track['velocity_m_s']))*1000)
    return {key:{'valid_samples':len(values),'mean_mm_s':float(np.mean(values)),
        'std_mm_s':float(np.std(values)),'peak_to_peak_mm_s':float(np.ptp(values))}
        for key,values in speeds.items()}


if __name__=='__main__':raise SystemExit(main())
