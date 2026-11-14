"""Per-exposure experiment evidence, separate from replaceable preview files.

Producers copy frozen inputs and enqueue without waiting for disk. The writer
never acquires a camera/device or touches a planner model. Overflow and I/O
errors are explicit evidence gaps, never reported as successful recordings.
"""
from copy import deepcopy
from dataclasses import asdict
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time

import numpy as np

from .collision_capture import frame_metadata
from .diagnostics import json_value


class ExperimentArchive:
    def __init__(self, output, emit, *, capacity=16):
        self.directory = Path(output)/'experiment'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.emit = emit
        self.queue = Queue(maxsize=capacity)
        self.lock = Lock()
        self.stop = Event()
        self.targets = set()
        self.counter = self.saved = self.dropped = 0
        self.errors = []
        self.write_wall_s = self.max_offer_wall_s = 0.
        self.thread = Thread(target=self._run, name='experiment-evidence-writer', daemon=True)
        self.thread.start()

    def save_configuration(self, config):
        """Startup-only I/O, before opening robot or camera connections."""
        assets = self.directory/'configuration'
        assets.mkdir()
        manifest = {'raw':{k:v for k,v in config.raw.items() if k!='taught_segments'},
            'velocity_limits':config.velocity_limits, 'acceleration_limits':config.acceleration_limits,
            'source_sha256':config.source_sha256, 'files':{}}
        files = {'effective_config':config.source}
        files.update({key:config.raw[key] for key in ('camera_calibration_file','tcp_calibration_file',
            'taught_transfer_file','taught_return_file') if config.raw.get(key)})
        for key, source in files.items():
            path = Path(source)
            data = path.read_bytes()
            destination = assets/(key+path.suffix)
            destination.write_bytes(data)
            manifest['files'][key] = {'source':str(path), 'saved':destination.name,
                                      'sha256':sha256(data).hexdigest()}
        (assets/'manifest.json').write_text(json.dumps(json_value(manifest),ensure_ascii=False,indent=2)+'\n')

    def _error(self, name, error):
        text = f'{name}: {type(error).__name__}: {error}'
        with self.lock: self.errors.append(text)
        try: self.emit('experiment_record_failed', record=name, error=text)
        except Exception: pass

    def _offer(self, kind, metadata, arrays=None):
        begin = time.monotonic()
        with self.lock:
            self.counter += 1
            name = f'{self.counter:06d}_{kind}'
        try:
            # Own arrays before the producer can reuse or mutate its buffers.
            packet = (name, kind, deepcopy(metadata),
                      {key:np.asarray(value).copy() for key,value in (arrays or {}).items()})
            self.queue.put_nowait(packet)
        except Exception as error:
            if isinstance(error, Full):
                with self.lock: self.dropped += 1
            self._error(name, error)
            return None
        finally:
            with self.lock:
                self.max_offer_wall_s = max(self.max_offer_wall_s, time.monotonic()-begin)
        return name

    def _frame(self, frame, d2c):
        return (frame_metadata(frame, d2c),
            {'frame_rgb':frame.rgb, 'frame_depth_m':frame.depth_m})

    def offer_target(self, target):
        key = (tuple(target.identity), float(target.capture_s))
        with self.lock:
            if key in self.targets: return
            self.targets.add(key)
        try:
            detection = target.detection
            meta, arrays = self._frame(detection['frame'], detection['depth_to_color_cv'])
            meta.update(identity=target.identity, capture_s=target.capture_s,
                received_s=target.received_s, device_s=target.device_s,
                center=target.center, entry_position=target.entry_position,
                velocity=target.velocity, speed_report=target.speed_report,
                capture_state=target.state,
                detection={k:detection[k] for k in ('belt_region','obb','plane','width','height',
                    'footprint_m','camera_origin_base_m','local_processing') if k in detection},
                depth_description='Frozen calibrated native depth; do not calibrate twice',
                physical_success=None)
            arrays.update(points=target.points, grasps=np.asarray(target.grasps))
            for k in ('scene_points','scene_pixel_indices','mask'):
                if k in detection: arrays[k] = detection[k]
            name = self._offer('target', meta, arrays)
            if name is None:
                with self.lock: self.targets.discard(key)
        except Exception as error:
            with self.lock: self.targets.discard(key)
            self._error('target', error)

    def bind_planner(self, planner):
        original = planner.intercept
        clock = getattr(planner, 'clock', time.monotonic)
        @wraps(original)
        def intercept(target, state, *args, **kwargs):
            self.offer_target(target)
            request = self._offer('planning_input', {
                'identity':target.identity, 'capture_s':target.capture_s,
                'submission_s':clock(), 'start_state':state,
                'intercept_not_before_s':getattr(planner,'intercept_not_before_s',None),
                'measured_planning_lead_s':getattr(planner,'measured_planning_lead_s',None),
                'measured_post_planning_s':getattr(planner,'measured_post_planning_s',None)})
            try:
                plan = original(target, state, *args, **kwargs)
            except BaseException as error:
                self._offer('planning_failed', {'input_record':request,
                    'identity':target.identity, 'capture_s':target.capture_s,
                    'completed_s':clock(), 'error_type':type(error).__name__,
                    'error':str(error), 'details':getattr(error,'details',{})})
                raise
            try:
                path = plan.path
                self._offer('plan', {'input_record':request, 'identity':target.identity,
                    'capture_s':target.capture_s, 'completed_s':clock(),
                    'version':path.version, 'scene_revision':plan.scene_revision,
                    'stage':path.stage, 'duration_s':path.duration, 'planning_s':path.planning_s,
                    'reference':{k:getattr(plan.reference,k) for k in
                        ('capture_s','start_s','duration_s','follow_s','brake_s','prealign_s',
                         'rotate_first_s','capture_tcp_world','velocity_world','start_tcp_world',
                         'start_quaternion_wxyz','target_quaternion_wxyz') if hasattr(plan.reference,k)},
                    'diagnostics':path.diagnostics, 'timings_s':plan.timings_s,
                    'candidate_failures':plan.failures,
                    'meaning':'certified_planner_output_not_evidence_of_dispatch_or_physical_success'},
                    {'curve_c':path.curve.c,'curve_x':path.curve.x,
                     'selected_grasp':plan.selected_grasp})
            except Exception as error: self._error('plan', error)
            return plan
        planner.intercept = intercept

    def bind_single_exposure(self, vision):
        original = vision._prepare_single
        @wraps(original)
        def prepare(frame, surface, candidate, state, captured, generation):
            try:
                meta, arrays = self._frame(frame, vision.d2c)
                meta.update(identity=(frame.header.clock_epoch,generation,frame.header.sequence),
                    capture_s=captured, capture_state=state,
                    surface={k:v for k,v in asdict(surface).items() if k!='geometry_fingerprint'},
                    candidate_position=candidate.position,
                    candidate_point_count=candidate.point_count,
                    depth_description='Frozen calibrated native depth; do not calibrate twice')
                arrays['seed_pixel_indices'] = candidate.seed_pixel_indices
                request = self._offer('geometry_input', meta, arrays)
            except Exception as error:
                self._error('geometry_input',error); request = None
            try:
                result = original(frame,surface,candidate,state,captured,generation)
            except BaseException as error:
                self._offer('geometry_result', {'input_record':request,'status':'exception',
                    'error_type':type(error).__name__,'error':str(error),
                    'details':getattr(error,'details',{})})
                raise
            self._offer('geometry_result', {'input_record':request,
                'status':'ready' if result is not None else 'rejected_or_cancelled',
                'diagnostics':vision.geometry_progress.snapshot()})
            if result is not None: self.offer_target(result[1])
            return result
        vision._prepare_single = prepare

    def _write(self, name, kind, metadata, arrays):
        path = self.directory/(name+('.npz' if arrays else '.json'))
        temp = path.with_suffix('.tmp')
        if arrays:
            with temp.open('wb') as stream:
                np.savez(stream, metadata=np.asarray(json.dumps(json_value(metadata),ensure_ascii=False)), **arrays)
        else:
            temp.write_text(json.dumps(json_value(metadata),ensure_ascii=False,indent=2)+'\n')
        temp.replace(path)
        with (self.directory/'index.jsonl').open('a') as stream:
            stream.write(json.dumps(json_value({'record':name,'kind':kind,'file':path.name,
                'identity':metadata.get('identity'),'input_record':metadata.get('input_record')}),ensure_ascii=False)+'\n')

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try: packet = self.queue.get(timeout=.05)
            except Empty: continue
            begin = time.monotonic()
            try:
                self._write(*packet)
                with self.lock: self.saved += 1
            except Exception as error: self._error(packet[0],error)
            finally:
                self.write_wall_s += time.monotonic()-begin
                self.queue.task_done()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10.)
        result = {'directory':str(self.directory), 'offered':self.counter,
            'saved':self.saved,'dropped':self.dropped,'errors':list(self.errors),
            'writer_alive':self.thread.is_alive(),'write_wall_s':self.write_wall_s,
            'maximum_enqueue_wall_s':self.max_offer_wall_s,
            'complete':not self.thread.is_alive() and not self.errors and self.saved==self.counter,
            'physical_success_inferred':False, 'robot_commands':0}
        try:
            (self.directory/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        except Exception as error:
            self._error('summary',error);result.update(complete=False,errors=list(self.errors))
        return result
