"""Bounded, asynchronous recording of the grasp camera's uncalibrated frames."""
import json
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
import time

import numpy as np
from .collision_capture import frame_metadata
from .diagnostics import json_value


class RGBDArchive:
    def __init__(self, output, *, rate_hz=2.):
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError('recording rate must be positive')
        self.directory = Path(output) / 'rgbd_frames'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.period = 1. / rate_hz
        self.queue = Queue(maxsize=4)
        self.stop = Event()
        self.last = -float('inf')
        self.offered = self.saved = self.dropped = 0
        self.errors = []
        self.write_wall_s = 0.
        self.thread = Thread(target=self._run, name='rgbd-record-writer', daemon=True)
        self.thread.start()

    def offer_frame(self, frame, d2c):
        # Host receive time is monotonic; camera clock resets do not stop recording.
        now = float(frame.header.received_time)
        if now >= self.last and now - self.last < self.period:
            return
        self.last = now
        self.offered += 1
        try:
            self.queue.put_nowait((frame, np.asarray(d2c).copy()))
        except Full:
            self.dropped += 1

    def _run(self):
        while not self.stop.is_set() or not self.queue.empty():
            try:
                frame, d2c = self.queue.get(timeout=.1)
            except Empty:
                continue
            started = time.monotonic()
            try:
                meta = frame_metadata(frame, d2c)
                meta['depth_description'] = ('SDK depth converted to metres and range filtered; '
                                             'before runtime depth calibration')
                name = f"epoch_{frame.header.clock_epoch:03d}_frame_{frame.header.sequence:09d}.npz"
                dest = self.directory / name
                temp = dest.with_suffix('.tmp')
                with temp.open('wb') as stream:
                    np.savez(stream, frame_rgb=frame.rgb, frame_depth_m=frame.depth_m,
                             metadata=np.asarray(json.dumps(json_value(meta), ensure_ascii=False)))
                temp.replace(dest)
                self.saved += 1
            except Exception as error:
                self.errors.append(f'{type(error).__name__}: {error}')
            finally:
                self.write_wall_s += time.monotonic() - started
                self.queue.task_done()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10.)
        return {'directory': str(self.directory), 'rate_hz': 1. / self.period,
                'offered': self.offered, 'saved': self.saved, 'dropped': self.dropped,
                'errors': list(self.errors), 'writer_alive': self.thread.is_alive(),
                'write_wall_s': self.write_wall_s}
