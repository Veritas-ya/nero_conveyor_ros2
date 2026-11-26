"""Emit one measured-coordinate target cloud per successful same-exposure Fast result."""
import json
from pathlib import Path
from threading import Lock
from uuid import uuid4

import numpy as np

from mujoco_conveyor_sorting.runtime.artifact_writer import ArtifactWriter


class FastCloudDelivery:
    def __init__(self, directory):
        self.directory = None if directory is None else Path(directory).resolve()
        self.writer = None
        self.session = uuid4().hex
        self.lock = Lock()
        self.count = 0
        self.saved = 0
        self.errors = []
        self.latest = None
        self.emit = None

    def start(self):
        if self.directory is not None and self.writer is None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.writer = ArtifactWriter(self.directory)

    def publish(self, frame, points, indices, worker_prefix):
        with self.lock:
            self.count += 1
            cloud_id = self.count
            header = frame.header
            meta = dict(cloud_id=cloud_id, delivery_session=self.session, frame_sequence=int(header.sequence), clock_epoch=int(header.clock_epoch),
                        source_device_s=header.source_time, frame_sample_s=header.sample_time,
                        serial_number=frame.serial_number, calibration_id=frame.calibration_id,
                        depth_shape=list(frame.depth_m.shape), point_count=len(points),
                        coordinates='robot_base_m', source='fast_depth_at_target_pixels',
                        worker_artifact_prefix=worker_prefix)
            if self.directory is not None:
                meta['file'] = str(self.directory / f'cloud_{cloud_id:06d}_f{header.sequence:09d}_{self.session}.npz')
            points.setflags(write=False)
            indices.setflags(write=False)
            self.latest = (meta, points, indices)
        if self.emit is not None:
            self.emit('fast_cloud_delivered', **meta)
        if self.directory is not None:
            if self.writer is None:
                # Also support standalone users of refine() with an already warmed backend.
                self._save(meta, points, indices)
            else:
                self.writer.submit(self._save, meta, points, indices)
        return meta

    def _save(self, meta, points, indices):
        dest = Path(meta['file'])
        tmp = dest.with_suffix('.tmp')
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open('wb') as f:
                np.savez(f, metadata=np.asarray(json.dumps(meta, ensure_ascii=False)),
                         points_base_m=points, depth_pixel_indices=indices)
            tmp.replace(dest)
        except OSError as error:
            message = f'{dest}: {error}'
            try:
                tmp.unlink(missing_ok=True)
            except OSError as cleanup_error:
                message += f' (temp cleanup: {cleanup_error})'
            self.errors.append(message)
            if self.emit is not None:
                self.emit('fast_cloud_save_error', cloud_id=meta['cloud_id'], error=message)
            return {'artifact_error':message}
        self.saved += 1

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def summary(self):
        errors = list(self.errors)
        if self.writer is not None:
            for job in self.writer.records:
                if job.get('status')=='failed' and job.get('error') not in errors:
                    errors.append(job['error'])
            if getattr(self.writer, 'report_error', None):
                errors.append(self.writer.report_error)
        return dict(delivered=self.count, saved=self.saved if self.directory is not None else None,
                    archive_errors=errors)
