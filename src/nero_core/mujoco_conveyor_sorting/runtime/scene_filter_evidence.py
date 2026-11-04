"""Restore same-input scene masks for playback, without recomputing a network."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(points):
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float64).tobytes()).hexdigest()


def restore_scene_filter_evidence(points, frame, evidence, protected_mask=None, *,
                                  expected_metadata=None):
    """Return the saved mask only for the exact native samples and capture.

    A missing/invalid optional attachment returns None, allowing the original
    native filtering path. Inputs from a different scene are never reusable.
    """
    if frame is None or not isinstance(evidence, dict) or not evidence.get('artifact_prefix'):
        return None
    try:
        if evidence.get('capture_header') != asdict(frame.header):
            return None
        expected = {
            'serial_number': frame.serial_number, 'calibration_id': frame.calibration_id,
            'camera_to_base_sha256': _sha(frame.camera_to_world_cv),
            'protected_target_mask_sha256': (None if protected_mask is None else
                hashlib.sha256(np.ascontiguousarray(protected_mask, dtype=bool).tobytes()).hexdigest())}
        expected.update(expected_metadata or {})
        prefix = Path(evidence['artifact_prefix'])
        document = json.loads(prefix.with_suffix('.json').read_text())
        saved = document['statistics']
        if saved.get('capture_header') != asdict(frame.header):
            return None
        current = np.asarray(points)
        before = np.load(str(prefix)+'_before.npy', allow_pickle=False)
        mask = np.load(str(prefix)+'_removed_mask.npy', allow_pickle=False)
        if (before.shape != current.shape or mask.shape != (len(current),)
                or mask.dtype != np.bool_ or not np.isfinite(before).all()
                or not np.array_equal(before, current)):
            return None
        digest = _sha(current)
        for metadata in (evidence, saved):
            if any(metadata.get(key) != value for key, value in expected.items()):
                return None
            if metadata['artifacts']['before']['sha256'] != digest:
                return None
            if metadata['artifacts']['removed']['sha256'] != _sha(current[mask]):
                return None
            if metadata['artifacts']['retained']['sha256'] != _sha(current[~mask]):
                return None
            mask_sha = hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()
            if metadata['artifacts']['removed_mask']['sha256'] != mask_sha:
                return None
        from .scene_stereo_filter import SceneStereoFilterResult
        stats = dict(saved, artifact_prefix=str(prefix), replayed_same_input_mask=True)
        return SceneStereoFilterResult(current.copy(), mask, stats, document.get('components', []))
    except Exception:
        return None


def relocated_scene_filter_evidence(recording, scene):
    """Resolve copied recordings using their own relative attachment folders."""
    if not isinstance(scene, dict):
        return {}
    result = {}
    for key, folder in (('scene_saturation_filter', 'saturation'),
                        ('scene_background_filter', 'background'), ('scene_stereo_filter', 'stereo')):
        metadata = scene.get(key)
        if not isinstance(metadata, dict) or not metadata.get('artifact_prefix'):
            continue
        original = Path(metadata['artifact_prefix'])
        local = Path(recording)/'scene_depth_recheck'/folder/original.name
        prefix = local if local.with_suffix('.json').is_file() else original
        result[key] = dict(metadata, artifact_prefix=str(prefix))
    return result
