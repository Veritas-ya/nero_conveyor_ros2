"""Never combine a path with another target's frozen exposure."""
import json
from pathlib import Path
import time

import numpy as np
import pytest

from nero_core.control_panel.media import ArtifactReader


def save(directory, name, sequence=1, identity=(0,0,1)):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {'header': {'sequence':sequence, 'received_time':time.monotonic()},
        'depth_intrinsics': {'fx':100.,'fy':100.,'cx':8.,'cy':8.},
        'depth_to_base_cv': np.eye(4).tolist(), 'depth_to_color_cv': np.eye(4).tolist(),
        'identity':list(identity), 'frame_sequence':sequence, 'obb':{}}
    arrays = {'frame_rgb':np.full((16,16,3),100,np.uint8),
        'frame_depth_m':np.ones((16,16),np.float32), 'points':np.array([[0.,0.,1.]]),
        'grasps':np.eye(4)[None], 'tcp_xyz_base_m':np.array([[0.,0.,1.],[.1,0.,1.]]),
        'contact_grasp_base':np.eye(4), 'predicted_contact_points_base_m':np.array([[0.,0.,1.]])}
    np.savez(directory/(name+'.npz'), metadata=np.asarray(json.dumps(metadata)), **arrays)


def test_live_frame_and_paired_target_plan_are_read_from_files_only(tmp_path):
    for kind in ('latest','target','plan'): save(tmp_path/'preview',kind)
    reader = ArtifactReader(); reader.update(tmp_path); snap=reader.snapshot()
    assert snap['frame']['available'] and snap['frame']['sequence']==1
    # Nanosecond file versions exceed JavaScript's safe integer range. They
    # must survive the browser round-trip exactly for exposure pairing.
    assert isinstance(snap['frame']['version'],str) and isinstance(snap['target']['version'],str)
    assert reader.live_image.startswith(b'\xff\xd8') and reader.frozen_image.startswith(b'\xff\xd8')
    assert snap['target']['point_count']==1 and snap['plan']['available']
    assert len(snap['plan']['tcp'])==2


@pytest.mark.parametrize('sequence,identity', [(2,(0,0,1)),(1,(0,0,2))])
def test_other_frame_or_identity_path_is_hidden(tmp_path,sequence,identity):
    save(tmp_path/'preview','target'); save(tmp_path/'preview','plan',sequence,identity)
    reader=ArtifactReader();reader.update(tmp_path);snap=reader.snapshot()
    assert not snap['plan']['available'] and snap['plan']['hidden_mismatch']
    assert snap['plan']['tcp']==[] and snap['plan']['contact_grasp'] is None


def test_rearm_removes_old_frozen_view_and_plan(tmp_path):
    save(tmp_path/'preview','target');save(tmp_path/'preview','plan')
    reader=ArtifactReader();reader.update(tmp_path)
    (tmp_path/'preview/target.npz').unlink();(tmp_path/'preview/plan.npz').unlink()
    reader.update(tmp_path);snap=reader.snapshot()
    assert not snap['target']['available'] and not snap['plan']['available']
    assert reader.frozen_image is None


def test_changing_runs_cannot_reuse_old_live_image_or_geometry(tmp_path):
    save(tmp_path/'first/preview','latest');save(tmp_path/'first/preview','target')
    reader=ArtifactReader();reader.update(tmp_path/'first');reader.update(tmp_path/'second')
    assert not reader.snapshot()['frame']['available']
    assert not reader.snapshot()['target']['available']


def test_unreadable_artifact_is_diagnostic_not_a_fabricated_path(tmp_path):
    (tmp_path/'preview').mkdir();(tmp_path/'preview/plan.npz').write_bytes(b'partial')
    reader=ArtifactReader();reader.update(tmp_path)
    assert reader.snapshot()['error'] and not reader.snapshot()['plan']['available']
