"""The real-time capture path is exercised here with a synthetic camera frame."""
from dataclasses import replace
import json
import numpy as np
from nero_core.two_second.rgbd_archive import RGBDArchive
from test_collision_visualization import fixture


def test_record_frames_with_timestamps_and_intrinsics(tmp_path):
    _, frame = fixture()
    writer = RGBDArchive(tmp_path)
    writer.offer_frame(frame, np.eye(4))
    writer.offer_frame(replace(frame, header=replace(frame.header, received_time=1.2, sequence=8)), np.eye(4))
    writer.offer_frame(replace(frame, header=replace(frame.header, received_time=1.6, sequence=9, clock_epoch=1)), np.eye(4))
    result = writer.close()
    assert result['saved'] == 2 and result['dropped'] == 0 and not result['errors']
    files = sorted((tmp_path / 'rgbd_frames').glob('*.npz'))
    with np.load(files[0], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['frame_rgb'], frame.rgb)
        np.testing.assert_array_equal(saved['frame_depth_m'], frame.depth_m)
        metadata = json.loads(str(saved['metadata']))
        assert metadata['header']['sequence'] == 7
        assert metadata['depth_intrinsics']['fx'] == frame.depth_intrinsics.fx
        assert 'before runtime depth calibration' in metadata['depth_description']
