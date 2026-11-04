"""Bounded recording, identity deduplication and complete failure evidence."""
import json
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from test_collision_visualization import fixture
from test_two_second_hardware import site
from nero_core.two_second.planning_test_record import PlanningTestRecord, RecordingTrajectoryPreview
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def sample():
    _, frame = fixture()
    track = SimpleNamespace(track_id=0, confirmed=True, misses=0, speed_valid=True,
        span_s=.5, velocity_base_m_s=np.array([.1, 0, 0]),
        observed_position_base=np.array([-.3, 0., 1.]))
    peer = SimpleNamespace(clock_epoch=0, track_generation=1, current=True, tracks=[track])
    points = np.tile([[-.3, 0., 1.]], (24, 1))
    config = SimpleNamespace(direction=np.array([1., 0, 0]), follow_bounds=np.array([-.2, .38]),
        raw={'velocity_source':'operator_fixed_belt_speed', 'fixed_belt_speed_m_s':.1,
             'maximum_observation_age_s':1., 'planning_reserve_mode':'measured',
             'synchronized_adaptive_freeze':True, 'synchronized_max_freeze_upstream_distance_m':.35,
             'freeze_upstream_distance_m':.1})
    done = {'geometry':{'frame':frame, 'd2c':np.eye(4), 'grasps':[np.eye(4)]},
        'peer':peer, 'capture':time.monotonic(),
        'detection':{'object_points':points, 'plane':{'normal_base':[0.,0.,1.]}}}
    return config, done, track


def test_ten_tracks_count_once_record_nested_failures_and_stop_new_planning(tmp_path, monkeypatch):
    config, done, track = sample()
    record = PlanningTestRecord(tmp_path, 10)
    preview = RecordingTrajectoryPreview(config, tmp_path/'paths', record)
    calls = []
    def calculate(done, track, gate, generation):
        calls.append(track.track_id)
        if track.track_id == 3:
            raise PickRejected('所有候选拒绝', details={'failures':[
                {'failed_phase':'sampled_geometry','reason':'机器人与实测工件点云碰撞',
                 'details':{'geometry':'right_finger_geom','hit_count':95}}]})
        return {'generation':generation, 'status':'模型路径已生成', 'file':str(tmp_path/'path'),
            'report':{'actual_remaining_send_lead_s':-.02 if track.track_id==5 else .3,
                'trajectory_duration_s':2.8, 'candidate_failures':[], 'timing':'virtual_clock'}}
    monkeypatch.setattr(preview, '_calculate', calculate)
    try:
        for identity in range(12):
            track.track_id = identity
            done['capture'] = time.monotonic()
            preview.last_submitted = -float('inf')
            preview.submit(done, .1)
            if preview.future is not None:
                try: preview.future.result()
                except PickRejected: pass
                record.refresh_archives(wait=True)
                preview.poll()
            # Reusing the same track after a reset must not spend another test.
            preview.reset()
            preview.last_submitted = -float('inf')
            preview.submit(done, .1)
            assert preview.future is None
        assert calls == list(range(10))
        assert record.complete
        report = json.loads((tmp_path/'summary.json').read_text())
        assert report['completed_attempts'] == 10
        assert report['model_paths_generated'] == 9 and report['planning_failures'] == 1
        assert report['deadline_misses'] == 1
        assert report['attempts'][3]['failure_details']['failures'][0]['details']['hit_count'] == 95
        assert report['attempts'][5]['timing_feasible'] is False
        assert len((tmp_path/'attempts.jsonl').read_text().splitlines()) == 10
        with np.load(tmp_path/'attempt_04.npz', allow_pickle=False) as evidence:
            np.testing.assert_array_equal(evidence['frame_rgb'], done['geometry']['frame'].rgb)
        assert report['can_connections'] == report['robot_commands_sent'] == 0
    finally:
        preview.close()


def test_compression_does_not_block_planning_and_archive_failure_is_reported(tmp_path, monkeypatch):
    from threading import Event
    from concurrent.futures import ThreadPoolExecutor
    config, done, track = sample()
    record = PlanningTestRecord(tmp_path, 1)
    entered, release = Event(), Event()
    def archive(*args, **kwargs):
        entered.set()
        assert release.wait(5.)
        raise OSError('injected archive failure')
    monkeypatch.setattr(np, 'savez_compressed', archive)
    row = record.begin(record.camera_session(), done, track)
    assert not entered.is_set() and not (tmp_path/'attempt_01.npz').exists()
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            record.finish(row, error=RuntimeError('injected planning rejection'), wall_s=.2, archive_pool=pool)
            assert entered.wait(5.)
            assert row['status'] == 'planning_failed' and not record.complete
            assert row['evidence_status'] == 'pending'
        finally:
            release.set()
            record.refresh_archives(wait=True)
    report = json.loads((tmp_path/'summary.json').read_text())
    assert record.complete and report['pending_evidence_files'] == 0
    assert report['attempts'][0]['evidence_status'] == 'failed'
    assert 'injected archive failure' in report['attempts'][0]['evidence_error']


@pytest.mark.parametrize('condition', ['upstream','stale','unqualified'])
def test_unaccepted_observation_does_not_spend_a_test(tmp_path, condition):
    config, done, track = sample()
    record = PlanningTestRecord(tmp_path, 10)
    preview = RecordingTrajectoryPreview(config, tmp_path/'paths', record)
    if condition == 'upstream': track.observed_position_base[0] = -.7
    elif condition == 'stale': done['capture'] -= 2.
    else: track.confirmed = False
    try:
        preview.submit(done, .1)
        assert preview.future is None and not record.rows
    finally:
        preview.close()


def test_new_camera_session_can_record_reused_tracker_numbers_without_overwriting(tmp_path):
    config, done, track = sample()
    record = PlanningTestRecord(tmp_path, 2)
    row = record.begin(record.camera_session(), done, track)
    record.finish(row, error=RuntimeError('injected'), wall_s=.1)
    row = record.begin(record.camera_session(), done, track)
    assert row['attempt'] == 2
    record.finish(row, error=RuntimeError('injected'), wall_s=.1)
    assert record.complete
    with pytest.raises(ValueError, match='目录已存在'):
        PlanningTestRecord(tmp_path, 2)


def test_exact_cached_plane_and_tracking_are_archived_and_path_io_failure_stays_separate(tmp_path):
    from concurrent.futures import Future
    _, done, track = sample()
    plane = np.eye(4); plane[2, 3] = -.123
    done['detection'].update(belt_region={'plane_to_base':plane, 'bounds_xy_m':[[-1,1],[-2,2]]},
                             scene_points=np.array([[.1,.2,-.123]]))
    record = PlanningTestRecord(tmp_path, 1)
    row = record.begin(record.camera_session(), done, track)
    broken = Future(); broken.set_exception(OSError('injected path archive failure'))
    record.finish(row, result={'file':str(tmp_path/'path'), 'archive_future':broken,
        'report':{'actual_remaining_send_lead_s':.1, 'trajectory_duration_s':2.8,
                  'candidate_failures':[], 'timing':'real_monotonic_test'}}, wall_s=.2)
    assert not record.complete
    record.refresh_archives()
    saved = json.loads((tmp_path/'summary.json').read_text())
    assert record.complete and saved['archive_failures'] == 1
    assert saved['planning_failures'] == 0 and saved['model_paths_generated'] == 1
    assert 'injected path archive failure' in saved['attempts'][0]['path_archive_error']
    with np.load(tmp_path/'attempt_01.npz', allow_pickle=False) as evidence:
        metadata = json.loads(str(evidence['detection_metadata']))
        np.testing.assert_array_equal(metadata['belt_region']['plane_to_base'], plane)
        np.testing.assert_array_equal(evidence['scene_points'], done['detection']['scene_points'])
        target = json.loads(str(evidence['target_metadata']))
        np.testing.assert_array_equal(target['observed_position_base_m'], track.observed_position_base)
        assert target['capture_monotonic_s'] == done['capture']


def test_fixed_speed_cli_records_effective_units_without_changing_site_or_opening_can(site, tmp_path, monkeypatch):
    from nero_core.two_second.planning_test_record import main
    from nero_core.two_second.config import CellConfig
    from nero_core.two_second.velocity_source import with_fixed_belt_speed, prediction_velocity
    import yaml
    path, _ = site
    original = path.read_bytes()
    called = []
    def camera_main(argv):
        called.append(argv)
        assert '--trajectory-preview' in argv
        assert '--read-robot-feedback' not in argv and '--follow-run' not in argv
        speed = float(argv[argv.index('--fixed-belt-speed-m-s')+1])
        assert speed == pytest.approx(.1727)
        cfg = with_fixed_belt_speed(CellConfig.load(path), speed)
        velocity = prediction_velocity(cfg, np.array([8., -3., 0.]), np.array([.01, .04, .999]))
        assert np.linalg.norm(velocity) == pytest.approx(.1727)
        return 0
    monkeypatch.setitem(sys.modules, 'nero_core.two_second.camera_view', SimpleNamespace(main=camera_main))
    output = tmp_path/'recording'
    assert main(['--config',str(path),'--output',str(output),'--belt-speed-cm-s','17.27']) == 0
    assert len(called) == 1 and path.read_bytes() == original
    report = json.loads((output/'summary.json').read_text())
    assert report['velocity_policy']['fixed_belt_speed_cm_s'] == pytest.approx(17.27)
    assert report['velocity_policy']['visual_velocity_used'] is False
    assert report['can_connections'] == report['robot_commands_sent'] == 0
    effective = yaml.safe_load((output/'site_config.yaml').read_text())
    assert effective['fixed_belt_speed_m_s'] == pytest.approx(.1727)
    assert effective['velocity_source'] == 'operator_fixed_belt_speed'
    assert (output/'source_site_config.yaml').read_bytes() == original
