import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('camera_profile',
    Path(__file__).resolve().parents[1]/'tools/profile_camera_geometry.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_nested_timer_exclusive_parts_sum_and_exception_is_recorded(monkeypatch):
    wall = iter([0., 1., 4., 6.])
    cpu = iter([0., .5, 2., 3.])
    monkeypatch.setattr(profile.time, 'perf_counter', lambda:next(wall))
    monkeypatch.setattr(profile.time, 'thread_time', lambda:next(cpu))
    timing = profile.Timings()
    with pytest.raises(ValueError):
        with timing.measure('total'):
            with timing.measure('child'):
                raise ValueError('rejected')
    assert timing.values['total']['wall_s'] == 6
    assert timing.values['total']['exclusive_wall_s'] == 3
    assert timing.values['child']['exclusive_wall_s'] == 3
    assert timing.values['total']['thread_cpu_s'] == 3
    assert timing.stack == []


def test_empty_and_rejected_samples_do_not_bias_success_distribution():
    def stage(value):
        return dict(wall_s=value, thread_cpu_s=value, exclusive_wall_s=value,
            exclusive_thread_cpu_s=value)
    result = profile.summarize([
        dict(status='geometry_ready', stages={'geometry_total':stage(.1)}),
        dict(status='geometry_rejected', stages={'geometry_total':stage(.01)}),
        dict(status='no_single_candidate', stages={})])
    assert result['geometry_ready']['stages']['geometry_total']['wall_s']['p50_ms'] == 100
    assert result['geometry_rejected']['count'] == 1
    assert result['no_single_candidate']['stages'] == {}
    assert profile.summarize([]) == {}
    assert profile.distribution([]) is None


def test_invalid_duration_fails_before_any_camera_connection(tmp_path):
    with pytest.raises(SystemExit) as error:
        profile.main(['--output', str(tmp_path/'sample'), '--seconds', 'nan'])
    assert error.value.code == 2
    assert not (tmp_path/'sample').exists()


def test_passages_require_sustained_empty_gap_and_preserve_rejected_occupancy():
    p = profile.PassageTracker()
    assert p.update(1,0) == (1,None)
    assert p.update(0,.1) == (1,None)
    assert p.update(1,.3) == (1,None)
    assert p.update(2,.4) == (1,None)
    assert p.update(0,.5) == (1,None)
    assert p.update(None,1.3) == (1,None)
    assert p.update(0,1.4) == (1,None)
    assert p.update(0,2.2) == (None,1)
    assert p.update(1,3) == (2,None)
    assert p.completed == [1]
