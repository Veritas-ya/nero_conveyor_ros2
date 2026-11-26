"""Deterministic scheduling and unchanged serialized observation contents."""
import json

import numpy as np

from mujoco_conveyor_sorting.runtime.archive_work import ArchiveWork
from mujoco_conveyor_sorting.runtime.execution_trace import ExecutionTrace
from nero_core.two_second.diagnostics import json_value


def test_budget_yields_in_small_slices_without_global_thread_setting_changes():
    now = [1.]
    sleeps = []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    work = ArchiveWork(clock=lambda: now[0], sleep=sleep)
    work.checkpoint()
    assert not sleeps
    now[0] += .002
    work.checkpoint()
    assert sleeps == [.001]
    assert work.diagnostics()['yield_wall_s'] > 0.


def test_chunked_json_preserves_nested_numpy_values_and_unicode(tmp_path):
    work = ArchiveWork()
    value = {'关节': np.arange(1000).reshape(100, 10),
             'feedback': [{'missing': float('nan'), 'ok': np.bool_(True)}]*2000}
    expected = json_value(value)
    path = tmp_path/'record.json'
    work.write_json(path, json_value(value, work=work))
    assert json.loads(path.read_text()) == expected
    assert work.yields > 0


def test_cooperative_trace_export_matches_normal_arrays_and_summary(tmp_path):
    trace = ExecutionTrace()
    for i in range(150):
        trace.record_feedback(dict(joints=[i*.001]*7, joint_stamp=i*.01,
                                   pose=[0.]*6), i*.01, i*.01+.001, stage='return')
    expected_arrays = trace.arrays()
    expected_summary = trace.summary(expected_arrays)
    result = trace.write(tmp_path)
    assert 'write_error' not in result
    with np.load(tmp_path/'execution_trace.npz', allow_pickle=False) as actual:
        assert set(actual.files) == set(expected_arrays)
        for key, array in expected_arrays.items():
            if array.dtype.kind in 'USb':
                np.testing.assert_array_equal(actual[key], array)
            else:
                np.testing.assert_allclose(actual[key], array, equal_nan=True)
    assert json.loads((tmp_path/'execution_trace_summary.json').read_text()) == expected_summary
    assert result['serialization_scheduling']['yields'] > 0
