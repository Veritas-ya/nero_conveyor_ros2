"""Receipt-time grouping uses saved 11:15 incident values without hardware."""
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy, transform_distance
from mujoco_conveyor_sorting.runtime.coherent_flange import CoherentFlangeCache, POSE_PACKET_NAMES
from mujoco_conveyor_sorting.runtime.single_pick import RobotModel


def parts(stamps, *, epoch=0):
    return SimpleNamespace(**{name: SimpleNamespace(timestamp=stamp, msg=None,
        feedback_time=SimpleNamespace(mapped_monotonic=stamp, clock_epoch=epoch,
                                      mapping_quality='mapped'))
        for name, stamp in zip(POSE_PACKET_NAMES, stamps)})


def test_recorded_mixed_packet_reuses_previously_captured_group():
    cache = CoherentFlangeCache(); cache.observe_epoch(0)
    before = np.array([-.380337, -.038435, .153143,
                       *np.deg2rad([-73.123, 87.344, -114.149])])
    mixed = np.array([-.380507, -.038317, .153351,
                      *np.deg2rad([-74.519, 87.344, -114.149])])
    prior_stamps = [68562.4574713707, 68562.45774388, 68562.45802283287]
    mixed_stamps = [68562.48046255112, 68562.48069930077, 68562.45802283287]
    parser = parts(prior_stamps)
    assert cache.capture(parser, lambda: SimpleNamespace(msg=before), 68562.4581)
    parser = parts(mixed_stamps)
    assert not cache.capture(parser, lambda: SimpleNamespace(msg=mixed), 68562.4808)
    group = cache.select(68562.4808)
    np.testing.assert_allclose(group.pose, before)
    np.testing.assert_allclose([p.feedback_time.mapped_monotonic for p in group.parts], prior_stamps)
    model = RobotModel()
    q = [0.06431538293599104, 0.791978054677467, -0.009711966813900692,
         1.6292553553361664, -0.6457145725474475, -0.5073618042775825, 0.5057316023073136]
    model.set_state(q, .1)
    assert transform_distance(model.flange(), pose_xyz_rpy(group.pose))[1] == pytest.approx(.0184, abs=.001)
    assert transform_distance(model.flange(), pose_xyz_rpy(mixed))[1] == pytest.approx(1.3782, abs=.001)


def test_cache_does_not_mutate_captured_group_or_refresh_its_clock():
    cache = CoherentFlangeCache(); cache.observe_epoch(1)
    parser = parts([1., 1.001, 1.002], epoch=1)
    pose = [1., 2., 3., .1, .2, .3]
    assert cache.capture(parser, lambda: SimpleNamespace(msg=pose), 1.003)
    pose[0] = 999.
    parser.end_pose_xy.timestamp = 9.
    assert cache.select(1.1).pose[0] == 1.
    assert cache.select(1.1).parts[0].timestamp == 1.
    assert cache.select(1.201) is None


def test_group_rejects_missing_mixed_epoch_unmapped_and_invalid_pose():
    cache = CoherentFlangeCache(); cache.observe_epoch(2)
    parser = parts([1., 1.001, 1.002], epoch=2)
    good_pose = lambda: SimpleNamespace(msg=[0.]*6)
    parser.end_pose_xy.feedback_time.clock_epoch = 1
    assert not cache.capture(parser, good_pose, 1.003)
    parser.end_pose_xy.feedback_time.clock_epoch = 2
    parser.end_pose_zrx.feedback_time.mapping_quality = 'ambiguous_epoch'
    assert not cache.capture(parser, good_pose, 1.003)
    parser.end_pose_zrx.feedback_time.mapping_quality = 'mapped'
    assert not cache.capture(parser, lambda: SimpleNamespace(msg=[float('nan')]*6), 1.003)
    del parser.end_pose_zrx
    assert not cache.capture(parser, good_pose, 1.003)


def test_new_epoch_immediately_invalidates_group():
    cache = CoherentFlangeCache(); cache.observe_epoch(0)
    parser = parts([1., 1.001, 1.002])
    assert cache.capture(parser, lambda: SimpleNamespace(msg=[0.]*6), 1.003)
    cache.observe_epoch(1)
    assert cache.select(1.004) is None
    assert not cache.capture(parser, lambda: SimpleNamespace(msg=[0.]*6), 1.004)
