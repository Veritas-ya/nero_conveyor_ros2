"""Known-speed ID association with synthetic real-frame observations, no devices."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.nero7_occlusion_tracker import OcclusionAwareBlockTracker
from vendor.nero7_speed_reference.nero7_perception.conveyor_model import BeltModel
from nero_core.two_second.config import CellConfig
from nero_core.two_second.perception import make_meter
from nero_core.two_second.velocity_source import (
    FIXED, policy, prediction_velocity, speed_report, with_fixed_belt_speed)


CONFIG = Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml'


def detection(position):
    return dict(u=400., v=400., centroid_base=np.asarray(position, dtype=float),
        height=.04, belt_z=0., width_px=30., area_px=800.)


def meter_for(velocity=None):
    meter = Nero7BeltVelocityMeter(None, maximum_measured_speed_m_s=1.,
        association_velocity_base_m_s=velocity)
    meter._direction = np.array([0., -1.])
    meter._epoch = 0
    meter._make_tracker()
    return meter


@pytest.mark.parametrize('speed', [.1727, .435, .507, 1.])
@pytest.mark.parametrize('capture_start', [0., 100000.])
def test_known_speed_keeps_one_id_and_confirms_only_after_two_distinct_real_observations(speed, capture_start):
    velocity = np.array([0., -speed, 0.])
    meter = meter_for(velocity)
    for i in range(30):
        position = np.array([.1, 0., .04])+velocity*(i/30.)
        meter._tracker.update([detection(position)], capture_start+i/30.)
        track, = meter._tracker.tracks
        assert track.id == 0 and track.hits == i+1
        assert track.confirmed is (i >= 1)
        np.testing.assert_array_equal(track.observed_centroid, position)
        np.testing.assert_allclose(track.centroid, position, atol=1e-10)
        assert track.motion.accepted_count == i+1
        assert not track.speed_valid  # Known-speed association is not visual speed measurement.
    assert meter._tracker._next_id == 1


def test_predictions_missing_frames_and_duplicate_timestamps_never_confirm_an_id():
    tracker = meter_for([0., -.435, 0.])._tracker
    first = np.array([.1, 0., .04])
    tracker.update([detection(first)], 0.)
    track, = tracker.tracks
    for index in range(1, 10):
        tracker.update([], index/30.)
        assert not track.confirmed and track.hits == track.motion.accepted_count == 1
        assert track.stamp == track.motion.observation_stamp == 0.
        np.testing.assert_array_equal(track.observed_centroid, first)
    tracker.update([detection(first+[0., -.435*.3, 0.])], .3)  # Duplicate last timestamp.
    tracker.update([detection(first)], .1)  # Out of order.
    assert track.hits == 1 and not track.confirmed
    position = first+np.array([0., -.435/3., 0.])
    tracker.update([detection(position)], 1/3.)
    assert tracker.tracks == [track] and track.confirmed and track.hits == 2
    np.testing.assert_array_equal(track.observed_centroid, position)


def test_expired_id_is_not_resurrected_by_a_correct_belt_prediction():
    tracker = meter_for([0., -.435, 0.])._tracker
    tracker.update([detection([0., 0., .04])], 0.)
    tracker.update([detection([0., -.435*.6, .04])], .6)
    track, = tracker.tracks
    assert track.id == 1 and not track.confirmed and track.hits == 1


def test_bad_real_detection_cannot_move_or_refresh_an_existing_id():
    tracker = meter_for([0., -.435, 0.])._tracker
    for i in range(3):
        tracker.update([detection([0., -.435*i/30., .04])], i/30.)
    track, = tracker.tracks
    position = track.observed_centroid.copy()
    tracker.update([detection([.1, -.435*.1, .04])], .1)  # Cross-belt jump.
    assert track.hits == 3 and track.misses == 1 and track.stamp == pytest.approx(2/30.)
    np.testing.assert_array_equal(track.observed_centroid, position)


def test_known_speed_does_not_guess_between_merged_objects():
    tracker = OcclusionAwareBlockTracker(BeltModel((1., 0.)), {}, lost_frames=60,
        association_velocity_xy=[.435, 0.])
    tracker.update([detection([0., -.005, .04]), detection([0., .005, .04])], 0.)
    tracker.update([detection([.0145, -.005, .04]), detection([.0145, .005, .04])], 1/30.)
    tracks = list(tracker.tracks)
    assert all(track.confirmed for track in tracks)
    tracker.update([detection([.029, 0., .04])], 2/30.)
    assert tracker.ambiguous_frames == 1 and tracker.last_ambiguous['track_ids'] == [0, 1]
    assert all(track.hits == 2 and track.misses == 1 for track in tracks)


def test_fixed_association_report_exposes_real_position_and_never_claims_measured_speed():
    meter = meter_for([0., -.435, 0.])
    for index in range(30):
        position = [.1, -.435*index/30., .04]
        meter._tracker.update([detection(position)], index/30.)
    meter._sequence = 29
    report = meter._report(29/30., current=True)
    track, = meter.latest_snapshot.tracks
    assert report['status'] == 'tracking'
    assert report['motion_source'] == track.velocity_source == FIXED
    assert not report['speed_valid'] and report['velocity_base_m_s'] is None
    assert track.velocity_base_m_s is track.legacy_velocity_base_m_s is None
    assert 1 < track.sample_count < 30  # Only genuine observations in the recent 0.6s window.
    assert .5 < track.span_s <= .6+1e-9
    assert report['diagnostics']['association_prediction_source'] == FIXED
    np.testing.assert_allclose(report['diagnostics']['association_velocity_base_m_s'], [0., -.435, 0.])
    assert report['diagnostics']['tracks'][0]['accepted_count'] == 30
    assert report['diagnostics']['tracks'][0]['velocity_base_m_s'] is None
    np.testing.assert_array_equal(track.observed_position_base, position)


def test_factory_uses_the_same_tilted_surface_velocity_as_grasp_prediction_only_when_fixed():
    config = CellConfig.load(CONFIG)
    normal = np.array([.02, -.03, np.sqrt(1.-.02**2-.03**2)])
    surface = SimpleNamespace(normal_base=normal, bounds_xy_m=np.array([[-1., 1.], [-1., 1.]]))
    fixed = with_fixed_belt_speed(config, .435)
    meter = make_meter(fixed, surface)
    expected = prediction_velocity(fixed, None, normal)
    np.testing.assert_allclose(meter.association_velocity_base_m_s, expected)
    assert np.linalg.norm(expected) == pytest.approx(.435)
    assert expected@normal == pytest.approx(0., abs=1e-12)
    assert make_meter(config, surface).association_velocity_base_m_s is None
    assert policy(fixed)['association_prediction_source'] == FIXED
    assert not policy(fixed)['visual_velocity_used']
    report = speed_report(fixed, SimpleNamespace(sample_count=2), expected)
    assert report['measured_velocity_base_m_s'] is None and not report['visual_velocity_used']


def test_new_generation_drops_old_ids_and_retains_the_explicit_velocity():
    meter = meter_for([0., -.435, 0.])
    meter._tracker.update([detection([0., 0., .04])], 0.)
    generation = meter._generation
    meter.reset()
    meter._make_tracker()
    meter._tracker.update([detection([0., 0., .04])], 0.)
    track, = meter._tracker.tracks
    assert track.id == 1 and not track.confirmed and meter._generation > generation
    np.testing.assert_allclose(track.motion.forecast(.1)[0][:2], [0., -.0435])


@pytest.mark.parametrize('fps', [15, 30, 60])
def test_variable_frame_gaps_and_depth_position_noise_preserve_observation_identity(fps):
    velocity = np.array([-.012393448040960157, -.43479489017035516, -.004980554929409665])
    tracker = meter_for(velocity)._tracker
    generator = np.random.default_rng(4)
    stamps = np.cumsum(generator.integers(1, 4, size=25)/fps)
    for i, stamp in enumerate(stamps):
        point = np.array([-.35, .7, -.06])+velocity*stamp
        point[:2] += generator.uniform(-.0005, .0005, 2)
        original = point.copy()
        tracker.update([detection(point)], stamp)
        track, = tracker.tracks
        assert track.id == 0 and track.hits == i+1 and track.misses == 0
        assert track.stamp == stamp
        np.testing.assert_array_equal(point, original)
        np.testing.assert_array_equal(track.observed_centroid, point)
    assert tracker._next_id == 1


@pytest.mark.parametrize('velocity', [[0., 0., 0.], [.1, 0.], [np.nan, 0., 0.], [0., -.1, np.inf]])
def test_invalid_prior_cannot_enable_association(velocity):
    with pytest.raises(ValueError):
        meter_for(velocity)


def test_visual_pointcloud_speed_cannot_be_mixed_with_an_operator_association_prior():
    with pytest.raises(ValueError):
        Nero7BeltVelocityMeter(None, pointcloud_speed=True, association_velocity_base_m_s=[0., -.435, 0.])
