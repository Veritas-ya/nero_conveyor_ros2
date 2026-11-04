import numpy as np

from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.nero7_occlusion_tracker import OcclusionAwareBlockTracker
from vendor.nero7_speed_reference.nero7_perception.conveyor_model import BeltModel


def detection(x, y=0.):
    return dict(u=400., v=400., centroid_base=np.array([x, y, .04]),
                height=.04, belt_z=0., width_px=30., area_px=800.)


def tracker():
    meter = Nero7BeltVelocityMeter(surface=None)
    meter._direction = np.array([1., 0.])
    meter._make_tracker()
    return meter._tracker


def test_short_camera_loss_keeps_id_without_inventing_observations():
    t = tracker()
    for i in range(31):
        t.update([detection(i*.001)], i/30.)
    old = t.tracks[0]
    count, stamp, observed = old.motion.accepted_count, old.stamp, old.observed_centroid.copy()
    for i in range(31, 43):  # 0.4 seconds: beyond old six-frame limit
        t.update([], i/30.)
    assert len(t.tracks) == 1 and old.id == t.tracks[0].id
    assert old.motion.accepted_count == count and old.stamp == stamp
    np.testing.assert_array_equal(old.observed_centroid, observed)
    t.update([detection(.043)], 43/30.)
    assert len(t.tracks) == 1 and t.tracks[0].id == old.id and old.misses == 0
    # A gap exceeding 0.3 seconds requires fresh speed observations.
    assert not old.speed_valid


def test_timeout_creates_new_identity_instead_of_reusing_expired_track():
    t = tracker()
    for i in range(31): t.update([detection(i*.001)], i/30.)
    old_id = t.tracks[0].id
    t.update([detection(.047)], 47/30.)
    assert len(t.tracks) == 1 and t.tracks[0].id != old_id


def test_merged_detection_cannot_silently_switch_or_create_an_id():
    t = OcclusionAwareBlockTracker(BeltModel((1., 0.)), {}, lost_frames=60)
    t.update([detection(0., -.005), detection(0., .005)], 0.)
    old = list(t.tracks)
    t.update([detection(.001, -.005), detection(.001, .005)], .1)
    before = [(v.id, v.stamp, v.observed_centroid.copy()) for v in old]
    # The merged detection is statistically possible for both retained IDs.
    # An unrelated extra detection must not disguise the shared component.
    t.update([detection(.002, 0.), detection(.25, .25)], .2)
    assert len(t.tracks) == 3 and t.ambiguous_frames == 1
    assert t.last_ambiguous['track_ids'] == [0, 1]
    for tr, (identity, stamp, position) in zip(old, before):
        assert tr.id == identity and tr.stamp == stamp and tr.misses == 1
        np.testing.assert_array_equal(tr.observed_centroid, position)
    t.update([detection(.003, .005), detection(.003, -.005)], .3)
    # Both assignments still fit the motion gates; do not guess identities.
    assert all(tr.misses == 2 for tr in old)
    assert all(tr.stamp == .1 for tr in old)


def test_split_candidate_is_not_arbitrarily_bound_to_lost_object():
    t = tracker()
    for i in range(10): t.update([detection(i*.001)], i/30.)
    t.update([], 10/30.)
    old = t.tracks[0]
    t.update([detection(.011, -.001), detection(.011, .001)], 11/30.)
    assert len(t.tracks) == 1 and old.misses == 2
    assert t.last_ambiguous['detection_count'] == 2


def test_sixty_fps_loss_keeps_track_for_configured_half_second():
    t = tracker()
    for i in range(61): t.update([detection(i*.0005)], i/60.)
    old = t.tracks[0]
    for i in range(61, 88): t.update([], i/60.)
    t.update([detection(.044)], 88/60.)
    assert len(t.tracks) == 1 and t.tracks[0] is old and old.misses == 0
