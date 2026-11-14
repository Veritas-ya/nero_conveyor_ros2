"""Merged RGB belt reflections are split only using same-frame depth."""
import numpy as np

from test_candidate_depth_isolation import case, advance
from mujoco_conveyor_sorting.runtime.nero7_height_split import split_mixed_belt_candidates
from vendor.nero7_speed_reference.nero7_perception.detection import load_detection_params
from pathlib import Path


def merged_frame():
    meter, frame = case(good=False)
    # Remove the unrelated unknown blob from the shared fixture.
    frame.rgb[20:40,25:45] = [20,140,20]
    frame.depth_m[20:40,25:45] = 1.
    # One connected gray RGB component: a raised object attached to a large
    # same-colored flat belt patch. Both retain real, distinct depth values.
    frame.rgb[45:70,25:90] = [150,150,150]
    frame.depth_m[45:70,25:90] = 1.
    frame.depth_m[45:70,65:90] = .96
    return meter, frame


def test_measured_height_splits_belt_reflection_and_preserves_id_and_raw_frames():
    baseline, frame = merged_frame()
    updated, replacement = merged_frame()
    updated.parameters['split_mixed_belt_candidates'] = True
    rgb, depth = replacement.rgb.copy(), replacement.depth_m.copy()
    old = baseline.update(frame, np.eye(4))
    assert old['diagnostics']['depth_admission']['accepted_candidates'] == 0
    report = updated.update(replacement, np.eye(4))
    assert report['diagnostics']['depth_admission']['accepted_candidates'] == 1
    assert len(report['diagnostics']['candidate_depth_diagnostics']['split_mixed_belt_candidates']) == 1
    original_id = updated.latest_snapshot.tracks[0].track_id
    advance(replacement)
    updated.update(replacement, np.eye(4))
    track = updated.latest_snapshot.tracks[0]
    assert track.confirmed and track.track_id == original_id
    np.testing.assert_array_equal(replacement.rgb, rgb)
    np.testing.assert_array_equal(replacement.depth_m, depth)


def test_clean_candidate_is_unchanged_and_flat_belt_never_becomes_target():
    p=load_detection_params(Path(__file__).parents[1]/'src/nero_core/vendor/nero7_speed_reference/upstream_perception.yaml')
    p['roi_px']=[0,0,160,120]
    color=np.zeros((120,160,3),np.uint8)
    color[40:65,55:80]=150
    depth=np.ones((120,160),np.float32)
    height=np.zeros(depth.shape,np.float32)
    height[40:65,55:80]=.04
    clean, decisions=split_mixed_belt_candidates(color,depth,height,p)
    assert not decisions
    np.testing.assert_array_equal(clean,color)
    height[:]=0
    flat, decisions=split_mixed_belt_candidates(color,depth,height,p)
    assert not decisions
    np.testing.assert_array_equal(flat,color)


def test_unknown_depth_is_not_filled_during_split():
    p=load_detection_params(Path(__file__).parents[1]/'src/nero_core/vendor/nero7_speed_reference/upstream_perception.yaml')
    p['roi_px']=[0,0,160,120]
    color=np.zeros((120,160,3),np.uint8);color[40:65,25:90]=150
    depth=np.ones((120,160),np.float32);height=np.zeros(depth.shape,np.float32)
    height[40:65,65:90]=.04
    depth[45:55,45:55]=np.nan
    original=depth.copy()
    out, decisions=split_mixed_belt_candidates(color,depth,height,p)
    assert len(decisions)==1 and decisions[0]['unobserved_depth_pixels']==100
    assert not out[45:55,45:55].any()
    np.testing.assert_array_equal(depth,original)


def test_two_measured_objects_do_not_become_one_target_when_belt_connects_them():
    meter, frame = merged_frame()
    meter.parameters['split_mixed_belt_candidates'] = True
    frame.rgb[45:70,25:130] = [150,150,150]
    frame.depth_m[45:70,25:130] = 1.
    frame.depth_m[45:70,40:65] = .96
    frame.depth_m[45:70,90:115] = .96
    report = meter.update(frame, np.eye(4))
    assert report['diagnostics']['depth_admission']['accepted_candidates'] == 2
    assert len(meter.latest_snapshot.tracks) == 2


def test_unobserved_split_does_not_turn_rejected_frame_into_empty_belt():
    meter, frame = merged_frame()
    meter.parameters['split_mixed_belt_candidates'] = True
    frame.depth_m[48:58,45:55] = np.nan
    meter.candidate_filter = lambda detections, *args: ([], {'accepted_candidates': 0})
    report = meter.update(frame, np.eye(4))
    assert not meter.latest_snapshot.current
    assert 'split candidate contains unobserved depth' in report['reason']
