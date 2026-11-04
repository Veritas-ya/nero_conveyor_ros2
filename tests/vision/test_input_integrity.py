"""Input identity and freshness failures, without object truth or MuJoCo."""
import copy
import json
import math

import numpy as np
import pytest

from experiments.graspgenx_sim.input_integrity import (
    build_input_provenance,
    select_captured_request,
    validate_candidate_admission,
)


def capture():
    depth = np.ones((6, 8), dtype=np.float32)
    labels = np.full(depth.shape, -1, dtype=np.int32)
    labels[2:4, 3:5] = 7
    # Independent geometric fixture: K maps these four GL rays through a
    # translated camera, without calling the provenance deprojector.
    K = np.array([[10., 0., 3.5], [0., 10., 2.5], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, 3] = [1., 2., 3.]
    cloud = np.array([[.95, 2.05, 2.], [1.05, 2.05, 2.],
                      [.95, 1.95, 2.], [1.05, 1.95, 2.]], dtype=np.float32)
    frame = {"timestamp": 2., "top_depth": depth, "top_instance_labels": labels,
             "top_camera": {"intrinsics": K, "camera_to_world": T,
                            "width": 8, "height": 6, "distortion": [0.] * 5,
                            "preprocessing": "native pinhole; no resize"}}
    # Its visible surface centroid is intentionally not the visual body center.
    observed = {"object_id": 7, "position": np.array([1., 2., 1.9]),
                "linear_velocity": np.array([.1, 0., 0.]),
                "quaternion": [1., 0., 0., 0.], "timestamp": 2.,
                "visible": True, "track_observation_age": 0.,
                "pose_source": "rgbd_depth_components"}
    return frame, cloud, observed


def provenance():
    frame, cloud, observed = capture()
    return build_input_provenance(frame, 7, cloud, observed, 9, 2.02)


def live(now=3.):
    # publication time does not make the last visual measurement new.
    return {"object_id": 7, "position": [1.1, 2., 1.9],
            "quaternion": [1., 0., 0., 0.], "timestamp": now,
            "observed_timestamp": 2.8, "track_observation_age": .15,
            "visible": False}


def admit(record=None, track=None, **kwargs):
    return validate_candidate_admission(record or provenance(), live() if track is None else track,
                                        9, 3., max_observation_age_s=4.8,
                                        max_result_age_s=4.8, **kwargs)


def test_cloud_display_and_tracker_provenance_exact_capture():
    record = provenance()
    assert record["input_valid"], record["validation_reasons"]
    assert record["mask_pixels"] == record["cloud_point_count"] == 4
    assert record["bbox_xyxy"] == [3, 2, 5, 4]
    assert record["depth_mask_projection_max_error_m"] < 2e-7
    assert record["cloud_storage_dtype"] == "float32"
    assert record["centroid_to_track_center_m"] == pytest.approx(.1)
    assert record["last_visual_measurement_sim_s"] == 2.
    assert record["scene_epoch"] == 9
    json.dumps(record, allow_nan=False)


def test_nonlast_selected_track_keeps_its_own_capture_record_and_admission():
    frame, cloud, observed = capture()
    # Two actual visual identities share one capture. The upstream identity is
    # encountered last, but the existing policy chooses the downstream one.
    frame["top_instance_labels"][0:2, 0:2] = 8
    upstream = {**copy.deepcopy(observed), "object_id": 8,
                "position": np.array([.75, 2.15, 1.9])}
    requested = [(observed["position"][0], 7, observed["position"].copy(), observed),
                 (upstream["position"][0], 8, upstream["position"].copy(), upstream)]
    _, identifier, position, selected_observed = select_captured_request(requested)
    assert identifier == 7
    assert selected_observed is observed
    np.testing.assert_array_equal(position, observed["position"])
    record = build_input_provenance(frame, identifier, cloud, selected_observed, 9, 2.02)
    assert record["input_valid"], record["validation_reasons"]
    accepted = validate_candidate_admission(record, live(), 9, 3., max_observation_age_s=4.8)
    assert accepted["allowed"], accepted["reasons"]
    # This is the previous runner bug: using the last loop variable while
    # keeping the selected downstream mask and cloud. It is safely rejected.
    wrong = build_input_provenance(frame, identifier, cloud, upstream, 9, 2.02)
    assert "capture_visual_track_id_mismatch" in wrong["validation_reasons"]
    assert not validate_candidate_admission(wrong, live(), 9, 3., max_observation_age_s=4.8)["allowed"]


def test_capture_selection_preserves_maximum_x_and_first_tie_without_new_rules():
    first = (.1, 7, np.array([.1, 0, 0]), {"object_id": 7})
    tied = (.1, 8, np.array([.1, 0, 0]), {"object_id": 8})
    upstream = (-.2, 999, np.array([-.2, 0, 0]), {"object_id": 999})
    assert select_captured_request([first, tied, upstream]) is first


@pytest.mark.parametrize("kind,reason", [
    ("wrong_cloud", "cloud_does_not_match_capture_depth_mask"),
    ("wrong_cloud_count", "cloud_point_count_differs_from_capture_mask"),
    ("wrong_visual_id", "capture_visual_track_id_mismatch"),
    ("old_visual_timestamp", "capture_visual_frame_timestamp_mismatch"),
    ("absent_mask", "track_absent_from_capture_mask"),
    ("old_coasted_visual", "capture_mask_belongs_to_coasted_track"),
    ("depth_nan", "capture_mask_contains_invalid_depth"),
    ("cloud_nan", "nonfinite_or_invalid_cloud"),
    ("cloud_strings", "nonfinite_or_invalid_cloud"),
    ("raster_mismatch", "depth_mask_raster_mismatch"),
    ("bad_K", "invalid_capture_camera_calibration"),
    ("resolution_mismatch", "camera_resolution_differs_from_depth_raster"),
    ("frame_epoch", "capture_frame_scene_epoch_mismatch"),
])
def test_input_mismatch_is_explicit_not_silent(kind, reason):
    frame, cloud, observed = capture()
    if kind == "wrong_cloud": cloud += [.02, 0., 0.]
    elif kind == "wrong_cloud_count": cloud = cloud[:-1]
    elif kind == "wrong_visual_id": observed["object_id"] = 8
    elif kind == "old_visual_timestamp": observed["timestamp"] = 1.8
    elif kind == "absent_mask": frame["top_instance_labels"][:] = -1
    elif kind == "old_coasted_visual": observed["visible"] = False
    elif kind == "depth_nan": frame["top_depth"][2, 3] = np.nan
    elif kind == "cloud_nan": cloud[0, 0] = np.nan
    elif kind == "cloud_strings": cloud = [["bad"] * 3] * 4
    elif kind == "raster_mismatch": frame["top_instance_labels"] = np.zeros((3, 4))
    elif kind == "bad_K": frame["top_camera"]["intrinsics"][0, 0] = 0.
    elif kind == "resolution_mismatch": frame["top_camera"]["width"] = 16
    elif kind == "frame_epoch": frame["scene_epoch"] = 10
    result = build_input_provenance(frame, 7, cloud, observed, 9, 2.02)
    assert not result["input_valid"]
    assert reason in result["validation_reasons"]
    json.dumps(result, allow_nan=False)


def test_brief_occlusion_preserves_identity_and_logs_real_capture_age():
    result = admit()
    assert result["allowed"], result["reasons"]
    assert not result["live_visible"]
    assert result["live_publication_timestamp_s"] == 3.
    assert result["live_observed_timestamp_s"] == 2.8
    assert result["live_measurement_age_s"] == pytest.approx(.35)
    assert result["result_age_s"] == 1.
    assert result["prediction_residual_m"] < 1e-12
    json.dumps(result, allow_nan=False)


def test_live_publication_timestamp_never_makes_old_measurement_fresh():
    track = live()
    track["observed_timestamp"] = -2.
    track["track_observation_age"] = 0.
    result = admit(track=track)
    assert not result["allowed"]
    assert result["live_measurement_age_s"] == 5.
    assert "live_visual_measurement_expired" in result["reasons"]


def test_sensor_boundary_total_age_not_counted_twice():
    track = live()
    track.update(measurement_timestamp=2.65, observed_timestamp=2.8,
                 track_observation_age=.35)
    result = admit(track=track)
    assert result["allowed"]
    assert result["live_measurement_age_s"] == pytest.approx(.35)
    assert result["live_age_basis"] == "last_actual_measurement_timestamp"


def test_live_identity_and_scene_epoch_cannot_rebind_a_candidate():
    record = provenance()
    track = live()
    track["object_id"] = 8
    result = validate_candidate_admission(record, track, 10, 3., max_observation_age_s=4.8)
    assert not result["allowed"]
    assert "scene_epoch_changed" in result["reasons"]
    assert "live_visual_track_id_mismatch" in result["reasons"]


def test_optional_generation_guards_identity_reuse_within_same_epoch():
    record = provenance()
    record["track_generation"] = "first"
    track = live()
    track["generation"] = "second"
    result = admit(record, track)
    assert "visual_track_generation_changed" in result["reasons"]
    del track["generation"]
    assert "live_visual_track_generation_unavailable" in admit(record, track)["reasons"]


def test_motion_and_rotation_residuals_are_logged_without_arbitrary_gates():
    track = live()
    track["position"] = [1.2, 2.02, 1.9]
    track["quaternion"] = [math.cos(math.pi / 12), 0., 0., math.sin(math.pi / 12)]
    result = admit(track=track)
    assert result["allowed"]
    assert result["capture_to_live_rotation_deg"] == pytest.approx(30.)
    assert result["prediction_residual_m"] == pytest.approx(math.hypot(.1, .02))
    assert not admit(track=track, max_prediction_residual_m=.05)["allowed"]
    track["quaternion"] = [-1., 0., 0., 0.]
    assert admit(track=track)["capture_to_live_rotation_deg"] == 0.


def test_missing_visual_track_and_expired_result_are_explicit():
    result = validate_candidate_admission(provenance(), None, 9, 7.,
                                          max_observation_age_s=4.8, max_result_age_s=4.8)
    assert not result["allowed"]
    assert result["reasons"] == ["inference_result_expired", "visual_track_missing_at_admission"]


def test_upstream_target_is_valid_until_it_passes_workspace_end():
    track = live()
    track["position"] = [-.4, 0., .5]
    assert admit(track=track, latest_allowed_x_m=.25)["allowed"]
    track["position"][0] = .251
    assert "visual_track_passed_grasp_workspace" in admit(
        track=track, latest_allowed_x_m=.25)["reasons"]


def test_only_reported_tracker_age_is_labelled_as_lower_bound():
    track = live()
    del track["observed_timestamp"]
    result = admit(track=track)
    assert result["allowed"]
    assert "lower_bound" in result["live_age_basis"]
    assert result["live_observed_timestamp_s"] is None
    del track["track_observation_age"]
    assert "live_visual_measurement_age_unavailable_or_invalid" in admit(track=track)["reasons"]


def test_audit_values_are_copied_not_later_mutated():
    frame, cloud, observed = capture()
    record = build_input_provenance(frame, 7, cloud, observed, 9, 2.02)
    saved = copy.deepcopy(record)
    cloud[:] = 99.
    observed["position"][:] = 99.
    frame["top_camera"]["intrinsics"][:] = 0.
    assert record == saved


@pytest.mark.parametrize("key,value", [("max_result_age_s", np.nan),
                                      ("max_observation_age_s", -.1),
                                      ("max_prediction_residual_m", np.inf),
                                      ("latest_allowed_x_m", np.nan)])
def test_invalid_limits_cannot_silently_disable_freshness_validation(key, value):
    parameters = {"max_observation_age_s": 4.8, key: value}
    with pytest.raises(ValueError, match=key):
        validate_candidate_admission(provenance(), live(), 9, 3., **parameters)
