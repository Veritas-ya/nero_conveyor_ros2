"""Candidate-local depth rejection with real detector, admission and tracking."""
from types import SimpleNamespace

import numpy as np
import pytest

from test_workpiece_admission import fixture as admission_fixture, component
from mujoco_conveyor_sorting.runtime.belt_surface import _geometry_fingerprint
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter
from mujoco_conveyor_sorting.runtime.nero7_boundary_depth import isolate_candidate_depth_gaps
from mujoco_conveyor_sorting.runtime.workpiece_admission import MeasuredWorkpieceAdmission
from vendor.nero7_speed_reference.nero7_perception.base_plane import BaseBeltPlane
from test_boundary_depth_gaps import example


def case(*, good=True, second_good=False, good_height=.04):
    surface,camera,T,depth,mask,frame,params=admission_fixture()
    # An interior unknown gray region: original detector would abort the frame.
    component(camera,T,depth,mask,frame,(25,20,20,20))
    depth[20:40,25:45]=np.nan
    depth[25,30]=.96
    if good:component(camera,T,depth,mask,frame,(65,45,25,25),z=1.-good_height)
    if second_good:component(camera,T,depth,mask,frame,(110,45,20,20))
    intr=SimpleNamespace(matrix=camera.K,width=160,height=120,distortion=(0.,)*5)
    frame.depth_m=depth
    frame.depth_intrinsics=frame.color_intrinsics=intr
    frame.camera_to_world_cv=T
    frame.header=SimpleNamespace(sequence=1,source_time=1.,clock_epoch=0)
    surface.geometry_fingerprint=_geometry_fingerprint(frame,np.eye(4))
    meter=Nero7BeltVelocityMeter(surface,belt_direction_base=[1.,0.,0.],
        maximum_measured_speed_m_s=1.,detection_mode='rgb_depth',
        green_belt_extractor=lambda *args:pytest.fail('synthetic plane already calibrated'),
        association_velocity_base_m_s=[.2,0.,0.])
    meter.parameters['isolate_boundary_depth_gaps']=True
    # This fixture has a known synthetic empty-belt plane, not a live calibration.
    meter._fused.plane=BaseBeltPlane(np.array([0.,0.,1.]),0.,'ransac3d',{})
    meter.candidate_filter=MeasuredWorkpieceAdmission(surface,.12).filter
    return meter,frame


def advance(frame):
    frame.header.sequence+=1
    frame.header.source_time+=1./30.


def test_interior_candidate_rejected_locally_without_changing_native_pixels():
    color,depth,params=example()
    depth[50:70,50:70]=np.nan
    original_color=color.copy();original_depth=depth.copy()
    out,rejected=isolate_candidate_depth_gaps(color,depth,params)
    assert {r['reason'] for r in rejected}=={'roi_boundary','insufficient_interior_depth'}
    assert next(r for r in rejected if r['reason']=='insufficient_interior_depth')['valid_pixels']==0
    assert not out[50:70,50:70].any()
    np.testing.assert_array_equal(color,original_color)
    np.testing.assert_array_equal(depth,original_depth)


@pytest.mark.parametrize('valid_pixels,ratio_passes',[(161,False),(162,True)])
def test_original_fifty_percent_requirement_is_preserved(valid_pixels,ratio_passes):
    color,depth,params=example()
    depth[20:40,5:25]=1.
    depth[50:70,50:70]=np.nan
    rows,cols=np.indices((18,18));selected=np.arange(324)<valid_pixels
    depth[rows.ravel()[selected]+51,cols.ravel()[selected]+51]=.96
    _,rejected=isolate_candidate_depth_gaps(color,depth,params)
    interior=[r for r in rejected if r['reason']=='insufficient_interior_depth']
    assert bool(interior) is not ratio_passes
    if interior:assert interior[0]['valid_pixels']==valid_pixels


def test_good_part_survives_unknown_interior_blob_and_needs_two_real_observations():
    meter,frame=case()
    rgb=frame.rgb.copy();depth=frame.depth_m.copy()
    report=meter.update(frame,np.eye(4))
    diag=report['diagnostics']['candidate_depth_diagnostics']['rejected_interior_depth_components']
    assert len(diag)==1 and diag[0]['valid_pixels']==1
    assert diag[0]['reason']=='insufficient_interior_depth'
    assert report['diagnostics']['depth_admission']['accepted_candidates']==1
    assert meter.latest_snapshot.current
    track=meter.latest_snapshot.tracks[0]
    assert not track.confirmed and track.sample_count==1
    first_id=track.track_id
    # A duplicate frame cannot replace the second real observation.
    meter.update(frame,np.eye(4))
    assert not meter.latest_snapshot.current
    assert meter._tracker.tracks[0].hits==1
    advance(frame)
    meter.update(frame,np.eye(4))
    tracks=meter.latest_snapshot.tracks
    assert meter.latest_snapshot.current and len(tracks)==1
    assert tracks[0].track_id==first_id and tracks[0].confirmed and tracks[0].sample_count==2
    np.testing.assert_array_equal(frame.rgb,rgb)
    np.testing.assert_array_equal(frame.depth_m,depth)


def test_independent_good_measurement_and_tracker_state_match_clean_frame():
    mixed,frame=case();clean,baseline=case()
    baseline.rgb[20:40,25:45]=[20,140,20]
    for _ in range(3):
        mixed.update(frame,np.eye(4));clean.update(baseline,np.eye(4))
        assert len(mixed._detections)==len(clean._detections)==1
        np.testing.assert_array_equal(mixed._detections[0]['centroid_base'],clean._detections[0]['centroid_base'])
        np.testing.assert_array_equal(mixed._tracker.tracks[0].motion.state,clean._tracker.tracks[0].motion.state)
        assert mixed._tracker.tracks[0].hits==clean._tracker.tracks[0].hits
        advance(frame);advance(baseline)


def test_only_unobservable_candidates_do_not_publish_empty_belt_or_allocate_ids():
    meter,frame=case(good=False)
    report=meter.update(frame,np.eye(4))
    assert not meter.latest_snapshot.current and not meter.latest_snapshot.tracks
    assert 'insufficient interior depth' in report['reason']
    assert not meter._tracker.tracks and meter._tracker._next_id==0


def test_bad_candidate_and_flat_patch_do_not_bypass_measured_workpiece_admission():
    meter,frame=case(good_height=0.)
    report=meter.update(frame,np.eye(4))
    assert not meter.latest_snapshot.current and not meter._tracker.tracks
    assert 'insufficient interior depth' in report['reason']


def test_unknown_frame_does_not_confirm_an_existing_track_with_prediction():
    meter,frame=case()
    meter.update(frame,np.eye(4))
    frame.rgb[45:70,65:90]=[20,140,20]
    advance(frame);meter.update(frame,np.eye(4))
    assert not meter.latest_snapshot.current
    assert meter._tracker.tracks[0].hits==1
    assert not meter._tracker.tracks[0].confirmed


def test_two_valid_parts_remain_two_distinct_candidates_for_owner_ambiguity_check():
    meter,frame=case(second_good=True)
    meter.update(frame,np.eye(4));advance(frame)
    report=meter.update(frame,np.eye(4))
    tracks=meter.latest_snapshot.tracks
    assert report['diagnostics']['depth_admission']['accepted_candidates']==2
    assert len(tracks)==2 and len({t.track_id for t in tracks})==2
    assert all(t.confirmed for t in tracks)


def test_disabled_isolation_keeps_original_full_frame_rejection():
    meter,frame=case()
    meter.parameters['isolate_boundary_depth_gaps']=False
    report=meter.update(frame,np.eye(4))
    assert not meter.latest_snapshot.current and not meter._tracker.tracks
    assert 'insufficient interior depth' in report['reason']
