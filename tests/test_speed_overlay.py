"""Synthetic display/transport checks; no SDK, CAN, physical motion, or new speed estimator."""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import json
import time
import numpy as np
import pytest

from test_collision_visualization import fixture
from nero_core.two_second.speed_overlay import live_speed_snapshot,frozen_speed_snapshot,paired_track_labels,speed_lines
from nero_core.two_second.visual_archive import VisualArchive
from nero_core.two_second.run_preview import RunPreviewWorker


def target_fixture():
    p,frame=fixture()
    t=p.target
    t.entry_position=np.array([0.,0.,1.]);t.velocity=np.array([0.,-.06,0.])
    t.speed_report={'source':'nero7_xy_kalman','sample_count':13,'span_s':.467,'speed_std_m_s':.002}
    t.grasps=tuple(np.eye(4) for _ in range(4))
    t.detection.update(obb={'center_world_m':[0.,0.,1.],'axes_world_columns':np.eye(3),'extent_m':[.02,.03,.04]},
        belt_region={'plane_to_base':np.eye(4)})
    return t,frame


def peer(frame, *, valid=True):
    track=SimpleNamespace(track_id=2,observed_position_base=(0.,0.,1.),velocity_base_m_s=(.02,-.0815,0.) if valid else None,
        speed_valid=valid,confirmed=True,misses=0,sample_count=13,span_s=.467)
    return SimpleNamespace(current=True,tracks=(track,),clock_epoch=frame.header.clock_epoch,track_generation=1)


def test_live_display_uses_original_filter_vector_and_actual_belt_direction():
    _,frame=target_fixture()
    live=live_speed_snapshot(frame,peer(frame),None,np.array([0.,-1.,0.]),1.)
    assert live['tracks'][0]['along_belt_m_s']==pytest.approx(.0815)
    # Warm-up estimates may be displayed, but must retain the unqualified label.
    warm=live_speed_snapshot(frame,peer(frame,valid=False),{'diagnostics':{'tracks':[
        {'id':2,'velocity_base_m_s':[.02,-.06,0.],'accepted_count':20}]}},np.array([0.,-1.,0.]),1.)
    assert not warm['tracks'][0]['speed_valid']
    assert '待确认' in speed_lines(warm,None,now=1.1,maximum_age_s=1.)[0][0]
    assert warm['tracks'][0]['accepted_count']==20


def test_frozen_position_is_observation_not_grasp_center_and_value_never_follows_live_update(tmp_path):
    t,frame=target_fixture();frozen=frozen_speed_snapshot(t,np.array([0.,-1.,0.]),2.)
    np.testing.assert_array_equal(frozen['position_base_m'],[0,0,1])
    assert frozen['along_belt_m_s']==pytest.approx(.06)
    archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True,direction=[0,-1,0])
    archive.offer_speed(frame,np.eye(4),peer(frame),None,capture_s=1.)
    archive.offer_target(t);result=archive.close()
    assert not result['errors']
    snap=RunPreviewWorker(tmp_path,direction=[0,-1,0]).poll()
    assert snap['live_speed']['tracks'][0]['along_belt_m_s']==pytest.approx(.0815)
    assert snap['frozen_speed']['along_belt_m_s']==pytest.approx(.06)
    assert snap['track_labels'][0]['id']==2


def test_tracker_labels_are_not_projected_onto_a_different_exposure_or_epoch():
    _,frame=target_fixture();live=live_speed_snapshot(frame,peer(frame),None,np.array([0,-1,0]),1.)
    assert paired_track_labels(live,frame,np.array([0,-1,0]))
    assert not paired_track_labels(live,replace(frame,header=replace(frame.header,sequence=8)),np.array([0,-1,0]))
    assert not paired_track_labels(live,replace(frame,header=replace(frame.header,clock_epoch=1)),np.array([0,-1,0]))
    assert '未更新' in speed_lines(live,None,now=3.,maximum_age_s=1.)[0][0]
    assert '历史卡尔曼' in speed_lines(live,None,now=3.,maximum_age_s=1.,historical=True)[0][0]


def test_older_saved_run_recovers_only_its_logged_frozen_velocity_and_position(tmp_path):
    t,frame=target_fixture();archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True,direction=[0,-1,0])
    archive.offer_target(t);assert not archive.close()['errors']
    path=tmp_path/'preview/target.npz'
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    meta=json.loads(str(arrays['metadata']));meta.pop('frozen_speed')
    arrays['metadata']=np.asarray(json.dumps(meta))
    with path.open('wb') as stream:np.savez(stream,**arrays)
    qualified={'event':'qualified_target','monotonic_s':2.,'data':{'identity':[0,1,2],
        'sequence':7,'capture_monotonic_s':1.,'speed_report':t.speed_report|{'measured_velocity_base_m_s':[0,-.06,0]}}}
    events=tmp_path/'events.jsonl';events.write_text(json.dumps(qualified)+'\n')
    reader=RunPreviewWorker(tmp_path,direction=[0,-1,0]);snap=reader.poll()
    assert snap['frozen_speed']['position_base_m'] is None
    assert snap.get('live_speed') is None  # never invent historic per-frame data
    submitted={'event':'execution_event','data':{'record':{'event':'planning_submitted',
        'data':{'identity':[0,1,2],'entry_observed_position_base_m':[0,0,1]}}}}
    events.write_text(json.dumps(qualified)+'\n'+json.dumps(submitted)+'\n')
    snap=reader.poll()
    assert snap['frozen_speed']['along_belt_m_s']==pytest.approx(.06)
    assert snap['frozen_speed']['position_base_m']==[0,0,1]


def test_qt_image_contains_yellow_freeze_marker_and_distinct_live_and_frozen_numbers(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen');pytest.importorskip('PyQt5')
    from PyQt5 import QtCore,QtWidgets
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    from nero_core.two_second.camera_view import PreviewWindow
    from nero_core.two_second.config import CellConfig
    t,frame=target_fixture();archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True,direction=[0,-1,0])
    archive.offer_speed(frame,np.eye(4),peer(frame),None,capture_s=1.);archive.offer_target(t);archive.close()
    reader=RunPreviewWorker(tmp_path,direction=[0,-1,0]);reader.poll()
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    window=PreviewWindow(reader,cfg)
    try:
        before=frame.rgb.copy();window.refresh()
        text=window.speed_status.text()
        assert '8.15' in text and '6.00' in text and '冻结速度' in text
        image=window.rgb.pixmap().toImage()
        color=image.pixelColor(image.width()//2,image.height()//2)
        assert color.red()>240 and color.green()>200 and color.blue()<100
        assert window.frozen_rgb.pixmap() is not None
        np.testing.assert_array_equal(before,frame.rgb)
    finally:window.close();app.processEvents()


def test_vision_telemetry_hook_gets_the_actual_camera_peer_without_another_meter(monkeypatch):
    from test_two_second_diagnostics import vision_fixture
    vision=vision_fixture(monkeypatch,'ready');rows=[]
    vision.speed_sink=lambda frame,d2c,peer,report,**kw:rows.append((frame.header.sequence,peer.sequence,kw['capture_s']))
    vision._run()
    assert len(rows)==1 and rows[0][0]==rows[0][1]==7
