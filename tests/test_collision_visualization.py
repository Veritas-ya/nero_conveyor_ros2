"""Known synthetic geometry/exposures; no physical camera, arm, or CAN."""
from pathlib import Path
from types import SimpleNamespace
import json
import time
import numpy as np
import pytest
from scipy.spatial import ConvexHull,cKDTree

from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.collision_view import prepare_view,save_packet,read_packet,project_base,overlay_collision
from nero_core.two_second.visual_archive import VisualArchive
from nero_core.two_second.run_preview import RunPreviewWorker
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from mujoco_conveyor_sorting.runtime.types import RGBDFrame,CameraIntrinsics,SampleHeader


def fixture(*,finger=True):
    vertices=np.array([[x,y,z] for x in (-.01,.01) for y in (-.01,.01) for z in (-.01,.01)])
    transform=np.eye(4);transform[2,3]=1.
    class Model:
        model=SimpleNamespace(geom=lambda g:SimpleNamespace(name='TEST_left_finger'),
            body=lambda g:SimpleNamespace(name='TEST_finger_body'),geom_bodyid=np.array([0]))
        meshes={0:(ConvexHull(vertices).equations,.02)}
        mesh_vertices={0:vertices};collision_radii={0:.03}
        fingers={0} if finger else set();gripper_geometries=[]
        data=SimpleNamespace(geom_xmat=np.eye(3).reshape(1,9),geom_xpos=np.array([[0.,0.,1.]]))
        scene=np.empty((0,3))
        def check_state(self,q,width,**kw):pass
        def tcp(self):return transform.copy()
        def inside_geom(self,points,g,margin):
            local=points-self.data.geom_xpos[g]
            e=self.meshes[g][0]
            return np.all(local@e[:,:3].T+e[:,3]<=margin,axis=1)
    config=SimpleNamespace(raw={'base_frame':'TEST_robot_base'},workspace=np.array([[-2,2]]*3),source_sha256='TEST_ONLY')
    p=MeasuredPlanner(config,model=Model(),clock=lambda:3.)
    points=np.array([[-.1,0.,1.],[-.087,0.,1.],[.1,0.,1.]])
    rgb=np.zeros((240,320,3),dtype=np.uint8);rgb[:]=[36,65,75]
    rgb[85:155,120:210]=[170,185,195]
    frame=RGBDFrame(header=SampleHeader(1.,1.01,7,source_time=21.,clock_epoch=0),
        depth_m=np.ones((240,320),np.float32),rgb=rgb,
        depth_intrinsics=CameraIntrinsics(320,240,200.,200.,160.,120.),
        color_intrinsics=CameraIntrinsics(320,240,200.,200.,160.,120.),
        camera_to_world_cv=np.eye(4),source='synthetic_offline_collision_fixture',
        calibration_id='TEST_ONLY',serial_number='NO_DEVICE')
    p.target=SimpleNamespace(points=points,identity=(0,1,2),epoch=0,sequence=7,capture_s=1.,
        velocity=np.array([.1,0.,0.]),state={'joints':[0.]*7,'pose':[0.]*6},
        detection={'frame':frame,'depth_to_color_cv':np.eye(4)})
    p.target_tree=cKDTree(points);p.identity=p.target.identity
    p.plane=np.eye(4);p.support_origin=np.array([0.,0.,-2.]);p.support=np.array([0.,0.,1.])
    p.belt_bounds=np.array([[-1,1],[-1,1]])
    p.geometry_context={'trajectory_stage':'smooth_intercept_follow_brake','origin_monotonic_s':.95,
        'contact_monotonic_s':2.95,'sample_elapsed_s':1.05,'candidate_index':0,'approach_s':2.}
    return p,frame


def make_packet():
    planner,frame=fixture();packets=[];planner.collision_sink=packets.append
    with pytest.raises(PickRejected) as rejected:planner.valid(np.zeros(7),.1,stamp=2.)
    return packets[0],rejected.value.details,planner


def test_exact_prediction_point_ids_margin_and_original_exposure_are_separate():
    packet,detail,p=make_packet()
    assert detail['geometry']=='TEST_left_finger' and detail['is_finger']
    assert detail['inside_nominal_hull_count']==1 and detail['margin_only_count']==1
    assert detail['trajectory_elapsed_s']==pytest.approx(1.05)
    assert not detail['contact_allowed'] and detail['prediction_not_measured_contact']
    np.testing.assert_array_equal(packet['arrays']['hit_indices'],[0,1])
    np.testing.assert_allclose(packet['arrays']['hit_points_base_m'],[[0,0,1],[.013,0,1]],atol=1e-12)
    p.target.points[:]=999;p.model.data.geom_xpos[:]=999
    np.testing.assert_allclose(packet['arrays']['object_points_exposure_base_m'][0],[-.1,0,1])
    np.testing.assert_allclose(packet['arrays']['geometry_to_base'][:3,3],[0,0,1])


def test_existing_contact_exception_and_nonfinger_rejection_are_preserved():
    p,_=fixture();captured=[];p.collision_sink=captured.append
    p.valid(np.zeros(7),.1,stamp=2.,contact=True)
    assert captured==[]
    p,_=fixture(finger=False)
    with pytest.raises(PickRejected,match='工件点云碰撞'):p.valid(np.zeros(7),.1,stamp=2.,contact=True)


def test_broken_diagnostic_sink_cannot_hide_collision():
    p,_=fixture()
    def fail(_):raise OSError('injected diagnostic failure')
    p.collision_sink=fail
    with pytest.raises(PickRejected) as error:p.valid(np.zeros(7),.1,stamp=2.)
    assert error.value.details['geometry']=='TEST_left_finger'
    assert 'injected diagnostic failure' in error.value.details['diagnostic_error']


def test_full_binary_evidence_and_rgb_overlay_round_trip_with_no_pickle(tmp_path):
    packet,detail,_=make_packet();prepare_view(packet)
    path=save_packet(tmp_path,packet);loaded=read_packet(path)
    assert path.with_suffix('.npz').exists() and path.with_suffix('.png').exists()
    assert loaded['metadata']['camera']['calibration_id']=='TEST_ONLY'
    assert loaded['metadata']['robot_feedback_at_exposure']=={'joints':[0.]*7,'pose':[0.]*6}
    np.testing.assert_array_equal(loaded['frame'].rgb,packet['frame'].rgb)
    for key,a in packet['arrays'].items():np.testing.assert_array_equal(loaded['arrays'][key],a)
    source=packet['frame'].rgb.copy()
    image=overlay_collision(source,packet['frame'],packet['d2c'],packet['arrays'])
    np.testing.assert_array_equal(source,packet['frame'].rgb)
    assert np.any(image!=source)


def test_projection_uses_base_to_depth_then_actual_color_extrinsic_and_rejects_hidden_points():
    _,frame=fixture();d2c=np.eye(4);d2c[0,3]=.1
    uv,visible=project_base([[0,0,1],[0,0,-1],[20,0,1]],frame,d2c)
    np.testing.assert_allclose(uv[0],[180,120])
    np.testing.assert_array_equal(visible,[True,False,False])
    assert project_base([],frame,d2c)[0].shape==(0,2)


def test_projection_keeps_camera_rotation_and_translation_in_the_same_frame():
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    _,frame=fixture();transform=np.eye(4)
    transform[:3,:3]=Rotation.from_euler('xyz',[.1,-.2,.5]).as_matrix();transform[:3,3]=[.3,-.4,.2]
    frame=replace(frame,camera_to_world_cv=transform)
    optical=np.array([[.1,.05,1.]])
    base=optical@transform[:3,:3].T+transform[:3,3]
    uv,visible=project_base(base,frame,np.eye(4))
    np.testing.assert_allclose(uv,[[180,130]],atol=1e-10)
    assert visible.all()


def test_async_archive_and_passive_view_use_the_same_frame_and_never_construct_devices(tmp_path,monkeypatch):
    packet,detail,_=make_packet();rows=[]
    archive=VisualArchive(tmp_path,lambda event,**data:rows.append((event,data)),live=True)
    archive.offer_frame(packet['frame'],packet['d2c']);archive.submit(packet)
    result=archive.close()
    assert not result['errors'] and not result['writer_alive'] and result['dropped']==0
    assert len(result['saved_collision_records'])==1
    assert (tmp_path/'preview/latest.npz').is_file()
    def forbidden(*a,**kw):raise AssertionError('passive viewer must never construct a device')
    monkeypatch.setattr('mujoco_conveyor_sorting.runtime.backends.orbbec.OrbbecCameraBackend',forbidden)
    monkeypatch.setattr('mujoco_conveyor_sorting.runtime.nero_pick_hardware.Hardware',forbidden)
    viewer=RunPreviewWorker(tmp_path);snap=viewer.poll()
    assert viewer.can_connections==viewer.camera_connections==0 and snap['shared_run']
    assert '离线合成验证' in snap['status']
    np.testing.assert_array_equal(snap['frame'].rgb,packet['frame'].rgb)
    assert snap['collision']['metadata']['target_identity']==[0,1,2]
    # A bad/missing new file never manufactures another observation.
    assert viewer.poll()['frame'].header.sequence==7


def test_archive_write_failure_is_reported_without_becoming_a_motion_result(tmp_path,monkeypatch):
    packet,_,_=make_packet()
    def fail(*a):raise OSError('injected disk failure')
    monkeypatch.setattr('nero_core.two_second.collision_view.save_packet',fail)
    archive=VisualArchive(tmp_path,lambda *a,**kw:None);archive.submit(packet)
    result=archive.close()
    assert len(result['errors'])==1 and 'injected disk failure' in result['errors'][0]
    assert not result['writer_alive'] and not result['saved_collision_records']


def test_production_frozen_obb_is_shared_without_running_a_second_detector(tmp_path):
    packet,_,p=make_packet()
    p.target.detection['obb']={'center_world_m':[0,0,1], 'axes_world_columns':np.eye(3),
        'extent_m':[.02,.03,.04]}
    p.target.detection['belt_region']={'plane_to_base':np.eye(4)}
    p.target.grasps=tuple(np.eye(4) for _ in range(4))
    archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True)
    archive.offer_target(p.target);result=archive.close()
    assert not result['errors']
    snap=RunPreviewWorker(tmp_path).poll()
    assert snap['production_target']==[0,1,2]
    np.testing.assert_array_equal(snap['live_geometry']['points'],p.target.points)
    assert len(snap['live_geometry']['grasps'])==4


def test_planned_fk_path_is_paired_to_same_frozen_exposure(tmp_path):
    from nero_core.two_second.trajectory import TimedPath
    packet,_,planner=make_packet();target=planner.target
    target.detection.update(obb={'center_world_m':[0,0,1], 'axes_world_columns':np.eye(3),
        'extent_m':[.02,.03,.04]},belt_region={'plane_to_base':np.eye(4)})
    target.grasps=tuple(np.eye(4) for _ in range(4))
    class PrivateModel:
        def set_state(self,q,width): self.q=q
        def tcp(self):
            t=np.eye(4);t[:3,3]=[self.q[0],0,1];return t
    curve=lambda t,d=0:np.r_[t if d==0 else 1. if d==1 else 0.,np.zeros(6)]
    path=TimedPath(curve,target.identity,7,'intercept',2.,.2)
    reference=SimpleNamespace(start_s=2.,duration_s=1.,follow_end_s=1.5,
        contact_s=3.,contact_xyz=np.array([2.,0.,1.]))
    plan=SimpleNamespace(path=path,reference=reference,target=target,selected_grasp=np.eye(4))
    archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True,path_model_factory=PrivateModel)
    archive.offer_target(target);archive.offer_plan(plan,width_m=.1)
    assert not archive.close()['errors']
    reader=RunPreviewWorker(tmp_path);snap=reader.poll();shown=snap['production_plan']
    assert shown['metadata']['path_version']==7
    assert shown['metadata']['meaning']=='planner_output_not_proof_of_dispatch_or_motion'
    np.testing.assert_allclose(shown['tcp_xyz_base_m'][0],[0,0,1])
    np.testing.assert_allclose(shown['tcp_xyz_base_m'][-1],[2,0,1])
    np.testing.assert_allclose(shown['predicted_contact_points_base_m'],target.points+[.2,0,0])
    # The saved path cannot be overlaid after the target changes.
    target.identity=(0,1,3)
    archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True)
    archive.offer_target(target);assert not archive.close()['errors']
    assert 'production_plan' not in reader.poll()


def test_historical_collision_uses_its_own_exposure_when_live_camera_epoch_changes(tmp_path):
    packet,_,_=make_packet();prepare_view(packet);save_packet(tmp_path/'collisions',packet)
    viewer=RunPreviewWorker(tmp_path);first=viewer.poll()
    assert first['replay'] and first['collision']['frame'].header.sequence==7
    # A live image from another sequence/epoch is separate from the prediction overlay.
    from dataclasses import replace
    frame=replace(packet['frame'],header=SampleHeader(3.,3.01,99,source_time=31.,clock_epoch=1))
    archive=VisualArchive(tmp_path,lambda *a,**kw:None,live=True);archive.offer_frame(frame,np.eye(4));archive.close()
    after=viewer.poll()
    assert after['frame'].header.clock_epoch==1 and after['collision']['frame'].header.clock_epoch==0
    assert after['collision']['metadata']['frame_sequence']==7


def test_qt_window_keeps_collision_image_on_refresh_and_exposes_no_motion_controls(tmp_path,monkeypatch):
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    pytest.importorskip('PyQt5')
    from PyQt5 import QtCore,QtWidgets
    import os
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
    from nero_core.two_second.camera_view import PreviewWindow
    from nero_core.two_second.config import CellConfig
    packet,_,_=make_packet();prepare_view(packet);save_packet(tmp_path/'collisions',packet)
    worker=RunPreviewWorker(tmp_path);worker.poll()
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    window=PreviewWindow(worker,cfg)
    try:
        window.refresh();window.refresh()  # identical snapshot must not clear the image
        assert window.frozen_rgb.pixmap() is not None and not window.frozen_rgb.pixmap().isNull()
        assert 'TEST_finger_body' in window.details.text()
        buttons=[b.text() for b in window.findChildren(QtWidgets.QPushButton)]
        assert not any('启动抓取' in b or '发送' in b or '使能' in b for b in buttons)
    finally:
        window.close();app.processEvents()
