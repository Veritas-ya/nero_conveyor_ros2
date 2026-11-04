"""Known synthetic clouds and native robot hulls; no device/physical grasp claim."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import yaml

from nero_core.two_second.belt_collision_scene import install_belt_surface, BELT_SURFACE_ONLY
from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import MeasuredPlanner, calibrated_model
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def scene_fixture():
    model = SimpleNamespace(belt_scene_filter_config=SimpleNamespace(voxel_size_m=.005),
        scene=np.array([[9.,9.,9.]]),tree=object())
    plane = np.eye(4)
    plane[:3,:3] = Rotation.from_euler('xyz',[.04,-.03,.6]).as_matrix()
    plane[:3,3] = [.1,-.2,.3]
    local = np.array([[0,0,0],[.02,0,.002],[0,0,.12],[0,0,-.02],[.4,0,0],[np.nan,0,0]])
    points = local@plane[:3,:3].T+plane[:3,3]
    target = SimpleNamespace(identity=(0,1,2),epoch=0,sequence=7,capture_s=1.,
        detection={'scene_points':points,'scene_pixel_indices':np.arange(100,106),
            'belt_region':{'plane_to_base':plane,'bounds_xy_m':[[-.1,.1],[-.1,.1]]}})
    return model,target


def test_only_measured_surface_keeps_native_points_indices_and_coverage():
    model,target = scene_fixture()
    original = target.detection['scene_points'].copy()
    report = install_belt_surface(model,target)
    assert report['scope']==BELT_SURFACE_ONLY and report['native_belt_points']==2
    assert report['excluded_non_belt_points']==4 and report['synthetic_points']==0
    assert report['downsample']['coverage_within_budget']
    np.testing.assert_allclose(model.scene,original[:2])
    np.testing.assert_array_equal(model.scene_native_pixel_indices['belt_surface_retained'],[100,101])
    assert not model.scene.flags.writeable
    target.detection['scene_points'][:]=999
    np.testing.assert_allclose(model.scene,original[:2])
    np.testing.assert_allclose(model.tree.query(model.scene)[0],0)


@pytest.mark.parametrize('invalid',['missing_region','non_rigid','unordered_bounds','empty_surface','wrong_pixels'])
def test_invalid_input_never_commits_an_empty_or_partial_scene(invalid):
    model,target = scene_fixture(); prior_scene,prior_tree=model.scene,model.tree
    if invalid=='missing_region':target.detection.pop('belt_region')
    elif invalid=='non_rigid':target.detection['belt_region']['plane_to_base'][0,0]=3
    elif invalid=='unordered_bounds':target.detection['belt_region']['bounds_xy_m']=[[1,-1],[-1,1]]
    elif invalid=='empty_surface':target.detection['scene_points']=np.ones((6,3))*8
    elif invalid=='wrong_pixels':target.detection['scene_pixel_indices']=np.array([1,2])
    with pytest.raises(PickRejected):install_belt_surface(model,target)
    assert model.scene is prior_scene and model.tree is prior_tree


@pytest.mark.parametrize('sequence,epoch',[(8,0),(7,1)])
def test_wrong_exposure_or_clock_epoch_is_not_reused(sequence,epoch):
    model,target=scene_fixture()
    target.detection['frame']=SimpleNamespace(header=SimpleNamespace(sequence=sequence,clock_epoch=epoch))
    with pytest.raises(PickRejected,match='不匹配'):install_belt_surface(model,target)


def test_new_target_installs_new_tree_and_clears_old_source_index():
    model,target=scene_fixture();install_belt_surface(model,target)
    old=model.tree
    target.sequence+=1;target.identity=(0,1,3)
    target.detection['scene_points'][0]+=target.detection['belt_region']['plane_to_base'][:3,0]*.03
    target.detection.pop('scene_pixel_indices')
    report=install_belt_surface(model,target)
    assert model.tree is not old and model.scene_native_pixel_indices=={}
    assert report['target_identity']==[0,1,3] and report['frame_sequence']==8


def test_real_hull_still_rejects_collision_with_a_retained_belt_point():
    """Deliberately put a synthetic belt through the actual arm hull."""
    cfg=CellConfig.load(Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml')
    model=calibrated_model(cfg)
    geom=model.model.geom('nero_link6_collision').id
    center=model.data.geom_xpos[geom].copy()
    plane=np.eye(4);plane[:3,3]=center
    target=SimpleNamespace(identity=(0,1,2),epoch=0,sequence=7,capture_s=1.,
        detection={'scene_points':center[None,:],
            'belt_region':{'plane_to_base':plane,'bounds_xy_m':[[-.1,.1],[-.1,.1]]}})
    install_belt_surface(model,target)
    with pytest.raises(PickRejected) as error:model.check_state(cfg.standby,.1,allow_finger_contact=True)
    assert error.value.details['kind']=='scene_collision'


def test_only_environment_selection_changes_and_model_checks_are_not_disabled():
    cfg=SimpleNamespace(raw={'environment_collision_scope':BELT_SURFACE_ONLY},workspace=np.array([[-2,2]]*3))
    class Model:
        gripper_geometries=[]
        meshes={}
        def check_state(self,q,width,*,allow_finger_contact=False,check_scene=True):
            assert check_scene is True
            raise PickRejected('TEST self collision')
    planner=MeasuredPlanner(cfg,model=Model())
    with pytest.raises(PickRejected,match='TEST self collision'):planner.valid(np.zeros(7),.1)


def test_planner_installs_scope_for_all_later_stages_without_general_scene_filters():
    model,target=scene_fixture()
    model.check_feedback=lambda s:None
    def forbidden(*args,**kwargs):pytest.fail('belt-only scope must not install general scenery')
    model.install_scene=forbidden
    target.points=np.array([[1.,1.,1.]]);target.state={}
    planner=MeasuredPlanner(SimpleNamespace(raw={'environment_collision_scope':BELT_SURFACE_ONLY}),model=model)
    planner.install(target,{})
    assert planner.identity==target.identity and planner.scene_revision==planner.version==1
    assert planner.diagnostics()['collision_scene']['scope']==BELT_SURFACE_ONLY
    assert len(planner.model.scene)==2 and not len(planner.model.target)


def test_scope_is_explicit_and_invalid_values_fail_before_devices(tmp_path):
    source=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'
    raw=yaml.safe_load(source.read_text())
    for key in ('camera_calibration_file','tcp_calibration_file'):raw[key]=str((source.parent/raw[key]).resolve())
    path=tmp_path/'config.yaml'
    for invalid in ('ignore_all',None,False):
        raw['environment_collision_scope']=invalid;path.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError,match='environment_collision_scope'):CellConfig.load(path)
    raw.pop('environment_collision_scope');path.write_text(yaml.safe_dump(raw))
    assert CellConfig.load(path).raw['environment_collision_scope']=='observed_scene'


def test_collision_evidence_and_passive_preview_keep_the_actual_environment_scope(tmp_path):
    from test_collision_visualization import fixture
    from nero_core.two_second.collision_view import prepare_view,save_packet
    from nero_core.two_second.run_preview import RunPreviewWorker
    planner,_=fixture();planner.config.raw['environment_collision_scope']=BELT_SURFACE_ONLY
    packets=[];planner.collision_sink=packets.append
    with pytest.raises(PickRejected):planner.valid(np.zeros(7),.1,stamp=2.)
    assert packets[0]['metadata']['environment_collision_scope']==BELT_SURFACE_ONLY
    prepare_view(packets[0]);save_packet(tmp_path/'collisions',packets[0])
    snap=RunPreviewWorker(tmp_path).poll()
    assert snap['scene_source']==BELT_SURFACE_ONLY


def test_belt_prefilter_preserves_exact_native_points_and_source_pixels():
    """Boundary candidates remain until the exact belt predicate is applied."""
    from scipy.spatial import cKDTree
    model,target=scene_fixture()
    original=target.detection['scene_points'].copy()
    target.detection['scene_points']=original
    original_report=install_belt_surface(model,target)
    exact_points=model.scene.copy()
    exact_pixels=model.scene_native_pixel_indices['belt_surface_retained'].copy()
    plane=target.detection['belt_region']['plane_to_base']
    bounds=np.asarray(target.detection['belt_region']['bounds_xy_m'])
    local=(original-plane[:3,3])@plane[:3,:3]
    epsilon=1e-8
    broad=(np.abs(local[:,2])<=.005+epsilon)
    broad &= (local[:,0]>=bounds[0,0]-epsilon)&(local[:,0]<=bounds[0,1]+epsilon)
    broad &= (local[:,1]>=bounds[1,0]-epsilon)&(local[:,1]<=bounds[1,1]+epsilon)
    target.detection['scene_points']=original[broad]
    target.detection['scene_pixel_indices']=np.arange(100,106)[broad]
    target.detection['scene_prefilter_scope']=BELT_SURFACE_ONLY
    target.detection['scene_input_count_before_prefilter']=len(original)
    filtered_report=install_belt_surface(model,target)
    np.testing.assert_array_equal(model.scene,exact_points)
    np.testing.assert_array_equal(model.scene_native_pixel_indices['belt_surface_retained'],exact_pixels)
    assert filtered_report['native_belt_points']==original_report['native_belt_points']
    assert filtered_report['input_scene_points']==original_report['input_scene_points']
    np.testing.assert_allclose(model.tree.query(exact_points)[0],0.)
