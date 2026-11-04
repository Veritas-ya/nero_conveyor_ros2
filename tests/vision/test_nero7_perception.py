"""Contracts for deployed Nero7 pixels/timestamps and unavailable velocity."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.graspgenx_sim.nero7_perception import CoLocatedNero7Meter, propagated_record
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7BeltVelocityMeter


def test_rate_report_keeps_sim_capture_frequency_separate_from_wall_throughput():
    from experiments.graspgenx_sim.nero7_run import measurement_rates
    rates = measurement_rates([0., .5, 1.], {'sim_seconds': 1., 'wall_seconds': 2.5})
    assert rates['processed_rate_hz'] == 2.
    assert rates['processed_rate_clock'] == 'simulation_capture_time'
    assert rates['processed_reports_per_wall_episode_s'] == 1.2
    assert measurement_rates([0., .5], {})['processed_reports_per_wall_episode_s'] is None


def test_camera_metadata_overrides_inherited_field_replay_claim(monkeypatch):
    from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation
    monkeypatch.setattr(Nero7FixedSimulation.__mro__[1], 'camera_record',
                        lambda _: {'scene_note': 'camera pose replayed from saved field data'})
    env = object.__new__(Nero7FixedSimulation)
    env.nero_base_to_world = np.eye(4)
    env.preserve_control_profile = False
    env.maximum_measured_speed_m_s = None
    env.simulation_camera_world_xyz = np.array([-.35, 0., 1.25])
    env.visual_workspace_x_world = None
    camera = env.camera_record()
    assert 'synthetic near-top-down' in camera['scene_note']
    assert 'not replayed field extrinsics' in camera['scene_note']


def test_candidate_snapshot_keeps_original_clone_and_uses_one_seed_dls(monkeypatch):
    from experiments.graspgenx_sim.nero7_perception import (
        Nero7SensorControlEnvironment, _single_seed_candidate_ik,
    )
    from experiments.graspgenx_sim.sensor_control import SensorControlEnvironment
    from experiments.graspgenx_sim.candidates import CandidateEnvironmentSnapshot
    calls=[]
    answer=object()
    def solve(position,quaternion,*,qpos_seed):
        calls.append((position,quaternion,qpos_seed))
        return answer
    snapshot=SimpleNamespace(ik=SimpleNamespace(solve=solve),model=object(),data=object())
    clones=[]
    monkeypatch.setattr(SensorControlEnvironment,'candidate_snapshot',
        lambda self:clones.append(self) or snapshot)
    env=object.__new__(Nero7SensorControlEnvironment)
    result=env.candidate_snapshot()
    assert result is snapshot and clones==[env]
    assert result.solve_ik_result.__func__ is _single_seed_candidate_ik
    position=np.array([.08,.02,.6]);quaternion=np.array([0.,1.,0.,0.]);seed=np.arange(7.)
    assert result.solve_ik_result(position,quaternion,seed) is answer
    assert result.solve_ik_result(position,quaternion,seed,allow_multiseed=False) is answer
    assert len(calls)==2 and all(call[2] is seed for call in calls)
    with pytest.raises(ValueError,match='单一关节种子'):
        result.solve_ik_result(position,quaternion,seed,allow_multiseed=True)
    assert len(calls)==2
    # The historical snapshot class and ordinary simulation entry retain their
    # original solver; only this new entry's private instance is adapted.
    assert CandidateEnvironmentSnapshot.solve_ik_result is not _single_seed_candidate_ik


def test_effective_policy_diagnostic_reads_small_actual_config_without_changing_it(monkeypatch):
    from experiments.graspgenx_sim.nero7_policy import Nero7NetworkSeedPolicy
    from experiments.graspgenx_sim.policy import NetworkSeedSortingPolicy
    from mujoco_conveyor_sorting.controllers.baseline_sorting_policy import BaselinePolicyConfig
    monkeypatch.setattr(NetworkSeedSortingPolicy,'get_info',lambda self:{'stage':'waiting'})
    policy=object.__new__(Nero7NetworkSeedPolicy)
    policy.config=BaselinePolicyConfig(dynamic_intercept_tracking=False,control_timestep=.01,
        descent_lead_time=.8,tracking_velocity_feedforward_time=.04,
        closing_tracking_lead_time=.04,closure_contact_lead_time=.04,
        closure_horizontal_tolerance=.008,closure_vertical_tolerance=.004,
        closure_orientation_tolerance=.025)
    policy._acquisition_tangent_priority=False
    original=policy.config
    info=policy.get_info()
    assert info['stage']=='waiting'
    effective=info['nero7_effective_policy_config']
    assert len(effective)==10
    assert effective['dynamic_intercept_tracking'] is False
    assert effective['acquisition_tangent_priority'] is False
    assert effective['control_timestep']==.01 and effective['descent_lead_time']==.8
    assert effective['tracking_velocity_feedforward_time']==.04
    assert effective['closure_horizontal_tolerance']==.008
    assert policy.config is original and not policy.config.dynamic_intercept_tracking


def test_default_dynamic_flag_reaches_actual_make_policy_and_adapter_as_only_config_difference():
    from copy import deepcopy
    from dataclasses import asdict
    from pathlib import Path
    import yaml
    from nero_core.sim_common import make_policy
    from experiments.graspgenx_sim.run import config_overrides
    from experiments.graspgenx_sim.nero7_run import _adapter
    root=Path(__file__).resolve().parents[2]/"src/nero_core"
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml').read_text())
    assert cfg['tracking']['dynamic_intercept']=={'enabled':True}
    scene=yaml.safe_load((root/'mujoco_conveyor_sorting/configs/scene.yaml').read_text())
    robot=yaml.safe_load((root/'mujoco_conveyor_sorting/configs/robot.yaml').read_text())['robot']
    configs={}
    for enabled in (False,True):
        varied=deepcopy(cfg)
        varied['tracking']['dynamic_intercept']['enabled']=enabled
        overrides=config_overrides(varied)
        effective_scene=deepcopy(scene)
        effective_scene.setdefault('workspace',{})['dynamic_intercept']=overrides['scene']['workspace']['dynamic_intercept']
        env=SimpleNamespace(action_type='ee_delta_pose',robot_config=deepcopy(robot),
            scene_config=effective_scene,perception_mode='rendered_rgbd',grasp_assist=False)
        initial=make_policy(env,compensation=True)
        try:
            assert initial.config.dynamic_intercept_tracking is enabled
        finally:
            initial.close()
        control=cfg['control']
        adapter=_adapter(env,
            enable_speculative_prefetch=control['speculative_prefetch'],
            acquisition_translation_step_m=control['acquisition_translation_step_m'],
            acquisition_command_integrator=control['acquisition_command_integrator'],
            visual_acquisition_tracking=control['visual_acquisition_tracking'],
            minimum_motor_hold_effort=control['minimum_motor_hold_effort_sim_N'],
            visual_servo_bias_compensation=control['visual_servo_bias_compensation'])
        try:
            configs[enabled]=asdict(adapter.policy.config)
            info=adapter.policy.get_info()['nero7_effective_policy_config']
            assert info['dynamic_intercept_tracking'] is enabled
            assert not info['acquisition_tangent_priority']
            assert info['control_timestep']==.01 and info['descent_lead_time']==.8
            assert info['tracking_velocity_feedforward_time']==.04
            assert info['closing_tracking_lead_time']==.04 and info['closure_contact_lead_time']==.04
            assert info['closure_horizontal_tolerance']==.008
            assert info['closure_vertical_tolerance']==.004 and info['closure_orientation_tolerance']==.025
        finally:
            adapter.policy.close()
    changed={key for key in configs[False] if configs[False][key]!=configs[True][key]}
    assert changed=={'dynamic_intercept_tracking'}
    for key in ('camera','control','graspgenx'):
        assert varied[key]==cfg[key]


def _surface():
    plane = np.eye(4); plane[2, 3] = -.1
    hull = np.array([[-.7,-.7],[.7,-.7],[.7,.7],[-.7,.7]],np.float32)
    return BeltSurface(plane,np.array([[-.7,.7],[-.7,.7]]),hull,{})


def _frame(sequence=0, width=160, height=120):
    focal=400. if width>160 else 120.
    K=np.array([[focal,0.,(width-1)/2],[0.,focal,(height-1)/2],[0.,0.,1.]])
    intr=SimpleNamespace(matrix=K,width=width,height=height,distortion=(0.,)*5)
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    depth=np.full((height,width),1.1,np.float32)
    return SimpleNamespace(rgb=np.zeros((height,width,3),np.uint8),depth_m=depth,
        color_intrinsics=intr,depth_intrinsics=intr,camera_to_world_cv=T,
        header=SimpleNamespace(source_time=sequence*.05,sequence=sequence,clock_epoch=1))


@pytest.mark.parametrize('bad', [np.nan,0.,-1.,np.inf])
def test_identity_registration_exact_pixels_with_invalid_depth_and_roi_edges(bad):
    frame=_frame(); frame.depth_m[0,0]=bad;frame.depth_m[-1,-1]=bad
    frame.depth_m[20,30]=bad
    frame.depth_m[50:70,60:90]=1.06
    ordinary=Nero7BeltVelocityMeter(_surface()); fast=CoLocatedNero7Meter(_surface())
    for meter in (ordinary,fast):meter._geometry(frame,np.eye(4))
    selected=ordinary._native_selection(frame.depth_m,np.eye(4))
    expected=ordinary._registered_depth(frame.depth_m,np.eye(4),selected)
    actual=fast._registered_depth(frame.depth_m,np.eye(4),selected)
    np.testing.assert_array_equal(actual,expected)


def test_different_intrinsics_falls_back(monkeypatch):
    meter=CoLocatedNero7Meter(_surface()); frame=_frame()
    meter._geometry(frame,np.eye(4));meter._depth_key=('different',)
    sentinel=object();calls=[]
    def registration(self,*args): calls.append(args);return sentinel
    monkeypatch.setattr(Nero7BeltVelocityMeter,'_registered_depth',registration)
    assert meter._registered_depth(frame.depth_m,np.eye(4),np.ones_like(frame.depth_m,bool)) is sentinel
    assert len(calls)==1


def test_nonidentity_transform_falls_back(monkeypatch):
    meter=CoLocatedNero7Meter(_surface());frame=_frame();meter._geometry(frame,np.eye(4))
    sentinel=object()
    monkeypatch.setattr(Nero7BeltVelocityMeter,'_registered_depth',lambda *args:sentinel)
    d2c=np.eye(4);d2c[0,3]=.001
    assert meter._registered_depth(frame.depth_m,d2c,np.ones_like(frame.depth_m,bool)) is sentinel


def test_peer_observations_qualification_and_velocity_identical_to_original():
    ordinary=Nero7BeltVelocityMeter(_surface(),belt_direction_base=[1.,0.])
    fast=CoLocatedNero7Meter(_surface(),belt_direction_base=[1.,0.])
    measured=[]
    for index in range(13):
        frame=_frame(index,width=512,height=384)
        frame.depth_m[160:190,180+index:210+index]=1.06
        left=ordinary.update(frame,np.eye(4));right=fast.update(frame,np.eye(4))
        assert ordinary.latest_snapshot==fast.latest_snapshot
        assert left['speed_valid']==right['speed_valid']
        if index<7: assert not right['speed_valid']
        measured.append(right['speed_valid'])
    assert any(measured)
    assert fast.parameters['speed_min_samples']==8
    assert fast.parameters['speed_min_span']==.3
    assert fast.parameters['speed_window_s']==.6


def test_no_qualified_velocity_does_not_invent_zero_or_nominal_speed():
    record={'measurement_timestamp':2.,'qualified_velocity_world_m_s':None}
    assert propagated_record(record,2.1,1.) is None


def test_position_and_velocity_keep_independent_original_capture_stamps():
    record={'measurement_timestamp':2.,'velocity_measurement_timestamp':1.7,
            'measured_position_world_m':[.2,.3,.4],
            'qualified_velocity_world_m_s':[.06,.02,0.]}
    position,velocity,age,speed_age=propagated_record(record,2.1,1.)
    np.testing.assert_allclose(position,[.206,.302,.4])
    np.testing.assert_allclose(velocity,[.06,.02,0.])
    assert age==pytest.approx(.1);assert speed_age==pytest.approx(.4)
    assert record['velocity_measurement_timestamp']==1.7


def test_expired_speed_is_not_refreshed_by_new_position():
    record={'measurement_timestamp':3.,'velocity_measurement_timestamp':1.7,
            'measured_position_world_m':[.2,.3,.4],
            'qualified_velocity_world_m_s':[.06,.02,0.]}
    assert propagated_record(record,3.01,1.) is None


def test_measured_stationary_zero_is_valid_and_future_time_is_rejected():
    record={'measurement_timestamp':2.,'velocity_measurement_timestamp':2.,
            'measured_position_world_m':[.2,.3,.4],
            'qualified_velocity_world_m_s':[0.,0.,0.]}
    assert propagated_record(record,2.1,1.) is not None
    assert propagated_record(record,1.9,1.) is None


def test_actual_shared_acquisition_functions_are_used_without_historical_override():
    from experiments.graspgenx_sim.nero7_policy import Nero7NetworkSeedPolicy
    assert all(Nero7NetworkSeedPolicy.shared_method_identity().values())
    from experiments.graspgenx_sim.policy import NetworkSeedSortingPolicy
    assert Nero7NetworkSeedPolicy._pregrasp_goal_seeds is NetworkSeedSortingPolicy._pregrasp_goal_seeds
    assert Nero7NetworkSeedPolicy.register_pregrasp_seed is NetworkSeedSortingPolicy.register_pregrasp_seed


def test_overhead_result_report_describes_actual_camera_and_unknown_uncertainty(tmp_path):
    import json
    from pathlib import Path
    import yaml
    from experiments.graspgenx_sim.nero7_run import _correct_result_metadata
    cfg=yaml.safe_load((Path(__file__).resolve().parents[2]/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml').read_text())
    result={'config':cfg,'status':'completed','evaluation':{'successful_sorted_objects':0},
        'limitations':['Fixed camera matches the saved relative pose; conveyor layout remains simulated.'],
        'camera':{'old':'reference'}}
    path=tmp_path/'result.json';path.write_text(json.dumps(result))
    camera={'pose_source':'declared synthetic near-top-down view','world_depth_cv':np.eye(4).tolist()}
    env=SimpleNamespace(camera_record=lambda:camera)
    _correct_result_metadata(tmp_path,env,{'processed_frames':12})
    corrected=json.loads(path.read_text())
    assert corrected['evaluation']==result['evaluation']
    assert corrected['camera']==camera
    assert corrected['nero7_perception']['prediction_uncertainty_status']=='not_estimated'
    assert not corrected['nero7_perception']['belt_encoder_used']
    assert all('matches the saved relative pose' not in line for line in corrected['limitations'])
    report=(tmp_path/'report.txt').read_text()
    assert '保存数据的固定外置相机相对基座姿态' not in report
    assert '明确声明的近俯视模拟相机' in report
    assert '数值零仅为兼容字段' in report
    assert '无带速编码器或固定速度先验' in report


def _planning_boundary():
    from experiments.graspgenx_sim.nero7_perception import Nero7SensorControlEnvironment
    from experiments.graspgenx_sim.sensor_control import SensorControlConfig,VisualTask
    env=object.__new__(Nero7SensorControlEnvironment)
    env._physics=SimpleNamespace(data=SimpleNamespace(time=5.),
        scene_config={'workspace':{'preferred_grasp_x':.08}},grasp_region=[-.2,.28],
        reach_y=[-.18,.18],belt_max_x=.625,_nero_records={},nero7_planning_diagnostics=[])
    env.config=SensorControlConfig(maximum_track_age_s=1.)
    env.tasks={0:VisualTask(0,6)};env.selected_target_id=None
    pose=np.array([-.2,0.,.5,1.,0.,0.,0.])
    prediction={'current_estimated_pose':pose,'estimated_velocity':np.array([.1,0.,0.]),
                'grasp_window':{'end_time':10.}}
    record={'object_id':0,'state':'TRACKING','window_valid':True,'grasp_window_end':10.,
            'position':pose[:3],'estimated_grasp_time':7.8,
            'predicted_grasp_pose':np.array([.08,0.,.5,1.,0.,0.,0.])}
    env._last_observation={'objects':[record],'predictions':{0:prediction}}
    return env


def test_nero7_task_selection_reserves_and_retimes_through_original_interface():
    boundary=_planning_boundary()
    selected=boundary.select_target(boundary._last_observation)
    assert selected['object_id']==0
    assert boundary.tasks[0].state=='PLANNING'
    assert boundary.selected_target_id==0
    result=boundary.refine_selected_grasp_plan(0,1.)
    assert result['accepted']
    assert result['estimated_grasp_time']==pytest.approx(7.8)
    assert boundary.tasks[0].state=='PLANNING'


def test_missing_prediction_reports_visual_loss_without_extending_age_or_changing_task_state():
    boundary=_planning_boundary();boundary.select_target(boundary._last_observation)
    boundary._last_observation['predictions'].clear()
    boundary._physics._nero_records[0]={'measurement_timestamp':3.9}
    result=boundary.refine_selected_grasp_plan(0,1.)
    assert not result['accepted']
    assert result['reason']=='visual_prediction_unavailable'
    assert result['measurement_age_s']==pytest.approx(1.1)
    assert result['maximum_track_age_s']==1.
    assert boundary.tasks[0].state=='PLANNING'
    assert boundary.selected_target_id==0


def test_nonplanning_task_still_fails_original_lifecycle_requirement():
    boundary=_planning_boundary();boundary.select_target(boundary._last_observation)
    boundary.tasks[0].state='EXECUTING'
    result=boundary.refine_selected_grasp_plan(0,1.)
    assert not result['accepted'] and result['reason']=='target_not_in_planning'


def test_sparse_label_transport_preserves_holes_edges_and_numeric_ids():
    from experiments.graspgenx_sim.nero7_perception import _pack_labels,_unpack_labels
    labels=np.full((70,93),-1,np.int32)
    labels[0:25,0:30]=17;labels[4:10,6:12]=-1
    labels[40:70,70:93]=12345;labels[50,80]=-1
    shape,regions=_pack_labels(labels)
    restored=_unpack_labels(shape,regions)
    np.testing.assert_array_equal(restored,labels)
    assert sum(region[-1].nbytes for region in regions)<labels.nbytes/10
    empty=np.full((13,19),-1,np.int32)
    shape,regions=_pack_labels(empty)
    np.testing.assert_array_equal(_unpack_labels(shape,regions),empty)


def test_shared_capture_transport_preserves_exact_peer_outputs_and_original_pixel_labels():
    from multiprocessing.shared_memory import SharedMemory
    from experiments.graspgenx_sim.nero7_perception import _initialize,_measure,_measure_shared,_unpack_labels
    class Classifier:
        def extract_components(self,*args):return ()
    frame=_frame(1,width=512,height=384)
    frame.depth_m[160:190,180:210]=1.06
    calibration={'intrinsics':frame.depth_intrinsics.matrix,'camera_to_world':np.eye(4)}
    _initialize(_surface(),Classifier(),np.eye(4))
    ordinary=_measure((frame,calibration))
    offset=(frame.rgb.size+7)//8*8
    memory=SharedMemory(create=True,size=offset+frame.depth_m.size*4)
    try:
        np.copyto(np.ndarray(frame.rgb.shape,np.uint8,buffer=memory.buf),frame.rgb)
        np.copyto(np.ndarray(frame.depth_m.shape,np.float32,buffer=memory.buf,offset=offset),frame.depth_m)
        metadata={key:getattr(frame,key) for key in
                  ('header','color_intrinsics','depth_intrinsics','camera_to_world_cv')}
        _initialize(_surface(),Classifier(),np.eye(4))
        shared=_measure_shared(metadata,memory.name,frame.rgb.shape,frame.depth_m.shape,offset,calibration)
        assert shared['peer']==ordinary['peer']
        np.testing.assert_array_equal(_unpack_labels(shared['label_shape'],shared['label_regions']),
                                      _unpack_labels(ordinary['label_shape'],ordinary['label_regions']))
        np.testing.assert_array_equal(frame.depth_m,np.ndarray(frame.depth_m.shape,np.float32,buffer=memory.buf,offset=offset))
    finally:
        memory.close();memory.unlink()


@pytest.mark.parametrize('shutdown_fails',[False,True])
def test_simulation_shared_capture_slot_is_unlinked_even_when_worker_shutdown_fails(monkeypatch,shutdown_fails):
    from multiprocessing.shared_memory import SharedMemory
    from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation,FixedGeminiSimulation
    env=object.__new__(Nero7FixedSimulation)
    calls=[]
    def shutdown(**kwargs):
        calls.append(kwargs)
        if shutdown_fails:raise RuntimeError('worker shutdown failed')
    memory=SharedMemory(create=True,size=64);name=memory.name
    env._nero_worker=SimpleNamespace(shutdown=shutdown)
    env._nero_memory=memory
    monkeypatch.setattr(FixedGeminiSimulation,'close',lambda self:calls.append('physics_closed'))
    if shutdown_fails:
        with pytest.raises(RuntimeError,match='worker shutdown failed'):env.close()
    else:env.close()
    assert env._nero_worker is None and env._nero_memory is None
    assert calls[-1]=='physics_closed'
    with pytest.raises(FileNotFoundError):SharedMemory(name=name)


def test_eight_term_zero_rational_projection_preserves_original_pixels():
    from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import SDKCameraGeometry, _compact_cv_distortion
    from vendor.nero7_speed_reference.nero7_perception.geometry import CameraGeometry
    distortion=(-.0296674594,.0354110487,-.0000111761,.0000054242,-.0128312344,0.,0.,0.)
    matrix=np.array([[603.5458,0,640.5368],[0,603.7946,360.7278],[0,0,1.]])
    reference=CameraGeometry(matrix,1280,720,'color',distortion)
    compact=SDKCameraGeometry(matrix,1280,720,'color',_compact_cv_distortion(distortion))
    rng=np.random.default_rng(47)
    points=np.column_stack((rng.uniform(-.8,.8,20000),rng.uniform(-.5,.5,20000),rng.uniform(.4,2.,20000)))
    actual=compact.project(points)
    expected=reference.project(points)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-9)
    np.testing.assert_array_equal(np.rint(actual),np.rint(expected))
    nonzero=distortion[:-1]+(.001,)
    assert _compact_cv_distortion(nonzero)==nonzero
