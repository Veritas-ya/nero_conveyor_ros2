from dataclasses import replace
from experiments.graspgenx_sim.nero7_identity import Nero7StableIdentityMap, _Binding
from mujoco_conveyor_sorting.runtime.nero7_grasp_tracking import Nero7GraspTracking
from mujoco_conveyor_sorting.runtime.nero7_belt_velocity import Nero7FrameSnapshot,Nero7TrackSnapshot

P={'tracker_gate_radius':.03}

def track(identifier,x,*,velocity=.1,valid=True,stamp=0.,misses=0):
    return Nero7TrackSnapshot(identifier,1,(x,0.,0.),(x,0.,0.),
        (velocity,0.,0.) if valid else None,valid,stamp,stamp,8,.4,.001,
        ((1e-6,0.),(0.,1e-6)),misses,1000.,True)

def peer(stamp,*tracks):return Nero7FrameSnapshot(stamp,int(stamp*1000),1,1,True,tracks,'')

def test_selection_is_actual_hardware_rule():
    assert _Binding._select_track is Nero7GraspTracking._select_track
    assert _Binding._observed is Nero7GraspTracking._observed

def test_original_id_keeps_task_id():
    identities=Nero7StableIdentityMap()
    a,_=identities.update(peer(0.,track(20,.2)),P)
    b,info=identities.update(peer(.1,track(20,.21,stamp=.1)),P,active_task_id=0)
    assert list(a)==list(b)==[0]
    assert info['task_to_actual_peer_id']=={0:20}

def test_expired_id_uses_same_position_velocity_rule():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(20,.2)),P)
    a,info=identities.update(peer(.6,track(80,.26,stamp=.6)),P,active_task_id=0)
    assert list(a)==[0]
    assert info['task_to_actual_peer_id']=={0:80}
    assert info['task_reassociation_reasons'][0]=='peer_observation_and_predicted_motion'

def test_missing_old_id_is_not_reassigned_before_peer_expiry():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(20,.2)),P)
    a,info=identities.update(peer(.2,track(20,.2,stamp=0.,misses=1),track(80,.22,stamp=.2)),P,active_task_id=0)
    assert 0 not in a
    assert info['task_to_actual_peer_id']=={1:80}
    b,info=identities.update(peer(.6,track(80,.26,stamp=.6)),P,active_task_id=0)
    assert list(b)==[0]
    assert info['dropped_provisional_task_ids']==[1]

def test_unqualified_startup_id_can_use_new_candidate_qualified_velocity():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(1,.2,valid=False)),P)
    a,_=identities.update(peer(.6,track(3,.26,stamp=.6)),P,active_task_id=0)
    assert list(a)==[0]

def test_ambiguous_tracks_do_not_rebind_saved_target():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(1,.2)),P)
    a,info=identities.update(peer(.6,track(3,.26,stamp=.6),track(4,.27,stamp=.6)),P,active_task_id=0)
    assert 0 not in a
    assert info['task_reassociation_reasons'][0]=='target_track_ambiguous'

def test_clock_reset_never_reuses_old_task_shape_id():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(1,.2)),P)
    changed=replace(peer(.6,track(1,.26,stamp=.6)),clock_epoch=2)
    a,info=identities.update(changed,P,active_task_id=0)
    assert 0 not in a and list(a)==[1]
    assert info['clock_or_generation_reset']


def test_new_unqualified_peer_does_not_replace_a_qualified_saved_target():
    identities=Nero7StableIdentityMap()
    identities.update(peer(0.,track(1,.2)),P)
    a,info=identities.update(peer(.6,track(3,.26,stamp=.6,valid=False)),P,active_task_id=0)
    assert 0 not in a
    assert info['task_reassociation_reasons'][0]=='nero7_speed_not_qualified'
    b,info=identities.update(peer(.7,track(3,.27,stamp=.7)),P,active_task_id=0)
    assert list(b)==[0]


def test_publication_keeps_task_mask_velocity_timestamp_and_observed_position_after_peer_change():
    from collections import deque
    from types import SimpleNamespace
    import numpy as np
    from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation
    env=object.__new__(Nero7FixedSimulation)
    env._nero_identity=Nero7StableIdentityMap()
    env._nero_records={};env._nero_active_task_id=0
    env.nero7_reports=[];env.data=SimpleNamespace(time=0.)
    env.nero_base_to_world=np.eye(4)
    frames=[]
    env.perception_buffer=SimpleNamespace(append_frame=lambda stamp,states,**kw:frames.append((stamp,states)))
    env._associate_visual_tracks_for_evaluation=lambda detections:None
    env._camera_frames=deque();env.vision_pipeline=SimpleNamespace(last_instance_labels=None)
    frame={'timestamp':0.,'source':'not forwarded'}
    feature=np.arange(9,dtype=float)
    def publish(stamp,identifier,x,valid=True):
        env.data.time=stamp+.05
        labels=np.full((4,4),identifier,np.int32)
        packet={'peer':peer(stamp,track(identifier,x,stamp=stamp,valid=valid)),
                'diagnostics':{'parameters':P},'labels':labels,
                'categories':{identifier:(6,.8,feature,.5)},'processing_finished_wall_s':0.}
        env._publish_peer(packet,dict(frame,timestamp=stamp,capture_wall_s=0.))
    publish(0.,20,.2)
    publish(.6,80,.26)
    assert list(env._nero_records)==[0]
    assert env._nero_records[0]['actual_peer_track_id']==80
    assert env._nero_records[0]['measurement_timestamp']==.6
    assert env._nero_records[0]['velocity_measurement_timestamp']==.6
    np.testing.assert_allclose(env._nero_records[0]['measured_position_world_m'],[.26,0.,0.])
    assert set(np.unique(env._camera_frames[-1]['top_instance_labels']))=={0}
    assert list(frames[-1][1])==['0']
    assert env.nero7_reports[-1]['stable_task_association']['task_to_actual_peer_id']=={0:80}
    publish(.8,80,.28,valid=False)
    assert env._nero_records[0]['measurement_timestamp']==.8
    assert env._nero_records[0]['velocity_measurement_timestamp']==.6
    assert env._nero_records[0]['velocity_query']['reason']=='recent_measured_speed'


def test_velocity_query_is_actual_hardware_function():
    from experiments.graspgenx_sim.nero7_identity import _HistorySource
    from mujoco_conveyor_sorting.runtime.nero7_velocity_source import Nero7VelocitySource
    assert _HistorySource.query is Nero7VelocitySource.query


def test_same_id_recent_speed_keeps_original_timestamp_then_expires():
    identities=Nero7StableIdentityMap(maximum_velocity_age_s=1.)
    identities.update(peer(0.,track(1,.2)),P)
    identities.update(peer(.2,track(1,.22,stamp=.2,valid=False)),P,active_task_id=0)
    measurement=identities.bindings[0].measurement
    assert measurement['speed_valid']
    assert measurement['reason']=='recent_measured_speed'
    assert measurement['device_timestamp']==0.
    identities.update(peer(1.1,track(1,.31,stamp=1.1,valid=False)),P,active_task_id=0)
    assert not identities.bindings[0].measurement['speed_valid']


def test_reassociated_id_recent_qualified_speed_keeps_new_id_original_timestamp():
    identities=Nero7StableIdentityMap(maximum_velocity_age_s=1.)
    identities.update(peer(0.,track(1,.2)),P)
    identities.update(peer(.2,track(1,.2,stamp=0.,misses=1),track(3,.22,stamp=.2)),P,active_task_id=0)
    assignments,_=identities.update(peer(.6,track(3,.26,stamp=.6,valid=False)),P,active_task_id=0)
    assert list(assignments)==[0]
    measurement=identities.bindings[0].measurement
    assert measurement['speed_valid']
    assert measurement['reason']=='recent_measured_speed'
    assert measurement['track_id']==3
    assert measurement['device_timestamp']==.2


def test_clock_reset_drops_recent_speed_history_even_with_reused_peer_id():
    identities=Nero7StableIdentityMap(maximum_velocity_age_s=1.)
    identities.update(peer(0.,track(1,.2)),P)
    changed=replace(peer(.1,track(1,.21,stamp=.1,valid=False)),clock_epoch=2)
    assignments,_=identities.update(changed,P,active_task_id=0)
    assert list(assignments)==[1]
    assert not identities.bindings[1].measurement['speed_valid']


def test_simulation_fresh_rgbd_geometry_uses_public_binding_begin_once_per_task():
    from collections import deque
    from types import SimpleNamespace
    import numpy as np
    from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation
    env=object.__new__(Nero7FixedSimulation)
    env._nero_identity=Nero7StableIdentityMap()
    env._nero_records={};env._nero_active_task_id=0
    env.nero7_reports=[];env.data=SimpleNamespace(time=.05)
    env.nero_base_to_world=np.eye(4)
    env.perception_buffer=SimpleNamespace(append_frame=lambda *args,**kwargs:None)
    env._associate_visual_tracks_for_evaluation=lambda detections:None
    env._camera_frames=deque();env.vision_pipeline=SimpleNamespace(last_instance_labels=None)
    T=np.eye(4);T[:3,3]=[.2,0.,1.]
    K=np.array([[100.,0.,1.5],[0.,100.,1.5],[0.,0.,1.]])
    def publish(stamp,x):
        packet={'peer':peer(stamp,track(20,x,stamp=stamp)),'diagnostics':{'parameters':P},
                'labels':np.full((4,4),20,np.int32),
                'categories':{20:(6,.8,np.arange(9,dtype=float),.5)},'processing_finished_wall_s':0.}
        frame={'timestamp':stamp,'capture_wall_s':0.,'top_depth':np.full((4,4),.97,np.float32),
               'top_camera':{'intrinsics':K,'camera_to_world':T}}
        env._publish_peer(packet,frame)
    publish(0.,.2)
    source=env._nero_identity.bindings[0].source
    assert source.diagnostics()['geometry_binding_serial']==1
    assert source.diagnostics()['geometry_binding_capture']==(1,0,0.)
    assert env._nero_records[0]['target_geometry_binding']['binding_reset']
    events=env.nero7_reports[-1]['geometry_binding_events']
    assert len(events)==1
    assert events[0]['operation']=='begin_target_binding'
    assert events[0]['task_id']==0 and events[0]['actual_peer_track_id']==20
    assert events[0]['result']['accepted'] and events[0]['result']['binding_reset']
    assert events[0]['result']['reason']=='new_measured_geometry_target'
    assert events[0]['result']['track_generation']==1 and not events[0]['generation_pending']
    assert events[0]['geometry_capture']=={'clock_epoch':1,'sequence':0,
        'device_timestamp':0.,'host_capture_timestamp':0.,'host_clock':'simulation_capture_time'}
    assert events[0]['captured_point_count']==16
    publish(.2,.22)
    assert source.diagnostics()['geometry_binding_serial']==1
    assert source.diagnostics()['bound_track_id']==20
    assert env.nero7_reports[-1]['geometry_binding_events']==[]
