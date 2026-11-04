"""Randomized simulation feed contracts; no idealized grasp or device access."""
from types import SimpleNamespace
import numpy as np
import pytest
from experiments.two_second_grasp.random_trials import make_trials, spawn_trial


def test_each_piece_has_its_own_reproducible_random_speed_and_lateral_position():
    kwargs=dict(speed_range=[.08,.12],y_range=[-.035,.035])
    trials=make_trials([12,6,9,0,14],1543,.1,**kwargs)
    assert trials==make_trials([12,6,9,0,14],1543,.1,**kwargs)
    assert len({r['belt_set_m_s'] for r in trials})==5
    assert len({r['spawn_y_world_m'] for r in trials})==5
    assert all(.08<=r['belt_set_m_s']<=.12 and -.035<=r['spawn_y_world_m']<=.035 for r in trials)
    assert [r['category_id'] for r in trials]==[12,6,9,0,14]
    assert all(r['spawn_y_world_m']!=0 for r in trials)
    y_only=make_trials([12,6,9,0,14],1543,.1,y_range=[-.035,.035])
    assert [r['spawn_y_world_m'] for r in y_only]==[r['spawn_y_world_m'] for r in trials]


def test_omitted_random_options_keep_fixed_speed_and_native_spawn_randomness():
    trials=make_trials([12,6],1543,.1)
    assert all(t['belt_set_m_s']==.1 and t['spawn_y_world_m'] is None for t in trials)


@pytest.mark.parametrize('kwargs', [dict(speed_range=[0,.1]),dict(speed_range=[.2,.1]),
    dict(speed_range=[.1,float('nan')]),dict(speed_range=[.1,1.1]),dict(y_range=[.1,.1]),
    dict(y_range=[0,float('inf')]),dict(y_range=[0]),dict(fixed_speed=float('nan'))])
def test_invalid_randomization_is_rejected_before_environment_creation(kwargs):
    settings=dict(seed=1,fixed_speed=.1);settings.update(kwargs)
    with pytest.raises(ValueError):make_trials([12],**settings)


def test_spawn_sets_only_environment_speed_and_normal_spawn_parameters():
    calls=[]
    trial=make_trials([12],1543,.1,speed_range=[.08,.12],y_range=[-.035,.035])[0]
    env=SimpleNamespace(data=SimpleNamespace(time=10.,qpos=np.array([-1.28,.02,.501])),
        belt_speed=.1,_objects=[SimpleNamespace(logical_id=4,qpos_adr=0)])
    def set_speed(v):calls.append(('speed',v));return v
    def spawn(**kwargs):calls.append(('spawn',kwargs));return 4
    env.set_belt_speed=set_speed;env.spawn_object=spawn
    logical_id,evidence=spawn_trial(env,trial)
    assert logical_id==4 and [c[0] for c in calls]==['speed','spawn']
    assert calls[1][1]==dict(category_id=12,y=trial['spawn_y_world_m'])
    np.testing.assert_array_equal(evidence['spawn_position_world_m'],[-1.28,.02,.501])
    evidence['spawn_position_world_m'][0]=999
    assert env.data.qpos[0]==-1.28


def test_clipped_speed_is_not_silently_accepted():
    trial=make_trials([12],1,.1)[0]
    env=SimpleNamespace(set_belt_speed=lambda v:.05,spawn_object=lambda **kw:pytest.fail('must not spawn'))
    with pytest.raises(ValueError,match='clipped'):spawn_trial(env,trial)


def test_explicit_initial_yaw_is_only_passed_to_native_spawn():
    trial=make_trials([12],1543,.0595)[0];trial['spawn_yaw_rad']=np.deg2rad(35.)
    q=np.array([-1.28,.02,.501,np.cos(trial['spawn_yaw_rad']/2),0.,0.,np.sin(trial['spawn_yaw_rad']/2)])
    calls=[]
    env=SimpleNamespace(data=SimpleNamespace(time=0.,qpos=q),belt_speed=.0595,
                        _objects=[SimpleNamespace(logical_id=4,qpos_adr=0)],
                        set_belt_speed=lambda value:value,
                        spawn_object=lambda **kwargs:(calls.append(kwargs) or 4))
    _,record=spawn_trial(env,trial)
    assert calls==[{'category_id':12,'yaw':pytest.approx(np.deg2rad(35.))}]
    assert record['spawn_orientation_yaw_rad']==pytest.approx(np.deg2rad(35.))
    np.testing.assert_array_equal(env.data.qpos,q)


def test_random_yaw_is_reproducible_and_does_not_change_speed_or_lateral_draws():
    args=dict(speed_range=[.08,.12],y_range=[-.035,.035])
    baseline=make_trials([12,6,9,0,14],1543,.1,**args)
    randomized=make_trials([12,6,9,0,14],1543,.1,**args,yaw_range=[-np.pi/2,np.pi/2])
    assert randomized==make_trials([12,6,9,0,14],1543,.1,**args,yaw_range=[-np.pi/2,np.pi/2])
    assert len({r['spawn_yaw_rad'] for r in randomized})==5
    assert all(-np.pi/2<=r['spawn_yaw_rad']<=np.pi/2 for r in randomized)
    assert [r['belt_set_m_s'] for r in baseline]==[r['belt_set_m_s'] for r in randomized]
    assert [r['spawn_y_world_m'] for r in baseline]==[r['spawn_y_world_m'] for r in randomized]
