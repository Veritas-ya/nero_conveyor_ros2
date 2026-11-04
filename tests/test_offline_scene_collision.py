"""Offline-only ablation must not acquire SDKs, clear other failures or certify paths."""
from pathlib import Path
from types import SimpleNamespace
import builtins,json
import numpy as np
import pytest
from nero_core.two_second.offline_scene_collision import compare,load_policy,saved_cases
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def saved(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (run/'summary.json').write_text(json.dumps({'details':{'failures':[{'candidate_index':0,
        'details':{'kind':'scene_collision','geometry':'TEST_link6','points_base_m':[[0,0,1]],
            'point_count':1,'joints_rad':[0.]*7,'width_m':.1}}]}}))
    return run


class Model:
    def __init__(self,other_failure=False):self.calls=[];self.other_failure=other_failure
    def check_state(self,q,width,*,allow_finger_contact,check_scene):
        self.calls.append(check_scene)
        if self.other_failure:raise PickRejected('TEST joint limit')
        if check_scene:raise PickRejected('TEST scene collision',details={'kind':'scene_collision'})


def test_record_only_retains_alarm_and_never_claims_a_hardware_ready_trajectory(tmp_path):
    model=Model();report=compare(saved(tmp_path),SimpleNamespace(source_sha256='TEST'), 'record_only',model=model)
    case=report['cases'][0]
    assert not case['scene_check']['passed'] and case['selected_saved_state_passed']
    assert case['scene_check']['error']=='TEST scene collision' and model.calls==[True,False]
    assert not report['trajectory_computed'] and not report['hardware_execution_authorized']
    assert report['hardware_connections']==report['commands_sent']==0


@pytest.mark.parametrize('action',['reject','record_only'])
def test_model_faults_remain_rejections_in_either_policy(tmp_path,action):
    report=compare(saved(tmp_path),SimpleNamespace(source_sha256='TEST'),action,model=Model(other_failure=True))
    assert not report['cases'][0]['selected_saved_state_passed']


def test_enforced_policy_and_invalid_policy_are_explicit(tmp_path):
    report=compare(saved(tmp_path),SimpleNamespace(source_sha256='TEST'),'reject',model=Model())
    assert not report['cases'][0]['selected_saved_state_passed']
    path=tmp_path/'policy.yaml';path.write_text('scope: hardware\nscene_collision_action: record_only\n')
    with pytest.raises(ValueError):load_policy(path)
    path.write_text('scope: offline_saved_state_only\nscene_collision_action: record_only\n')
    assert load_policy(path)=='record_only'


def test_missing_or_nonfinite_inputs_are_not_substituted(tmp_path):
    run=saved(tmp_path)
    p=run/'summary.json';data=json.loads(p.read_text())
    data['details']['failures'][0]['details']['joints_rad'][0]=float('nan');p.write_text(json.dumps(data))
    with pytest.raises(ValueError):saved_cases(run)
    p.write_text('{"details":{}}')
    with pytest.raises(ValueError):saved_cases(run)


def test_import_and_comparison_do_not_touch_hardware_modules(tmp_path,monkeypatch):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if 'nero_pick_hardware' in name or name.startswith(('pyAgxArm','pyorbbecsdk')) or name=='can':
            raise AssertionError('offline comparison must not import a device SDK')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    report=compare(saved(tmp_path),SimpleNamespace(source_sha256='TEST'),'record_only',model=Model())
    assert report['commands_sent']==0


def test_offline_policy_cannot_be_used_as_a_live_pick_configuration(tmp_path,monkeypatch):
    from nero_core.two_second.main import main
    policy=tmp_path/'offline.yaml'
    policy.write_text('scope: offline_saved_state_only\nscene_collision_action: record_only\n')
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if 'nero_pick_hardware' in name or 'backends.orbbec' in name:
            raise AssertionError('invalid offline policy must fail before device imports')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    output=tmp_path/'must_not_be_created'
    assert main(['--config',str(policy),'--execute','--output',str(output)])==2
    assert not output.exists()
