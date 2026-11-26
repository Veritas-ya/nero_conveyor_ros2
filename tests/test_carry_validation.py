"""Private carry certificates keep all measured-input and ownership requirements."""
import numpy as np
import pytest

from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
from nero_core.two_second import carry_validation as cv
from test_carry_lookahead import candidate


def request_case(monkeypatch):
    p,state,bundle,result=candidate()
    def snapshot(planner,path):
        return {'token':'validation-1','identity':planner.identity,'scene_revision':planner.scene_revision}
    monkeypatch.setattr(cv,'scene_snapshot',snapshot)
    request=cv.validation_request(p,state,bundle.carry.q0,result,token='current',source_version=p.version)
    return p,state,bundle,result,request


def worker_answer(monkeypatch,request,*,reject=False,cancel=False):
    private,_,_,_=candidate()
    private.identity=request['snapshot']['identity'];private.scene_revision=request['snapshot']['scene_revision']
    monkeypatch.setattr(cv,'restore_scene',lambda *args:None)
    seen=[]
    def check(path,width,**kw):
        seen.append((width,private.payload_local.copy()))
        if reject:raise PickRejected('actual payload hits rim')
    private.check_path=check
    answer=cv.validate_in_private_model(private,request,cancelled=lambda:cancel)
    return answer,seen


def accept(p,state,bundle,result,request,answer,cancel=False):
    return cv.accept_validation(p,state,bundle.carry.q0,result,request,answer,
        token='current',source_version=p.version,cancelled=lambda:cancel)


def test_private_full_check_and_parent_handoff_use_same_exact_measured_attachment(monkeypatch):
    p,state,bundle,result,request=request_case(monkeypatch)
    original=p.payload_local.copy();answer,seen=worker_answer(monkeypatch,request)
    assert answer['ok'] and len(seen)==1
    assert seen[0][0]==state['width_m']
    np.testing.assert_array_equal(seen[0][1],p.payload_local)
    p.calls.clear()
    accepted,evidence=accept(p,state,bundle,result,request,answer)
    assert accepted is bundle and not p.calls  # no geometry/search repeated in parent
    assert evidence['validation_execution']=='private_process'
    assert evidence['return_validation']=='deferred_until_actual_release'
    np.testing.assert_array_equal(p.payload_local,original)


@pytest.mark.parametrize('change',['payload','width','support_anchor','joints','token','task','scene','version','cancel'])
def test_changed_inputs_or_late_certificate_cannot_activate_motion(monkeypatch,change):
    p,state,bundle,result,request=request_case(monkeypatch)
    answer,_=worker_answer(monkeypatch,request)
    if change=='payload':p.payload_local[0,0]+=.001
    if change=='width':state['width_m']+=.000001
    if change=='support_anchor':p.capture_tcp[0,3]+=.000001
    if change=='joints':state['joints'][0]+=.000001
    if change=='token':answer['validation_token']='old'
    if change=='task':answer['identity']=(9,9,9)
    if change=='scene':p.scene_revision+=1
    if change=='version':p.version+=1
    with pytest.raises(PickRejected):accept(p,state,bundle,result,request,answer,cancel=change=='cancel')


@pytest.mark.parametrize('cancel',[False,True])
def test_worker_collision_or_cancellation_is_not_hidden(monkeypatch,cancel):
    p,state,bundle,result,request=request_case(monkeypatch)
    answer,seen=worker_answer(monkeypatch,request,reject=not cancel,cancel=cancel)
    assert not answer['ok']
    with pytest.raises(PickRejected,match='取消' if cancel else 'actual payload hits rim'):
        accept(p,state,bundle,result,request,answer)


def test_request_copies_measured_geometry_and_feedback(monkeypatch):
    p,state,bundle,result,request=request_case(monkeypatch)
    p.payload_local[:]=4.;p.capture_tcp[:]=3.;state['width_m']=.07
    assert request['state']['width_m']==.04
    assert np.all(request['snapshot']['payload_local']==1.)
    np.testing.assert_array_equal(request['snapshot']['capture_tcp'],np.eye(4))


def test_worker_collision_diagnostics_return_to_parent_sink(monkeypatch):
    from types import SimpleNamespace
    p,state,bundle,result,request=request_case(monkeypatch)
    private,_,_,_=candidate()
    monkeypatch.setattr(cv,'restore_scene',lambda *args:None)
    def reject(*a,**kw):
        private.collision_sink({'reason':'test-payload-collision'})
        raise PickRejected('actual payload hits rim')
    private.check_path=reject
    answer=cv.validate_in_private_model(private,request,cancelled=lambda:False)
    packets=[];p.collision_sink=packets.append;p.target=SimpleNamespace(frame='frozen-frame')
    with pytest.raises(PickRejected,match='actual payload hits rim'):
        accept(p,state,bundle,result,request,answer)
    assert packets==[{'reason':'test-payload-collision','frame':'frozen-frame'}]
    assert private.collision_sink is None
