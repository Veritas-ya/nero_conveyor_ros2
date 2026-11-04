import pytest
from nero_core.task import Stage, TaskGate

def test_late_cancel_switch_and_reset():
    t = TaskGate(); first = t.start('a', 'one'); old = t.request()
    assert not t.accepts(first) and t.accepts(old)
    t.cancel(); assert not t.accepts(old)
    current = t.start('b','two'); assert not t.accepts(old)
    t.reset_clock(); assert not t.accepts(current)
    assert t.start('c','three').epoch == 1
    t.cancel()
    with pytest.raises(ValueError): t.start('again','one')

def test_failed_closure_never_lifts():
    t = TaskGate(); t.start('a','one'); t.advance('certified'); t.advance('at_follow')
    assert t.advance('grasp_failed') is Stage.FAILED
    with pytest.raises(ValueError): t.advance('lifted')
    t.cancel(); assert t.stage is Stage.WAIT

def test_full_demo_waits_again():
    t=TaskGate(); t.start('a','one')
    for event in ('certified','at_follow','grasp_confirmed','lifted','released','home'): t.advance(event)
    assert t.stage is Stage.WAIT and t.active is None
    with pytest.raises(ValueError): t.start('b','one')
    t.start('c','two')
