"""Manual opening sends one jaw target only after fresh stationary feedback."""
from threading import Event
from types import SimpleNamespace

import pytest

from nero_core.control_panel.gripper_worker import open_gripper, GripperOpenError


def fixture(fault=None):
    current=[10.]
    stopped=Event()
    calls=[]
    config=SimpleNamespace(raw={'hardware': {'can_channel':'fake0','firmware':'v121',
        'joint_receive_offset_s':0.}},settings=SimpleNamespace(open_width_m=.1,move_timeout_s=1.))
    class Device:
        def __init__(self,*a):self.commanded=False
        def start(self):calls.append(('start',))
        def close(self):calls.append(('close',))
        def grip(self,width,force):
            self.commanded=True
            calls.append(('grip',width,force))
            if fault=='cancel_after':stopped.set()
            if fault=='send_error':raise RuntimeError('SDK send uncertain')
        def snapshot(self,**kw):
            if fault=='driver':raise RuntimeError('关节未使能')
            q=[0.]*7
            if fault=='moving':q=[(current[0]-10.)*.2]*7
            if fault=='moves_after' and self.commanded:q=[.01]*7
            stamp=current[0]-.001
            if fault=='stale':stamp-=.3
            grip_stamp=10.155 if fault=='repeated_after' and self.commanded else stamp
            width=.1 if self.commanded and fault!='not_opened' else .02
            if fault=='nan':width=float('nan')
            return {'joints':q,'width_m':width,'force_n':0.,
                'pose_stamp':stamp,'joint_stamp':stamp,'gripper_stamp':grip_stamp}
    def sleep(dt):current[0]+=dt
    return config,stopped,Device,lambda:current[0],sleep,calls


def test_one_open_command_and_real_feedback_confirmation_without_arm_targets():
    cfg,stop,factory,clock,sleep,calls=fixture()
    result=open_gripper(cfg,stop,factory=factory,clock=clock,sleep=sleep)
    assert result['opened'] and result['width_m']==.1
    assert result['motion_commands']==0 and result['gripper_commands']==1
    assert calls==[('start',),('grip',.1,1.),('close',)]
    assert result['elapsed_s']>=.3


@pytest.mark.parametrize('fault',['stale','moving','driver','nan'])
def test_bad_or_moving_feedback_rejects_before_any_jaw_target(fault):
    cfg,stop,factory,clock,sleep,calls=fixture(fault)
    with pytest.raises(GripperOpenError) as error:
        open_gripper(cfg,stop,factory=factory,clock=clock,sleep=sleep)
    assert not error.value.command_attempted
    assert not any(c[0]=='grip' for c in calls)
    assert calls[-1]==('close',)


@pytest.mark.parametrize('fault',['not_opened','moves_after','cancel_after','repeated_after','send_error'])
def test_no_success_without_new_confirmation_and_no_compensating_arm_or_close(fault):
    cfg,stop,factory,clock,sleep,calls=fixture(fault)
    with pytest.raises(GripperOpenError) as error:
        open_gripper(cfg,stop,factory=factory,clock=clock,sleep=sleep)
    assert error.value.command_attempted
    assert [c for c in calls if c[0]=='grip']==[('grip',.1,1.)]
    assert calls[-1]==('close',)


def test_cancelled_before_start_never_constructs_or_connects_hardware():
    cfg,stop,_,clock,sleep,calls=fixture()
    stop.set()
    def forbidden(*a):pytest.fail('must not construct hardware after cancellation')
    with pytest.raises(GripperOpenError):open_gripper(cfg,stop,factory=forbidden,clock=clock,sleep=sleep)
    assert not calls
