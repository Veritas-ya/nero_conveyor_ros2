"""Continuous-cycle and Ctrl+C fault injection; no physical devices are opened."""
import builtins
import importlib
import json
from pathlib import Path
import signal
from threading import Event
import time
from types import SimpleNamespace as NS

import pytest

from nero_core.two_second.cycle_loop import run_pick_cycles
from nero_core.two_second.console import ConsoleProgress


CONFIG=Path(__file__).parents[1]/'src/nero_bringup/config/two_second.hardware.legacy_observed.yaml'


def test_continuous_configuration_accepts_fixed_speed_without_device_imports(monkeypatch,capsys):
    from nero_core.two_second.main import main
    original=builtins.__import__
    def guarded(name,*args,**kw):
        assert 'nero_pick_hardware' not in name and 'backends.orbbec' not in name
        return original(name,*args,**kw)
    monkeypatch.setattr(builtins,'__import__',guarded)
    assert main(['--config',str(CONFIG),'--continuous','--check-config','--belt-speed-cm-s','4.53'])==0
    report=json.loads(capsys.readouterr().out)
    assert report['continuous'] and report['requested_count'] is None
    assert report['velocity_policy']['fixed_belt_speed_m_s']==pytest.approx(.0453)
    assert report['hardware_connections']==report['commands_sent']==0


def test_count_and_continuous_are_mutually_exclusive():
    from nero_core.two_second.main import main
    with pytest.raises(SystemExit) as failure:
        main(['--config',str(CONFIG),'--count','2','--continuous'])
    assert failure.value.code==2


def target(index=1):
    return NS(identity=(0,index,1),capture_s=10.,received_s=10.,device_s=1.,
        sequence=index,speed_report={},grasps=(None,),detection={})


def test_continuous_mixed_results_rearm_after_return_and_persist_with_bounded_memory():
    cancelled=Event();at_wait=True;attempt=0;saved=[];consumed=[];events=[]
    class Vision:
        clock_epoch=0
        rearmed=0
        def resume_targets(self):
            assert at_wait and saved[-1]==attempt
            self.rearmed+=1
        def poll(self):return target(self.rearmed+1)
        def consume(self,identity):
            nonlocal at_wait
            assert at_wait
            at_wait=False;consumed.append(identity)
    class Executor:
        def __init__(self,*a,**kw):pass
        def run(self,item):
            nonlocal attempt,at_wait
            attempt+=1
            # Both successful and failed attempts must finish the return.
            at_wait=True
            return {'status':'cycle_completed_from_feedback' if attempt%2==0 else
                'attempt_cancelled_returned_to_wait','returned_to_wait':True}
    def persist(executor,index,report):
        assert at_wait and report['returned_to_wait']
        saved.append(index)
        if index==50:cancelled.set()
    result={'completed_feedback_cycles':0};reports=[];executors=[];vision=Vision()
    run_pick_cycles(NS(raw={'maximum_observation_age_s':1.}),None,None,vision,
        count=None,cancelled=cancelled,result=result,reports=reports,executors=executors,
        set_phase=lambda *a,**kw:None,journal=NS(emit=lambda name,**kw:events.append(name)),
        executor_factory=Executor,finish_attempt=persist,clock=lambda:10.)
    assert result['status']=='cancelled'
    assert result['completed_feedback_cycles']==result['recovered_attempts']==25
    assert result['attempt_count']==50 and vision.rearmed==49
    assert len(set(consumed))==50 and saved==list(range(1,51))
    assert len(reports)==20 and len(executors)==1
    assert events.count('cycle_finished')==events.count('attempt_recovered')==25


@pytest.mark.parametrize('cancel_race',[False,True])
def test_ctrl_c_waiting_never_constructs_an_executor(cancel_race):
    cancelled=Event();calls=[]
    def poll():
        cancelled.set()
        return target() if cancel_race else None
    def forbidden(*a,**kw):pytest.fail('cancelled waiting must not select a target or send motion')
    vision=NS(clock_epoch=0,poll=poll,consume=forbidden,resume_targets=forbidden)
    result={'completed_feedback_cycles':0}
    run_pick_cycles(NS(raw={'maximum_observation_age_s':1.}),None,None,vision,
        count=None,cancelled=cancelled,result=result,reports=[],executors=[],set_phase=lambda *a,**kw:None,
        journal=NS(emit=lambda *a,**kw:calls.append(a)),executor_factory=forbidden,
        finish_attempt=lambda *a:None,clock=lambda:10.)
    assert result['status']=='cancelled' and not calls


@pytest.mark.parametrize('status',['cycle_completed_from_feedback','attempt_cancelled_returned_to_wait'])
def test_an_unconfirmed_return_never_rearms(status):
    calls=[]
    vision=NS(clock_epoch=0,poll=lambda:target(),consume=lambda *a:None,
        resume_targets=lambda:calls.append('rearm'))
    class Executor:
        def __init__(self,*a,**kw):pass
        def run(self,item):return {'status':status,'returned_to_wait':False}
    with pytest.raises(RuntimeError,match='等待位'):
        run_pick_cycles(NS(raw={'maximum_observation_age_s':1.}),None,None,vision,
            count=None,cancelled=Event(),result={'completed_feedback_cycles':0},reports=[],executors=[],
            set_phase=lambda *a,**kw:None,journal=NS(emit=lambda *a,**kw:None),
            executor_factory=Executor,finish_attempt=lambda *a:None,clock=lambda:10.)
    assert calls==[]


@pytest.mark.parametrize('preview',[False,True])
def test_sigint_entry_persists_both_results_and_closes_all_device_doubles(tmp_path,monkeypatch,preview):
    from test_two_second_diagnostics import wire_offline_entry
    entry,path,calls=wire_offline_entry(monkeypatch,'waiting')
    # This fixture tests cancellation of the original serial owner; the
    # lookahead lifecycle and overlapping return have their own device doubles.
    from nero_core.two_second.config import CellConfig
    original_load = CellConfig.load
    def serial_fixture(config_path):
        config = original_load(config_path)
        config.raw['continuous_lookahead_enabled'] = False
        return config
    monkeypatch.setattr(CellConfig, 'load', serial_fixture)
    class Viewer:
        pid=12345
        running=True
        def poll(self):return None if self.running else 0
        def terminate(self):self.running=False;calls.append('close_test_viewer')
        def wait(self,timeout):assert not self.running;return 0
    if preview:monkeypatch.setattr(entry.subprocess,'Popen',lambda *a,**kw:Viewer())
    class Vision:
        clock_epoch=0
        index=1
        def __init__(self,*a):pass
        def start(self):return self
        def close(self):calls.append('close_test_vision')
        def diagnostics(self):return {}
        def consume(self,*a):pass
        def resume_targets(self):self.index+=1
        def poll(self):
            if self.index==3:
                signal.raise_signal(signal.SIGINT)
                return None
            item=target(self.index)
            item.capture_s=item.received_s=time.monotonic()
            return item
    next_attempt_started=Event()
    class Trace:
        def write(self,directory):
            if directory.name=='cycle_001':
                # A synchronous first archive cannot satisfy this: the next
                # executor must start before the first disk operation ends.
                assert next_attempt_started.wait(timeout=2.)
            directory.mkdir(parents=True,exist_ok=True)
            (directory/'test_trace.json').write_text('{"offline_test_only":true}')
    class Executor:
        stage='waiting'
        def __init__(self,*a,**kw):
            self.trace=Trace();self.events=[];self.event_logging_errors=[]
            self.command_history=NS(diagnostics=lambda:{'offline_test_only':True})
        def run(self,item):
            if item.sequence==2:next_attempt_started.set()
            return {'status':'cycle_completed_from_feedback' if item.sequence==1 else
                'attempt_cancelled_returned_to_wait','returned_to_wait':True}
    monkeypatch.setattr(importlib.import_module('nero_core.two_second.vision'),'LiveVision',Vision)
    monkeypatch.setattr(importlib.import_module('nero_core.two_second.execution'),'TwoSecondExecutor',Executor)
    output=tmp_path/'continuous'
    argv=['--config',str(path),'--continuous','--execute','--belt-speed-cm-s','5.95','--output',str(output)]
    if preview:argv.append('--preview')
    assert entry.main(argv)==1
    summary=json.loads((output/'summary.json').read_text())
    assert summary['status']=='cancelled' and summary['received_signal']=='SIGINT'
    assert summary['continuous'] and summary['requested_count'] is None
    assert summary['completed_feedback_cycles']==summary['recovered_attempts']==1
    assert summary['archived_attempt_count']==2 and summary['log_save_errors']==[]
    assert summary['attempt_archive']['saved']==2
    assert not summary['attempt_archive']['writer_alive']
    for index in (1,2):
        assert (output/f'cycle_{index:03d}'/'test_trace.json').exists()
        assert json.loads((output/f'cycle_{index:03d}'/'report.json').read_text())['returned_to_wait']
    assert not (output/'cycle_003').exists()
    assert calls.count('construct_test_device')==calls.count('construct_test_camera')==1
    assert calls.count('close_test_device')==calls.count('close_test_camera')==calls.count('close_test_vision')==1
    assert 'return_wait_target' not in calls and 'test_plan' not in calls
    assert not summary['journal']['observer_errors']
    if preview:
        assert calls.count('close_test_viewer')==1
        assert calls.index('close_test_device')<calls.index('close_test_viewer')
        assert summary['preview_closed_by_parent'] and summary['preview_process_exit_code']==0


def test_console_displays_continuous_progress_without_an_invented_total():
    lines=[];console=ConsoleProgress(write=lines.append)
    def send(name,**data):console.observe({'event':name,'monotonic_s':1.,'data':data})
    send('run_started',continuous=True,requested_count=None)
    send('cycle_finished',cycle_index=3,completed_feedback_cycles=2)
    assert '连续真机抓取' in lines[0] and 'Ctrl+C' in lines[0]
    assert '第 2 件' in lines[-1] and 'None' not in ''.join(lines)
