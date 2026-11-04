"""Terminal presentation only: no SDK, device, physical success or fake motion."""
import json
from threading import Event, current_thread
import time

import numpy as np
import pytest

from nero_core.two_second.console import ConsoleProgress
from nero_core.two_second.diagnostics import RunJournal


def send(console, event, now=1., **data):
    console.observe({'event': event, 'monotonic_s': now, 'data': data})


def execution(console, event, **data):
    send(console, 'execution_event', record={'event':event,'data':data})


def test_steps_distinguish_obb_candidates_from_delivered_path_and_feedback_from_physics(tmp_path):
    lines=[]
    console=ConsoleProgress(direction=(0.,-1.,0.), write=lines.append)
    send(console,'run_started',requested_count=1)
    send(console,'phase_started',phase='waiting_for_qualified_target',cycle_index=1)
    send(console,'qualified_target',speed_report={'measured_velocity_base_m_s':[.001,-.047,0.],
        'sample_count':11,'span_s':.534},grasp_candidate_count=4)
    execution(console,'planning_submitted')
    assert any('4.70 cm/s' in line for line in lines)  # along measured belt, not base X
    assert any('4 个候选' in line for line in lines)
    assert not any('已得到' in line for line in lines)  # a pose is not a delivered trajectory
    pose=np.eye(4); pose[:3,3]=[-.4,.1,.02]
    execution(console,'planning_delivery',selected_grasp_tcp_base=pose.tolist(),
        predicted_contact_tcp_base_m=[-.4,-.05,.02],planning_wall_s=.713,
        trajectory_diagnostics={'detailed_array':list(range(512))})
    # Frozen-exposure grasp position must not be displayed as the future contact.
    assert any('TCP=(-400.0, -50.0, 20.0) mm' in line for line in lines)
    execution(console,'stage_started',stage='smooth_intercept_follow_brake')
    execution(console,'close_during_follow')
    execution(console,'payload_estimate_attached')
    execution(console,'grip_confirmed_from_width_force')
    for stage in ('lift','transfer'):
        execution(console,'route_submitted',stage=stage)
        execution(console,'route_delivered',stage=stage,planning_wall_s=.2)
        execution(console,'stage_started',stage=stage)
    execution(console,'release_open_requested')
    execution(console,'route_submitted',stage='return')
    execution(console,'route_delivered',stage='return',planning_wall_s=.1)
    execution(console,'stage_started',stage='return')
    send(console,'cycle_finished',cycle_index=1)
    console.finish({'status':'requested_feedback_cycles_completed',
        'completed_feedback_cycles':1,'requested_count':1},tmp_path)
    text='\n'.join(lines)
    assert len(lines)<=24 and max(map(len,lines))<150
    assert text.count('[夹持]')==1
    assert '实物入框需现场确认' in text and '抓取成功' not in text
    assert all('{' not in line and 'detailed_array' not in line for line in lines)


def test_repeated_and_alternating_frame_stages_do_not_flood_terminal():
    lines=[]; console=ConsoleProgress(write=lines.append)
    send(console,'phase_started',now=0.,phase='waiting_for_qualified_target')
    for now in np.arange(0.,31.,.01):
        stage='tracking_and_speed' if int(now*100)%2 else 'detecting_target'
        send(console,'runtime_progress',now=float(now),phase='waiting_for_qualified_target',
            vision={'stage':stage,'tracks':[{'id':1}],'frames_received':1})
    assert len(lines)<=5
    send(console,'runtime_progress',now=35.,phase='waiting_for_qualified_target',
        vision={'stage':'waiting_upstream_window','entry_gate':{'reason':'insufficient_planning_time'}})
    send(console,'runtime_progress',now=36.,phase='waiting_for_qualified_target',
        vision={'stage':'waiting_upstream_window','entry_gate':{'reason':'insufficient_planning_time'}})
    assert '规划时间不足' in lines[-1]


def test_recent_persisted_blocker_is_shown_even_when_snapshot_is_waiting_for_rgbd():
    lines=[];console=ConsoleProgress(write=lines.append)
    send(console,'phase_started',now=0.,phase='waiting_for_qualified_target')
    vision={'stage':'waiting_rgbd','tracks':[{'id':1}],'frames_received':100,
        'snapshot_monotonic_s':10.,'entry_gate':None,
        'last_blocker':{'stage':'waiting_upstream_window','reason':'insufficient_planning_time','monotonic_s':9.9}}
    send(console,'runtime_progress',now=10.,phase='waiting_for_qualified_target',vision=vision)
    send(console,'runtime_progress',now=12.,phase='waiting_for_qualified_target',vision=vision)
    assert '规划时间不足' in lines[-1]
    # Once tracking is gone, an old rejection must not masquerade as a live target.
    vision={**vision,'tracks':[],'snapshot_monotonic_s':20.}
    send(console,'runtime_progress',now=20.,phase='waiting_for_qualified_target',vision=vision)
    send(console,'runtime_progress',now=22.,phase='waiting_for_qualified_target',vision=vision)
    assert '等待检测到工件' in lines[-1]


def test_belt_calibration_is_not_reported_as_missing_workpiece():
    lines=[];console=ConsoleProgress(write=lines.append)
    send(console,'phase_started',now=0.,phase='waiting_for_qualified_target')
    v={'stage':'waiting_rgbd','frames_received':10,'tracks':[],
       'perception':{'plane_calibration':{'ready':False,'frames':8,'required_frames':15}}}
    send(console,'runtime_progress',now=2.,phase='waiting_for_qualified_target',vision=v)
    send(console,'runtime_progress',now=4.,phase='waiting_for_qualified_target',vision=v)
    assert '实测带面初始化 8/15' in lines[-1]


@pytest.mark.parametrize('rpy',[(.2,-.3,.5),(.4,np.pi/2,.7),(.4,-np.pi/2,.7)])
def test_displayed_euler_convention_matches_robot_geometry(rpy):
    from mujoco_conveyor_sorting.calibration.geometry import pose_xyz_rpy
    import re
    lines=[]; console=ConsoleProgress(write=lines.append)
    pose=pose_xyz_rpy([0.,0.,0.,*rpy])
    execution(console,'planning_delivery',selected_grasp_tcp_base=pose.tolist(),planning_wall_s=.1)
    angles=[float(v) for v in re.search(r'RPY=\(([^)]*)\)',lines[0]).group(1).split(',')]
    reconstructed=pose_xyz_rpy([0.,0.,0.,*np.deg2rad(angles)])
    np.testing.assert_allclose(reconstructed,pose,atol=.002)


def test_console_output_runs_in_writer_and_cannot_block_producer_or_lose_full_records(tmp_path):
    entered, release=Event(),Event()
    threads=[]
    def slow_terminal(row):
        threads.append(current_thread().name)
        entered.set()
        assert release.wait(2.)
    journal=RunJournal(tmp_path,capacity=1,observer=slow_terminal)
    try:
        assert journal.emit('first',detailed_array=list(range(512)))
        assert entered.wait(2.)
        start=time.monotonic()
        assert journal.emit('second')
        assert not journal.emit('queue_full')
        assert time.monotonic()-start<.1
    finally:
        release.set()
        summary=journal.close()
    assert threads==['two-second-log-writer']*2
    assert not summary['observer_errors'] and summary['dropped']==1
    assert json.loads(journal.path.read_text().splitlines()[0])['data']['detailed_array']==list(range(512))


def test_terminal_fault_does_not_destroy_persistent_journal(tmp_path):
    def broken_terminal(row): raise OSError('injected broken terminal')
    journal=RunJournal(tmp_path,observer=broken_terminal)
    assert journal.emit('test',detail={'source_hash':'preserved'})
    summary=journal.close()
    assert summary['written']==1 and summary['errors']==[]
    assert 'injected broken terminal' in summary['observer_errors'][0]
    assert json.loads(journal.path.read_text())['data']['detail']['source_hash']=='preserved'


def test_failure_is_short_but_keeps_current_step_and_log_location(tmp_path):
    lines=[]; console=ConsoleProgress(write=lines.append)
    send(console,'run_failed',phase='planning_and_execution',execution_stage='planning_intercept',
        error='未找到可行抓取轨迹',details={'candidates':list(range(512))})
    console.finish({'status':'stopped','error':'未找到可行抓取轨迹'},tmp_path)
    assert len(lines)==2
    assert lines[0]=='[停止] 抓取轨迹规划：未找到可行抓取轨迹'
    assert lines[1]==f'[日志] {tmp_path.resolve()}'
