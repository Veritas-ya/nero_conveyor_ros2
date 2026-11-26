"""Independent low-speed staged-interception entry. No devices by default."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import yaml
from .config import CellConfig
from .diagnostics import json_value
from .hover_config import DEFAULTS, KEY, authorize
from .phase_timing_test import prepare_test, DEFAULT_BASE, ROOT
from .phase_timing_experiment import digest, source_hashes, write_new


def prepare_config(base, output, *, enabled=False, grasp=False):
    path=prepare_test(base,output,join_times=tuple(DEFAULTS['descent_times_s']),brake_times=(.4,))
    raw=yaml.safe_load(path.read_text())
    raw[KEY]={**DEFAULTS,'enabled':enabled,'grasp':grasp}
    raw.update(continuous_lookahead_enabled=False,carry_lookahead_enabled=False,
        grasp_workpiece_collision_check_enabled=True,runtime_sweep_mode='inline',
        brake_duration_s=.4)
    path.write_text(yaml.safe_dump(raw,allow_unicode=True,sort_keys=False))
    CellConfig.load(path)
    manifest=json.loads((path.parent/'experiment.json').read_text())
    manifest.update(schema='nero_hover_descent_hardware_test_v1',config_sha256=digest(path),
        algorithm='preplanned_sync_prefix_feedback_gated_descent_or_stop',hover_descent=raw[KEY],
        source_sha256=source_hashes(),physical_parameters_validated=False)
    (path.parent/'experiment.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return path


def prepare_replay_config(base, output):
    """Reuse archived experimental settings only for the device-free replay."""
    cfg=CellConfig.load(base)
    if 'phase_timing' not in cfg.raw:
        return prepare_config(base,output,enabled=True)
    raw=cfg.raw.copy()
    overrides={}
    if KEY not in raw:
        overrides=dict(continuous_lookahead_enabled=False,carry_lookahead_enabled=False,
            grasp_workpiece_collision_check_enabled=True,runtime_sweep_mode='inline',brake_duration_s=.4,
            phase_timing={**raw['phase_timing'],'join_candidates_s':list(DEFAULTS['descent_times_s']),
                          'brake_candidates_s':[.4]})
        raw.update(overrides)
    raw[KEY]={**DEFAULTS,**raw.get(KEY,{}),'enabled':True}
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('回放目录已存在：'+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        stage=Path(temp)/'replay';stage.mkdir();path=stage/'effective.yaml'
        path.write_text(yaml.safe_dump(raw,allow_unicode=True,sort_keys=False))
        CellConfig.load(path)
        write_new(stage/'experiment.json',dict(schema='nero_hover_descent_offline_replay_v1',
            source_config=str(cfg.source),source_config_sha256=cfg.source_sha256,
            replay_overrides=overrides,config_sha256=digest(path),
            source_sha256=source_hashes(),hardware_connections=0))
        stage.rename(output)
    return output/'effective.yaml'


def replay(config, run, output, speed):
    from .paper_replay import restore_input
    from .planning import MeasuredPlanner
    from .hover_planning import prepare_hover, HoverService, concatenate
    from .runtime_sweep import scene_snapshot
    cfg=CellConfig.load(config)
    first_cycle=Path(run)/'cycle_001/events.json'
    if first_cycle.exists() and any(r.get('event')=='hover_planning_submitted'
            for r in json.loads(first_cycle.read_text())):
        from .hover_replay import restore_hover_input
        target,state,info=restore_hover_input(run,cfg)
    else:
        target,state,info=restore_input(Path(run),cfg)
    recorded=np.asarray(target.velocity).copy()
    synthetic=not np.isclose(np.linalg.norm(recorded),speed,atol=1e-10,rtol=0)
    if synthetic:
        velocity=recorded/np.linalg.norm(recorded)*speed
        target=replace(target,velocity=velocity,speed_report={**target.speed_report,
            'configured_speed_m_s':speed,'prediction_velocity_base_m_s':velocity.tolist(),
            'synthetic_counterfactual':True})
    planner=MeasuredPlanner(cfg)
    service=HoverService(cfg)  # Exercise spawn/serialization without devices.
    started=time.monotonic()
    planner.clock=lambda:info['submission_s']+time.monotonic()-started
    result=dict(hardware_connections=0,commands_sent=0,physics_simulation=False,
        input=info,replay_speed_m_s=speed,synthetic_counterfactual=synthetic,
        synthetic_low_speed_m_s=speed if synthetic else None,original_velocity_m_s=recorded,
        physical_success=None,source_sha256=source_hashes())
    try:
        bundle=prepare_hover(planner,target,state)
        # Worker clock is the current host's monotonic time, whereas archived
        # reference times remain historical. Pass a prospective CPU deadline.
        remaining=(bundle.old.reference.start_s+bundle.switch_s-cfg.raw['command_advance_s']-
                   cfg.raw['hover_descent']['send_guard_s']-planner.clock())
        future=service.submit(scene_snapshot(planner,bundle.prefix),bundle,state,time.monotonic()+remaining)
        answer=future.result(timeout=10.)
        result.update(prefix=bundle.evidence,suffix=answer)
        for name,path in [('prefix',bundle.prefix),('stop',bundle.stopping)]:
            np.savez_compressed(Path(output).parent/(name+'.npz'),c=path.curve.c,x=path.curve.x)
        if answer['ok']:
            full=concatenate(bundle.prefix,answer['path'])
            np.savez_compressed(Path(output).parent/'combined.npz',c=full.curve.c,x=full.curve.x)
            result['suffix']={k:v for k,v in answer.items() if k not in ('path','reference')}
        result['status']='offline_plan_ready' if answer['ok'] else 'suffix_rejected_stop_available'
    except Exception as error:
        result.update(status='offline_rejected',error=str(error),details=getattr(error,'details',{}))
    finally:
        service.close()
    write_new(output,result)
    print(json.dumps(json_value({k:v for k,v in result.items() if k in
        ('status','error','hardware_connections','commands_sent','synthetic_low_speed_m_s')}),ensure_ascii=False))
    return 0 if result['status']=='offline_plan_ready' else 1


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,help='现场原配置；回放默认采用归档 effective.yaml')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--enable-hover-descent',action='store_true',help='显式启用新模式；原抓取入口不变')
    parser.add_argument('--grasp',action='store_true',help='允许同步验证通过后下降闭爪；默认仅同步后停车')
    action=parser.add_mutually_exclusive_group()
    action.add_argument('--execute',action='store_true',help='连接真实相机/CAN；不会自动使能')
    action.add_argument('--replay-run',type=Path,help='离线归档输入；速度变化会标记为假设回放')
    action.add_argument('--check-config',action='store_true')
    parser.add_argument('--belt-speed-cm-s',type=float)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(argv)
    try:
        if (args.execute or args.replay_run or args.grasp) and not args.enable_hover_descent:
            raise ValueError('请显式添加 --enable-hover-descent；默认不执行新模式')
        speed=args.belt_speed_cm_s
        if (args.execute or args.replay_run) and speed is None:
            raise ValueError('必须提供 --belt-speed-cm-s，真机须与实际传送带速度一致')
        if speed is not None and (not np.isfinite(speed) or not 0<speed<=20):
            raise ValueError('首版分段接续只允许0至20cm/s')
        output=args.output or ROOT/'runs/hover_descent_test'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        archived=args.replay_run.parent/'effective.yaml' if args.replay_run else None
        base=args.config or (archived if archived is not None and archived.is_file() else DEFAULT_BASE)
        config=(prepare_replay_config(base,output) if args.replay_run else
                prepare_config(base,output,enabled=args.enable_hover_descent,grasp=args.grasp))
        write_new(config.parent/'invocation.json',dict(arguments=vars(args),effective_config=str(config),
            execution_requested=args.execute,hardware_execution_verified=False))
        if args.replay_run:
            return replay(config,args.replay_run,config.parent/'replay.json',speed/100.)
        if not args.execute:
            print(json.dumps(dict(status='configuration_valid',mode_enabled=args.enable_hover_descent,
                grasp_enabled=args.grasp,config=str(config),hardware_connections=0,commands_sent=0),ensure_ascii=False))
            return 0
        from .main import main as hardware_main
        forwarded=['--config',str(config),'--output',str(config.parent/'hardware'),'--execute',
            '--count','1','--belt-speed-cm-s',str(speed),'--record-rgbd']
        if args.preview: forwarded.append('--preview')
        return hardware_main(forwarded,freeze_source='single_exposure_native_depth',
            phase_timing_hardware_test=True,hover_descent_entry=True)
    except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
        print(json.dumps(dict(status='hover_entry_rejected',error=str(error),
            hardware_execution_verified=False),ensure_ascii=False))
        return 2


if __name__=='__main__':
    raise SystemExit(main())
