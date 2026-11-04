"""Offline saved-state scene-collision comparison. No device or trajectory sender."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import yaml
from .config import CellConfig
from .diagnostics import json_value
from .planning import calibrated_model


def load_policy(path):
    raw=yaml.safe_load(Path(path).read_text())
    if (not isinstance(raw,dict) or set(raw)!={'scope','scene_collision_action'} or
            raw['scope']!='offline_saved_state_only' or
            raw['scene_collision_action'] not in ('reject','record_only')):
        raise ValueError('仅接受离线保存状态策略：scope=offline_saved_state_only，action=reject或record_only')
    return raw['scene_collision_action']


def saved_cases(run):
    run=Path(run)
    summary=run/'summary.json'
    data=json.loads(summary.read_text())
    cases=[]
    for attempt in (data.get('details') or {}).get('failures',[]):
        detail=attempt.get('details') or {}
        if not isinstance(detail,dict) or detail.get('kind')!='scene_collision':continue
        q=np.asarray(detail['joints_rad'],float)
        points=np.asarray(detail['points_base_m'],float)
        width=float(detail['width_m'])
        if (q.shape!=(7,) or points.ndim!=2 or points.shape[1]!=3 or not len(points) or
                not np.isfinite(q).all() or not np.isfinite(points).all() or not np.isfinite(width) or width<0):
            raise ValueError('保存的关节/触发点/开口无效；不制造替代数据')
        cases.append({'candidate_index':attempt['candidate_index'],'geometry':detail['geometry'],
            'q':q,'points':points,'width':width,'logged_point_count':detail.get('point_count'),
            'recorded_sample_s':(attempt.get('last_progress') or {}).get('trajectory_time_s')})
    if not cases:raise ValueError('该记录没有含触发点和关节角的场景碰撞，无法做本对照')
    return cases,hashlib.sha256(summary.read_bytes()).hexdigest()


def check_saved_state(model, case, *, scene):
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    try:
        model.check_state(case['q'],case['width'],allow_finger_contact=True,check_scene=scene)
        return {'passed':True,'error':None,'details':None}
    except PickRejected as error:
        return {'passed':False,'error':str(error),'details':error.details}


def compare(run, config, action, *, model=None):
    if action not in ('reject','record_only'):raise ValueError('未知离线策略')
    cases,digest=saved_cases(run)
    model=calibrated_model(config) if model is None else model
    results=[]
    for case in cases:
        # This private offline model sees ONLY the recorded trigger points.
        # They are not a replacement for the missing complete scene cloud.
        model.scene=case['points'].copy();model.tree=cKDTree(model.scene)
        model.target=np.empty((0,3))
        enforced=check_saved_state(model,case,scene=True)
        model_only=check_saved_state(model,case,scene=False)
        selected=enforced if action=='reject' else model_only
        results.append({'candidate_number':case['candidate_index']+1,
            'saved_geometry':case['geometry'],'saved_sample_s':case['recorded_sample_s'],
            'recorded_trigger_points':len(case['points']),'logged_collision_point_count':case['logged_point_count'],
            'points_base_m':case['points'].tolist(),'scene_check':enforced,
            'without_scene_check':model_only,'selected_saved_state_passed':selected['passed']})
    return {'scope':'offline_saved_state_only','scene_collision_action':action,
        'source_run':str(Path(run).resolve()),'source_summary_sha256':digest,
        'cell_config_sha256':config.source_sha256,'cases':results,
        'hardware_connections':0,'commands_sent':0,'trajectory_computed':False,
        'full_scene_certified':False,'hardware_execution_authorized':False,
        'note':'仅对照保存的触发点/关节状态；场景告警仍记录。通过不代表完整路径无碰撞或可发送真机。'}


def main(argv=None):
    from ..move_to_wait import default_config
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--cell-config',type=Path)
    parser.add_argument('--policy-config',type=Path)
    parser.add_argument('--output',type=Path,required=True,help='新的离线对照目录')
    args=parser.parse_args(argv)
    try:
        cell_path=args.cell_config or default_config()
        config=CellConfig.load(cell_path)
        action=load_policy(args.policy_config or Path(cell_path).parent/'offline_scene_collision.yaml')
        if args.output.exists():raise ValueError('输出目录必须是新目录')
        report=compare(args.run,config,action)
        args.output.mkdir(parents=True,exist_ok=False)
        (args.output/'comparison.json').write_text(json.dumps(json_value(report),ensure_ascii=False,indent=2,allow_nan=False))
    except (ValueError,OSError,KeyError,TypeError) as error:
        print(f'[离线对照] 未完成：{error}',flush=True)
        return 2
    for case in report['cases']:
        outcome='记录告警，模型状态检查通过' if case['selected_saved_state_passed'] else '保留拒绝'
        print(f'[离线对照] 候选{case["candidate_number"]}：{outcome}（{case["saved_geometry"]}）',flush=True)
    print(f'[记录] {args.output.resolve()}/comparison.json；真机命令=0',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
