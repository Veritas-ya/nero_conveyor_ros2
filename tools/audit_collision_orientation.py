#!/usr/bin/env python3
"""Offline counterfactual at one saved collision sample; never sends commands."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.precise_ik import PrecisePoseSolver
from nero_core.two_second.collision_view import read_packet


def audit(config, record):
    packet=read_packet(record)
    d=packet['metadata']; context=d['geometry_context']
    q=np.asarray(d['joints_rad'],float); width=float(d['gripper_width_m'])
    cloud=packet['arrays']['checked_points_base_m']
    model=calibrated_model(config); model.set_state(q,width)
    geom=model.model.geom(d['geometry']).id
    before=model.tcp(); geom_before=model.data.geom_xmat[geom].reshape(3,3).copy()
    geom_pos_before=model.data.geom_xpos[geom].copy()
    before_hits=model.inside_geom(cloud,geom,d['collision_margin_m'])
    goal=np.asarray(context['selected_grasp_tcp_base'],float)
    goal_q=Rotation.from_matrix(goal[:3,:3]).as_quat()[[3,0,1,2]]
    solved=PrecisePoseSolver(model).solve(before[:3,3],goal_q,q)
    report={'source_record':str(Path(record).resolve()),'input_candidate':context['input_grasp_index']+1,
        'time_s':d['trajectory_elapsed_s'],'geometry':d['geometry'],
        'collision_margin_m':d['collision_margin_m'], 'hardware_connections':0,'commands_sent':0,
        'physical_execution':False,'path_certified':False,'scene_reconstructed':False,
        'before_hits':int(before_hits.sum()), 'ik_success':bool(solved.success),
        'before_joint_deg':np.rad2deg(q).tolist(), 'before_tcp':before.tolist(),
        'target_tcp_rotation':goal[:3,:3].tolist(),
        'angular_error_before_deg':float(np.rad2deg(Rotation.from_matrix(before[:3,:3].T@goal[:3,:3]).magnitude()))}
    if solved.success:
        model.set_state(solved.joint_positions,width)
        after=model.tcp(); geom_after=model.data.geom_xmat[geom].reshape(3,3).copy()
        geom_pos_after=model.data.geom_xpos[geom].copy()
        report.update(after_hits=int(model.inside_geom(cloud,geom,d['collision_margin_m']).sum()),
            after_joint_deg=np.rad2deg(solved.joint_positions).tolist(),after_tcp=after.tolist(),
            tcp_translation_mm=float(np.linalg.norm(after[:3,3]-before[:3,3])*1000),
            after_tcp_goal_error_deg=float(np.rad2deg(Rotation.from_matrix(after[:3,:3].T@goal[:3,:3]).magnitude())),
            geom_rotation_deg=float(np.rad2deg(Rotation.from_matrix(geom_before.T@geom_after).magnitude())),
            geom_translation_mm=float(np.linalg.norm(geom_pos_after-geom_pos_before)*1000))
        try:
            model.check_state(solved.joint_positions,width,allow_finger_contact=True)
            report['robot_state_self_check']='passed_without_replaying_belt_scene'
        except Exception as e:
            report['robot_state_self_check']=str(e)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--record',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(CellConfig.load(args.config),args.record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
    print(f"候选 {r['input_candidate']}：旋转前 {r['before_hits']} 点，旋转后 {r.get('after_hits','未取得逆解')} 点；未连接设备")
    return 0 if r['ik_success'] else 1

if __name__=='__main__':raise SystemExit(main())
