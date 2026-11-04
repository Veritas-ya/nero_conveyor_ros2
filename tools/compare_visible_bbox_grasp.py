"""Offline geometric prefilter comparison using one saved network output.

No IK, path certification, physical contact, or network rescoring is done.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import hashlib
import numpy as np
import yaml
from experiments.graspgenx_sim.bbox_center_grasp import center_network_tcp_on_visible_box, center_network_tcp_on_visible_obb
from experiments.graspgenx_sim.candidates import _assets, _geometric_directions

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/vision_network_warm_20261027')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); prefix='track_0000_try_01'
    cloud_path=args.run/'inputs'/f'{prefix}.npz'
    pose_path=args.run/'inference'/f'{prefix}.npz'
    cloud=np.load(cloud_path,allow_pickle=False)['points']
    stored=np.load(pose_path,allow_pickle=False); poses=stored['poses'];scores=stored['scores']
    moved,info=center_network_tcp_on_visible_box(cloud,poses)
    obb_moved,obb_info=center_network_tcp_on_visible_obb(cloud,poses)
    p=ROOT/'src/nero_core/experiments/graspgenx_sim/gripper_agx'
    transform,vertices,_=_assets(str(p/'transforms.json'),str(p/'sweep_volume_params.json'))
    cfg=yaml.safe_load((ROOT/'src/nero_core/experiments/graspgenx_sim/nero7_config.yaml').read_text())['graspgenx']
    # Simulation scene configuration defines the belt support, not a real site.
    scene=yaml.safe_load((ROOT/'src/nero_core/mujoco_conveyor_sorting/configs/scene.yaml').read_text())
    belt_height=float(scene['conveyor']['surface_height'])
    variants=[('original',np.eye(4)),('parallel_jaw_z_pi',np.diag([-1.,-1.,1.,1.]))]
    def evaluate(value):
        accepted,rejected=_geometric_directions(value,scores,variants,canonical_to_tcp=transform,
            finger_vertices=vertices,belt_height=belt_height,minimum_score=cfg['minimum_score'],
            tilt_limit=cfg['maximum_approach_tilt_deg'],minimum_height=cfg['minimum_tcp_height_above_belt_m'],
            clearance=cfg['fingertip_clearance_m'])
        return {'evaluated_directions':len(accepted)+len(rejected),'passed_directions':len(accepted),
            'passed_distinct_network_poses':len({r['source_index'] for r in accepted}),
            'rejection_reasons':dict(Counter(r['reason'] for r in rejected))}
    report={'input_cloud_sha256':digest(cloud_path),'network_output_sha256':digest(pose_path),
        'source':'saved_simulated_rgbd_and_network_output','belt_height_sim_world_m':belt_height,
        'scope':'SE3, score, TCP height and open-finger belt clearance only; not IK, full path or physics',
        'original':evaluate(poses),'visible_bbox_center':evaluate(moved),'center_info':info,
        'visible_obb_center':evaluate(obb_moved),'obb_info':obb_info}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
