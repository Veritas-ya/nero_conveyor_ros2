"""Geometry-only experimental top-down grasp, with no network or score."""
from pathlib import Path
import json
import numpy as np
from .bbox_center_grasp import visible_belt_aligned_obb


def obb_top_down_grasp(points_world, *, closing_axis="short"):
    """Use visible OBB center as ee_site; align closing axis to short XY side.

    The canonical +Z points down; this is one chosen top-down interpretation
    of an OBB, whose signs and approach direction are intrinsically ambiguous.
    """
    box = visible_belt_aligned_obb(points_world)
    axes=np.asarray(box['axes_world_columns']); extent=np.asarray(box['extent_m'])
    if closing_axis not in ('short','long'):
        raise ValueError('closing_axis must be short or long')
    axis_index = int(np.argmin(extent[:2]) if closing_axis == 'short' else np.argmax(extent[:2]))
    x=axes[:,axis_index]; z=np.array([0.,0.,-1.]); y=np.cross(z,x)
    canonical=np.eye(4); canonical[:3,:3]=np.column_stack([x,y,z])
    transform=np.asarray(json.loads((Path(__file__).parent/'gripper_agx/transforms.json').read_text())['T_canonical_ee_site'])
    canonical[:3,3]=np.asarray(box['center_world_m'])-canonical[:3,:3]@transform[:3,3]
    tcp=canonical@transform
    return {'box':box,'canonical':canonical,'tcp':tcp,'score':None,
            'source':'visible_obb_geometry_only', 'approach':'world_negative_z',
            'closing_axis':f'{closing_axis}_visible_obb_horizontal_axis'}


def select_obb_candidate(cloud, snapshot, position, cfg):
    """Run the existing candidate safety checks for both horizontal OBB axes.

    The shared selector requires a numeric sorting key. Zero is a neutral
    placeholder, not a confidence; its threshold is disabled for this source.
    """
    import time
    from .candidates import select_candidates
    started=time.perf_counter()
    grasp=obb_top_down_grasp(cloud)
    alternative=obb_top_down_grasp(cloud,closing_axis='long')
    poses=np.asarray([grasp['canonical'],alternative['canonical']]); keys=np.zeros(2)
    local_cfg=dict(cfg); local_cfg['minimum_score']=0.
    accepted,rejected=select_candidates(snapshot,poses,keys,position,local_cfg,observed_cloud_world=cloud,
        candidate_metadata={'candidate_sources':['obb_top','obb_top'],
                            'candidate_task_preferences':[{'group':0},{'group':1}]})
    for row in accepted:
        row['original_obb_pose_world']=row.pop('original_network_pose_world', poses[row['source_index']].copy())
        row['source']='visible_obb_geometry_only'
        row['obb_closing_axis']=('short','long')[row['source_index']]
        row['grasp_position_mode']='visible_obb_geometry_only'
        row['score_semantics']='ordering_placeholder_no_model_confidence'
        row['visible_box_center_world_m']=grasp['box']['center_world_m']
    return {'poses':poses,'scores':keys,'accepted_candidates':accepted,'rejected_candidates':rejected,
            'obb_info':grasp['box'], 'metadata':{'inference_seconds':0.,
            'candidate_filter_wall_s':time.perf_counter()-started,
            'candidate_snapshot_sim_s':snapshot.source_timestamp,
            'grasp_source':'visible_obb_geometry_only','network_called':False}}
