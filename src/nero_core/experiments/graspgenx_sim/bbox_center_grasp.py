"""Experimental visible-point box center with genuine network orientations.

The modified poses require the normal candidate and full-path checks.  A
network score belongs to the original pose, not to the translated one.
"""
from pathlib import Path
import json
import numpy as np

_TRANSFORMS = Path(__file__).resolve().parent / 'gripper_agx' / 'transforms.json'


def center_network_tcp_on_visible_box(points_world, poses_world, *, canonical_to_tcp=None):
    """Put each TCP at the world-axis-aligned box center of the visible cloud.

    The simulation declares its belt aligned with world XY and world Z. This
    is not an oriented box, reconstructed object center or hidden geometry.
    Poses are original GraspGen-X T_world_canonical transforms.
    """
    points=np.asarray(points_world,dtype=float)
    poses=np.asarray(poses_world,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=3 or len(points)<21
            or not np.isfinite(points).all()):
        raise ValueError('need at least 21 finite visible world points')
    if (poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses)
            or not np.isfinite(poses).all()):
        raise ValueError('need finite N x 4 x 4 network poses')
    rotations=poses[:,:3,:3]
    if (not np.allclose(poses[:,3,:], [0,0,0,1],atol=1e-7)
            or not np.allclose(rotations.transpose(0,2,1)@rotations,np.eye(3),atol=1e-4)
            or not np.allclose(np.linalg.det(rotations),1.,atol=1e-4)):
        raise ValueError('network poses must be rigid transforms')
    transform=(np.asarray(json.loads(_TRANSFORMS.read_text())['T_canonical_ee_site'],dtype=float)
               if canonical_to_tcp is None else np.asarray(canonical_to_tcp,dtype=float))
    if (transform.shape!=(4,4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0,0,0,1],atol=1e-7)
            or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-4)
            or not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-4)):
        raise ValueError('canonical-to-TCP must be a rigid transform')
    lower=points.min(axis=0);upper=points.max(axis=0);center=(lower+upper)/2.
    modified=poses.copy()
    modified[:,:3,3]=center-np.einsum('nij,j->ni',rotations,transform[:3,3])
    return modified, {'mode':'visible_world_axis_aligned_box_center_tcp',
        'source':'single_exposure_visible_target_points', 'coordinate_frame':'world',
        'bounds_min_m':lower.tolist(),'bounds_max_m':upper.tolist(),
        'tcp_center_world_m':center.tolist(), 'visible_point_count':len(points),
        'score_semantics':'original_network_pose_score_not_rescored_after_translation',
        'assumption':'box center is not established physical object center or grasp success',
        'tcp_definition':'simulated ee_site; not a measured physical TCP'}


def annotate_center_candidates(rows, original_poses, center_info):
    """Preserve provenance of the original network pose alongside selection."""
    for row in rows:
        index=int(row['source_index'])
        row['original_network_pose_world']=np.asarray(original_poses[index]).copy()
        row['grasp_position_mode']=center_info['mode']
        row['score_semantics']=center_info['score_semantics']
        row['visible_box_center_world_m']=list(center_info['tcp_center_world_m'])


from nero_core.two_second.grasp import visible_belt_aligned_obb


def center_network_tcp_on_visible_obb(points_world, poses_world, *, canonical_to_tcp=None):
    """Translate genuine network orientations to a visible belt-aligned OBB center."""
    # Reuse rigid-transform and TCP validation from the existing AABB path.
    modified, info = center_network_tcp_on_visible_box(points_world, poses_world,
                                                       canonical_to_tcp=canonical_to_tcp)
    box = visible_belt_aligned_obb(points_world)
    center = np.asarray(box['center_world_m'])
    modified[:, :3, 3] += center - np.asarray(info['tcp_center_world_m'])
    info.update(mode='visible_belt_aligned_obb_center_tcp',
                tcp_center_world_m=center.tolist(), obb=box,
                assumption='visible-surface OBB is not the physical object center or grasp success')
    return modified, info
