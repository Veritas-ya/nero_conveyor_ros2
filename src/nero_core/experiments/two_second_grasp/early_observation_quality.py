"""Frame-local, simulation-only quality and temporal OBB stability checks."""
from collections import deque
import numpy as np

from nero_core.two_second.grasp import VisibleOBBError, visible_belt_aligned_obb


def axis_angle(box):
    axes=np.asarray(box['axes_world_columns'],float)
    extent=np.asarray(box['extent_m'],float)
    axis=axes[:2,int(np.argmin(extent[:2]))]
    return float(np.arctan2(axis[1],axis[0]))


def nearly_square(box, *, aspect_threshold=1.10):
    extent=np.asarray(box['extent_m'],float)[:2]
    return bool(extent.min()>0 and extent.max()/extent.min()<=aspect_threshold)


def axial_difference(a, b, period):
    return float(abs((a-b+period/2)%period-period/2))


class EarlyObservationQuality:
    """Require recent independent frames; never combine points across frames.

    Mask/depth coverage is only a visible-image quality proxy and cannot prove
    the hidden geometry or material symmetry of an object.
    """
    def __init__(self, *, frames=4, min_span_s=.08, max_gap_s=.20,
                 angle_tolerance_deg=12., min_depth_fraction=.8, min_points=100,
                 border_margin_px=2, max_count_ratio=1.5):
        self.frames=int(frames);self.min_span_s=float(min_span_s);self.max_gap_s=float(max_gap_s)
        self.angle_tolerance_rad=np.deg2rad(angle_tolerance_deg)
        self.min_depth_fraction=float(min_depth_fraction);self.min_points=int(min_points)
        self.border_margin_px=int(border_margin_px);self.max_count_ratio=float(max_count_ratio)
        self.history=deque(maxlen=self.frames);self.identity=None

    def clear(self):
        self.history.clear();self.identity=None

    def update(self, *, identity, stamp, mask, depth, cloud):
        stamp=float(stamp);mask=np.asarray(mask,bool);depth=np.asarray(depth,float);cloud=np.asarray(cloud,float)
        if self.identity!=identity or (self.history and (stamp<=self.history[-1]['stamp'] or stamp-self.history[-1]['stamp']>self.max_gap_s)):
            self.history.clear()
        self.identity=identity
        quality={'ready':False,'reason':'invalid_frame'}
        if mask.ndim!=2 or depth.shape!=mask.shape or not np.isfinite(stamp):
            self.history.clear();return quality
        rows,cols=np.nonzero(mask);count=len(rows)
        if count==0:
            self.history.clear();return {'ready':False,'reason':'empty_mask'}
        height,width=mask.shape;edge=self.border_margin_px
        touches=bool(rows.min()<=edge or cols.min()<=edge or rows.max()>=height-1-edge or cols.max()>=width-1-edge)
        valid=np.isfinite(depth[mask]) & (depth[mask]>0.)
        fraction=float(valid.mean())
        quality.update(mask_pixels=count,depth_valid_fraction=fraction,touches_border=touches,point_count=int(len(cloud)))
        if touches or fraction<self.min_depth_fraction or len(cloud)<self.min_points:
            self.history.clear();quality['reason']='frame_quality';return quality
        try:box=visible_belt_aligned_obb(cloud)
        except (VisibleOBBError,ValueError):
            self.history.clear();quality['reason']='obb_unavailable';return quality
        self.history.append({'stamp':stamp,'angle':axis_angle(box),'square':nearly_square(box),
                             'count':int(len(cloud)),'extent_xy':np.sort(np.asarray(box['extent_m'])[:2])})
        quality.update(visible_extent_m=box['extent_m'],near_square=nearly_square(box),
                       stable_frames=len(self.history),frame_span_s=self.history[-1]['stamp']-self.history[0]['stamp'])
        if len(self.history)<self.frames or quality['frame_span_s']<self.min_span_s:
            quality['reason']='collecting_frames';return quality
        counts=[r['count'] for r in self.history]
        if max(counts)>self.max_count_ratio*min(counts):
            quality['reason']='visible_count_unstable';return quality
        extents=np.asarray([r['extent_xy'] for r in self.history])
        if np.any(extents.min(axis=0)<=0) or np.any(extents.max(axis=0)>1.20*extents.min(axis=0)):
            quality['reason']='visible_extent_unstable';return quality
        # A square has an intrinsically ambiguous 90-degree box-axis swap.
        period=np.pi/2 if all(r['square'] for r in self.history) else np.pi
        spread=max(axial_difference(r['angle'],self.history[-1]['angle'],period) for r in self.history)
        quality['axis_spread_deg']=float(np.rad2deg(spread))
        quality['orientation_unreliable']=all(r['square'] for r in self.history)
        if not quality['orientation_unreliable'] and spread>self.angle_tolerance_rad:
            quality['reason']='obb_axis_unstable';return quality
        quality.update(ready=True,reason='stable_visible_frames');return quality
