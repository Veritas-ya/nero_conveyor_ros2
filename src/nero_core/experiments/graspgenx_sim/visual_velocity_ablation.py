"""Evaluation methods: 3D cloud translation ICP and RGB-only LK optical flow.

These are experimental, independent measurement methods; they do not change
Nero7 or a robot controller. RGB uses a colour-only orange fixture mask to initialise the target, and
projects pixel rays to a fixed calibrated belt plane.
"""
import time
import numpy as np
import cv2
from scipy.spatial import cKDTree


def cloud_translation(prev, current, dt, *, limit=2000, iterations=12):
    start=time.perf_counter()
    a=np.asarray(prev,float);b=np.asarray(current,float)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1:]!=(3,) or b.shape[1:]!=(3,) or min(len(a),len(b))<30 or not np.isfinite(dt) or dt<=0:
        return None,{'reason':'insufficient_cloud_or_time'}
    a=a[np.isfinite(a).all(axis=1)];b=b[np.isfinite(b).all(axis=1)]
    if min(len(a),len(b))<30:return None,{'reason':'insufficient_finite_points'}
    # Deterministic subsampling; initialise with measured visible medians, not a belt-speed prior.
    a=a[np.linspace(0,len(a)-1,min(limit,len(a)),dtype=int)]
    b=b[np.linspace(0,len(b)-1,min(limit,len(b)),dtype=int)]
    shift=np.median(b,axis=0)-np.median(a,axis=0);tree=cKDTree(b)
    for _ in range(iterations):
        dist,idx=tree.query(a+shift,k=1); keep=np.isfinite(dist)&(dist<.015)
        if keep.sum()<30:return None,{'reason':'insufficient_icp_overlap','wall_s':time.perf_counter()-start}
        delta=np.median(b[idx[keep]]-(a[keep]+shift),axis=0);shift+=delta
        if np.linalg.norm(delta)<1e-5:break
    return shift/float(dt),{'reason':'ok','wall_s':time.perf_counter()-start,'matches':int(keep.sum()),'median_residual_m':float(np.median(dist[keep])),'translation_m':shift.tolist()}


def rgb_orange_fixture_mask(rgb):
    """RGB-only orange-object initialization for this simulated fixture."""
    hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
    raw=cv2.inRange(hsv,np.array([0,90,40],np.uint8),np.array([25,255,255],np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(raw,8)
    if count<2: return np.zeros(raw.shape,dtype=bool)
    index=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    if stats[index,cv2.CC_STAT_AREA]<30:return np.zeros(raw.shape,dtype=bool)
    return labels==index


def ray_intersections(uv,K,T_world_cv,plane_z):
    uv=np.asarray(uv,float).reshape(-1,2); K=np.asarray(K,float);T=np.asarray(T_world_cv,float)
    rays=np.column_stack(((uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(len(uv))))@T[:3,:3].T
    scale=(plane_z-T[2,3])/rays[:,2]
    return T[:3,3]+rays*scale[:,None]


class RGBOpticalFlow:
    def __init__(self):self.previous_gray=None;self.points=None;self.previous_stamp=None;self.track_id=None
    def update(self,rgb,mask,stamp,K,T_world_cv,plane_z,track_id):
        start=time.perf_counter();gray=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2GRAY)
        def initialize():
            # Only first frame uses the RGB-only colour mask.
            self.points=cv2.goodFeaturesToTrack(gray,maxCorners=150,qualityLevel=.01,minDistance=4,mask=np.asarray(mask,np.uint8)*255,blockSize=5)
            self.previous_gray=gray;self.previous_stamp=float(stamp);self.track_id=track_id
        if self.previous_gray is None or self.track_id!=track_id:
            initialize();return None,{'reason':'initialized_rgb_mask','features':0 if self.points is None else len(self.points),'wall_s':time.perf_counter()-start}
        dt=float(stamp)-self.previous_stamp
        if dt<=0 or dt>.25 or self.points is None or len(self.points)<4:
            # Do not reinitialize within a track: coverage loss is reported.
            self.previous_gray=gray;self.previous_stamp=float(stamp);self.points=None
            return None,{'reason':'rgb_track_unavailable','wall_s':time.perf_counter()-start}
        p1,status,_=cv2.calcOpticalFlowPyrLK(self.previous_gray,gray,self.points,None,winSize=(21,21),maxLevel=3)
        if p1 is None: self.points=None;return None,{'reason':'lk_failed','wall_s':time.perf_counter()-start}
        back,backstatus,_=cv2.calcOpticalFlowPyrLK(gray,self.previous_gray,p1,None,winSize=(21,21),maxLevel=3)
        if back is None:self.points=None;return None,{'reason':'lk_back_failed','wall_s':time.perf_counter()-start}
        p0=self.points.reshape(-1,2);p1=p1.reshape(-1,2);back=back.reshape(-1,2)
        keep=(status.ravel()>0)&(backstatus.ravel()>0)&(np.linalg.norm(back-p0,axis=1)<1.)
        self.previous_gray=gray;self.previous_stamp=float(stamp);self.points=p1[keep].reshape(-1,1,2).astype(np.float32)
        if keep.sum()<4:return None,{'reason':'insufficient_rgb_features','features':int(keep.sum()),'wall_s':time.perf_counter()-start}
        p0=p0[keep];p1=p1[keep];displacement=p1-p0;med=np.median(displacement,axis=0); residual=np.linalg.norm(displacement-med,axis=1)
        good=residual <= max(1.,3.*float(np.median(residual)))
        if good.sum()<4:return None,{'reason':'inconsistent_rgb_flow','wall_s':time.perf_counter()-start}
        before=ray_intersections(p0[good],K,T_world_cv,plane_z);after=ray_intersections(p1[good],K,T_world_cv,plane_z)
        vel=np.median(after-before,axis=0)/dt
        return vel,{'reason':'ok','wall_s':time.perf_counter()-start,'features':int(good.sum()),'flow_median_px':med.tolist(),'calibration':'fixed_belt_plane_no_depth_after_initialization'}
