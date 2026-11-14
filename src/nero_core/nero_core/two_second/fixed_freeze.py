"""Single-exposure depth trigger for the independent camera test interface.

No tracker, device, CAN adapter or robot command is owned by this module.
The locator uses a wide search crop; the 2 cm band selects a component's
centre and never slices the workpiece cloud into a narrow strip.
"""
from collections import deque
from dataclasses import dataclass,replace
from threading import Condition
import time

import cv2
import numpy as np

from mujoco_conveyor_sorting.perception import DepthGeometryVisionPipeline, GeometryPrototype
from .reference import EntryNotReady
from .synchronized_reference import admit_synchronized_observation, freeze_window_for_speed


class ExposureBuffer:
    """Bounded original-frame FIFO, independent of expensive geometry/planning."""
    def __init__(self, capacity=16):
        if isinstance(capacity,bool) or not isinstance(capacity,int) or not 3<=capacity<=90:
            raise ValueError('buffer frames must be an integer from 3 to 90')
        self.capacity=capacity
        self.frames=deque()
        self.condition=Condition()
        self.closed=False
        self.received=self.dropped=self.sequence_skips=0
        self.previous=None

    def put(self,frame):
        key=(frame.header.clock_epoch,frame.header.sequence)
        with self.condition:
            if self.closed:return
            if self.previous is not None and key[0]==self.previous[0]:
                if key[1]<=self.previous[1]:return
                self.sequence_skips+=max(0,key[1]-self.previous[1]-1)
            if len(self.frames)==self.capacity:
                self.frames.popleft();self.dropped+=1
            self.frames.append(frame);self.previous=key;self.received+=1
            self.condition.notify_all()

    def take(self,timeout=.2):
        end=time.monotonic()+timeout
        with self.condition:
            while not self.frames and not self.closed:
                left=end-time.monotonic()
                if left<=0:return None
                self.condition.wait(left)
            return self.frames.popleft() if self.frames else None

    def close(self):
        with self.condition:
            self.closed=True;self.condition.notify_all()

    def diagnostics(self):
        with self.condition:
            return dict(capacity=self.capacity,queued=len(self.frames),received=self.received,
                        dropped=self.dropped,capture_sequence_skips=self.sequence_skips)


def capture_admission(config,capture_s,position,velocity,now_s):
    """Check the true saved exposure, without projecting or retimestamping it."""
    if not 0<=now_s-capture_s<=config.raw['maximum_observation_age_s']:
        raise EntryNotReady('原始曝光已过期或来自未来','stale_exposure')
    window=freeze_window_for_speed(config,float(velocity@config.direction))
    return admit_synchronized_observation(capture_s,float(position@config.direction),
        float(velocity@config.direction),config.follow_bounds,now_s,window['distance_m'],
        minimum_upstream_distance_m=window['minimum_distance_m'])


def _clip_polygon(polygon,axis,limit,keep_greater):
    result=[]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        da,db=float(a@axis-limit),float(b@axis-limit)
        ina=da>=0 if keep_greater else da<=0
        inb=db>=0 if keep_greater else db<=0
        if ina:result.append(a)
        if ina!=inb:result.append(a+(b-a)*(da/(da-db)))
    return np.asarray(result).reshape(-1,3)


@dataclass(frozen=True)
class DepthCandidate:
    position: np.ndarray
    point_count: int
    seed_pixel_indices: np.ndarray


def fixed_freeze_detection_surface(surface):
    """Search the measured belt hull, including its sparse upstream edge.

    The native-depth 2/98% quantiles are useful for the original tracker but
    can exclude a configured freeze line that still has measured belt depth.
    This private detection copy adds no points and changes no plane, hull,
    scene boundary or collision geometry.
    """
    hull=np.asarray(surface.hull_xy).reshape(-1,2)
    bounds=np.column_stack((hull.min(axis=0),hull.max(axis=0)))
    bounds[:,0]=np.minimum(bounds[:,0],surface.bounds_xy_m[:,0])
    bounds[:,1]=np.maximum(bounds[:,1],surface.bounds_xy_m[:,1])
    return replace(surface,bounds_xy_m=bounds,
        scene_bounds_xy_m=(surface.bounds_xy_m.copy() if surface.scene_bounds_xy_m is None
                           else surface.scene_bounds_xy_m))


class NativeDepthLocator:
    """Fast single-frame native-depth components around the freezing line.

    This is a trigger only. The selected exposure subsequently passes the
    existing complete workpiece/colour/geometry detector before planning.
    """
    def __init__(self,config,surface,frame):
        self.original_bounds=surface.bounds_xy_m.copy()
        surface=fixed_freeze_detection_surface(surface)
        self.config,self.surface=config,surface
        if np.any(np.abs(frame.depth_intrinsics.distortion)>1.e-10):
            raise ValueError('固定冻结测试需要零畸变原生深度流')
        self.crop=self._crop(frame)
        y0,y1,x0,x1=self.crop
        y,x=np.mgrid[y0:y1,x0:x1]
        k=frame.depth_intrinsics
        rays=np.stack(((x-k.cx)/k.fx,(y-k.cy)/k.fy,np.ones_like(x)),axis=-1)
        self.rays_base=rays@frame.camera_to_world_cv[:3,:3].T
        self.origin=np.asarray(frame.camera_to_world_cv[:3,3])
        self.pipeline=DepthGeometryVisionPipeline(
            [GeometryPrototype(category_id=0,dimensions=(.05,.05,.05),symmetric=True)],
            support_plane_height=0.,roi_x=surface.bounds_xy_m[0],roi_y=surface.bounds_xy_m[1],
            minimum_height=.012,maximum_height=config.raw['maximum_target_height_m'],
            minimum_mask_pixels=70,minimum_footprint=.012,
            maximum_footprint=config.raw.get('maximum_target_length_m', .16),
            classification_enabled=False)
        self.intrinsics=k.matrix.copy();self.intrinsics[:2,2]-=[x0,y0]

    def _crop(self,frame):
        plane=self.surface.plane_to_base
        bounds=self.surface.bounds_xy_m
        rectangle=np.array([[bounds[0,0],bounds[1,0],0.],
                            [bounds[0,1],bounds[1,0],0.],
                            [bounds[0,1],bounds[1,1],0.],
                            [bounds[0,0],bounds[1,1],0.]])
        rectangle=rectangle@plane[:3,:3].T+plane[:3,3]
        coordinate=float(self.config.follow_bounds[0]-self.config.raw['freeze_position_upstream_m'])
        # Preserve the legacy crop at 16cm; larger supported parts need enough
        # room for their complete native-depth component, never a sliced cloud.
        extra=max(0., self.config.raw.get('maximum_target_length_m', .16)-.16)
        polygon=_clip_polygon(rectangle,self.config.direction,coordinate-.20-extra,True)
        if len(polygon):polygon=_clip_polygon(polygon,self.config.direction,coordinate+.22+extra,False)
        if not len(polygon):raise ValueError('冻结线及搜索范围没有落在实测传送带视野内')
        prism=np.vstack((polygon,polygon+plane[:3,2]*self.config.raw['maximum_target_height_m']))
        optical=(prism-frame.camera_to_world_cv[:3,3])@frame.camera_to_world_cv[:3,:3]
        if np.any(optical[:,2]<=0):raise ValueError('冻结搜索范围位于相机后方')
        uv=optical[:,:2]/optical[:,2,None]
        k=frame.depth_intrinsics
        uv=uv*[k.fx,k.fy]+[k.cx,k.cy]
        lo=np.floor(uv.min(axis=0)-4).astype(int);hi=np.ceil(uv.max(axis=0)+5).astype(int)
        x0,y0=np.maximum(lo,0);x1,y1=np.minimum(hi,[k.width,k.height])
        if x1<=x0 or y1<=y0:raise ValueError('冻结搜索范围不在原生深度画面内')
        return int(y0),int(y1),int(x0),int(x1)

    def locate(self,frame,*,robot_model=None,robot_state=None):
        y0,y1,x0,x1=self.crop
        z=frame.depth_m[y0:y1,x0:x1]
        base=self.rays_base*z[...,None]+self.origin
        plane=self.surface.plane_to_base
        local=(base-plane[:3,3])@plane[:3,:3]
        foreground=np.isfinite(z)&(z>0)&np.isfinite(local).all(axis=2)
        foreground&=(local[:,:,2]>=.012)&(local[:,:,2]<=self.config.raw['maximum_target_height_m'])
        if (robot_model is None)!=(robot_state is None):
            raise ValueError('自身点云过滤需要真实模型和配对反馈同时提供')
        removed=0
        if robot_model is not None and foreground.any():
            filtered=robot_model.classify_robot_points(base[foreground],robot_state,
                camera_origin=frame.camera_to_world_cv[:3,3])
            removed=int(np.count_nonzero(filtered.removed_mask))
            clean=foreground.copy();clean[foreground]=~filtered.removed_mask;foreground=clean
        components=self.pipeline.extract_components(None,z,self.intrinsics,np.eye(4),
            points_world=local,foreground_mask=foreground,roi_selects_components=True)
        candidates=[];partial=0;near_zone_components=0;size_decisions=[]
        line=self.config.follow_bounds[0]-self.config.raw['freeze_position_upstream_m']
        cutoff=line+self.config.raw['freeze_position_tolerance_m']
        for component in components:
            decision={'estimated_footprint_m':component.feature_vector[:2].tolist(),
                      'maximum_length_m':self.pipeline.maximum_footprint, 'accepted':False}
            size_decisions.append(decision)
            mask=component.mask
            if cv2.pointPolygonTest(self.surface.hull_xy,tuple(map(float,component.position[:2])),True)<=.025:
                decision['reason']='belt_edge'
                continue
            rows,cols=np.nonzero(mask)
            position=np.median(base[mask],axis=0)
            if line-.08<=position@self.config.direction<=cutoff+.08:near_zone_components+=1
            if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():
                decision['reason']='partial_component_at_crop'
                partial+=1;continue
            decision.update(accepted=True,reason='size_and_visibility_passed',
                            distance_to_entry_m=float(self.config.follow_bounds[0]-position@self.config.direction))
            candidates.append(DepthCandidate(position,len(rows),
                (rows+y0)*frame.depth_intrinsics.width+cols+x0))
        return candidates,dict(search_crop_yxyx=self.crop,candidate_count=len(candidates),
            detection_bounds_xy_m=self.surface.bounds_xy_m.tolist(),
            original_quantile_bounds_xy_m=self.original_bounds.tolist(),
            detection_bounds_source='measured_native_belt_hull',
            partial_components=partial,evaluated_depth_pixels=int(z.size),
            native_depth_pixels=int(frame.depth_m.size),
            native_valid_depth_ratio=float(np.mean(np.isfinite(z)&(z>0))),
            near_freeze_zone_components=near_zone_components,robot_points_removed=removed,
            maximum_target_length_m=self.pipeline.maximum_footprint,
            size_decisions=size_decisions[:32],
            component_extraction=self.pipeline.last_component_workload)


def frozen_geometry(config,frame,d2c,surface,candidate,capture_s,*,clock=time.monotonic,
                    robot_model=None,robot_state=None,detect_fn=None,grasp_fn=None):
    """Shared complete single-exposure geometry for camera tests and hardware.

    Hardware passes measured feedback and its private model; camera-only
    callers never supply an assumed state as an actual observation.
    """
    from .measured_rgbd import fit_plane,green_belt_points
    from .velocity_source import prediction_velocity
    from .vision import obb_grasps
    from mujoco_conveyor_sorting.runtime.single_pick import detect_target,PickRejected
    if (robot_model is None)!=(robot_state is None):
        raise ValueError('完整工件自身过滤需要配对反馈')
    detection_surface=fixed_freeze_detection_surface(surface)
    detector=detect_fn or detect_target
    if detect_fn is None and config.raw.get('local_target_geometry_enabled', True):
        from mujoco_conveyor_sorting.runtime.local_target_geometry import detect_seed_local
        original_detector=detector
        detector=lambda *args,**kwargs: detect_seed_local(*args,detector=original_detector,**kwargs)
    detection=detector(frame,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points,
        target_shape='simple',measured_surface=detection_surface,robot_model=robot_model,robot_state=robot_state,
        target_seed_pixel_indices=candidate.seed_pixel_indices,
        maximum_target_height_m=config.raw['maximum_target_height_m'],
        maximum_target_length_m=config.raw.get('maximum_target_length_m', .16),
        scene_scope=config.raw['environment_collision_scope'])
    overlap=float(np.mean(detection['mask'].ravel()[candidate.seed_pixel_indices]))
    if overlap<.5:
        raise PickRejected('完整工件分割与本帧触发区域不一致',details={'seed_overlap_ratio':overlap})
    # Downstream planning retains the original measured scene geometry. Only
    # target search used the wider native-depth hull, never inferred depth.
    detection['belt_region']={**surface.region_metadata(),
        'target_search_bounds_xy_m':detection_surface.bounds_xy_m.copy(),
        'target_search_bounds_source':'measured_native_belt_hull'}
    position=np.median(detection['object_points'],axis=0)
    velocity=prediction_velocity(config,None,surface.normal_base)
    capture_admission(config,capture_s,position,velocity,clock())
    grasps,box=(grasp_fn or obb_grasps)(detection['object_points'],surface.plane_to_base,config.canonical_tcp)
    detection.update(obb=box,frame=frame,depth_to_color_cv=d2c,scene_state=robot_state)
    return detection,position,velocity,tuple(grasps),overlap
