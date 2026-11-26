"""Measured native pixels, one exposure at a time. All distances are metres."""
from dataclasses import dataclass
import cv2
import numpy as np
from scipy.spatial import cKDTree

from mujoco_conveyor_sorting.runtime.frame_products import green_rgb_mask, world_points
from mujoco_conveyor_sorting.runtime.shape_pick import plane_pixel_steps
from mujoco_conveyor_sorting.runtime.visible_target import selected_rgb_silhouette, visible_depth_mask
from nero_core.two_second.measured_rgbd import project_to_rgb
from nero_core.two_second.vision import obb_grasps
from nero_core.two_second.grasp import VisibleOBBError


@dataclass(frozen=True)
class Parameters:
    rgb_area_px: int = 100
    rgb_depth_fraction: float = .20
    rgb_depth_pixels: int = 20
    min_height_m: float = .008
    min_points: int = 30
    min_width_m: float = .006
    edge_m: float = .010
    length_diagnostic_m: float = .160
    association_gate_m: float = .030

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (
                self.rgb_area_px, self.rgb_depth_pixels, self.min_points)):
            raise ValueError('pixel and point counts must be positive integers')
        scales = (self.rgb_depth_fraction, self.min_height_m, self.min_width_m,
                  self.edge_m, self.length_diagnostic_m, self.association_gate_m)
        if not all(np.isfinite(x) and x > 0 for x in scales) or self.rgb_depth_fraction > 1:
            raise ValueError('invalid native observation parameters')


@dataclass
class Region:
    indices: np.ndarray
    points: np.ndarray
    uv: np.ndarray


@dataclass
class Observation:
    indices: np.ndarray
    points: np.ndarray
    center: np.ndarray
    grasps: tuple
    box: dict
    uv_center: np.ndarray
    extent_m: tuple


def project_base(points, frame, d2c):
    t = frame.camera_to_world_cv
    uv, _ = project_to_rgb((points-t[:3, 3])@t[:3, :3], frame, d2c)
    return uv


class NativeGeometry:
    def __init__(self, surface, frame, config, parameters=Parameters()):
        self.surface, self.config, self.p = surface, config, parameters
        self.hull = surface.detection_hull_xy
        if self.hull is None:
            self.hull = surface.hull_xy
        self.hull = np.asarray(self.hull, np.float32).reshape(-1,2)
        self.plane = surface.plane_to_base
        self.height = config.raw['maximum_target_height_m']
        k = frame.depth_intrinsics
        if np.any(np.abs(k.distortion)>1e-10):
            raise ValueError('Native observation requires rectified depth')
        # Project the full measured belt prism, including tall pieces at its edge.
        hull3 = np.vstack([np.c_[self.hull, np.full(len(self.hull), z)] for z in (0., self.height)])
        base = hull3@self.plane[:3,:3].T + self.plane[:3,3]
        t = frame.camera_to_world_cv
        cam = (base-t[:3,3])@t[:3,:3]
        pix = cam[:,:2]/cam[:,2:3]*[k.fx,k.fy]+[k.cx,k.cy]
        x0,y0=np.maximum(0,np.floor(pix.min(axis=0)).astype(int)-2)
        x1,y1=np.minimum([k.width,k.height],np.ceil(pix.max(axis=0)).astype(int)+3)
        self.crop=(y0,y1,x0,x1)
        y,x=np.mgrid[y0:y1,x0:x1]
        self.rays=np.stack(((x-k.cx)/k.fx,(y-k.cy)/k.fy,np.ones_like(x)),axis=-1)
        self.base_rays=self.rays@t[:3,:3].T
        self.origin=t[:3,3]
        self.height_rays=self.base_rays@self.plane[:3,2]
        self.height_origin=float((self.origin-self.plane[:3,3])@self.plane[:3,2])
        self.camera_to_plane=np.linalg.inv(self.plane)@t

    def regions(self, frame, d2c, *, feedback=None, model=None, rgb_grid=False):
        """Colour only classifies measured rays. Unregistered rays are not object evidence."""
        y0,y1,x0,x1=self.crop
        depth=frame.depth_m[y0:y1,x0:x1]
        h=depth*self.height_rays+self.height_origin
        y,x=np.nonzero((depth>0)&np.isfinite(depth)&(h>=self.p.min_height_m)&(h<=self.height))
        z=depth[y,x,None]
        pts=self.base_rays[y,x]*z+self.origin
        flat=(pts-self.plane[:3,3])@self.plane[:3,:3]
        inside=np.all((flat[:,:2]>=self.hull.min(0))&(flat[:,:2]<=self.hull.max(0)),axis=1)
        inside &= (pts[:,1]>=self.config.raw['perception_detector_workspace_y_m'][0])&(pts[:,1]<=self.config.raw['perception_detector_workspace_y_m'][1])
        y,x,pts=y[inside],x[inside],pts[inside]
        uv,positive=project_to_rgb(self.rays[y,x]*depth[y,x,None],frame,d2c)
        uv=np.rint(uv).astype(int)
        rh,rw=frame.rgb.shape[:2]
        vis=positive&(uv[:,0]>=0)&(uv[:,0]<rw)&(uv[:,1]>=0)&(uv[:,1]<rh)
        y,x,pts,uv=y[vis],x[vis],pts[vis],uv[vis]
        green=green_rgb_mask(frame,d2c)
        keep=~green[uv[:,1],uv[:,0]]
        if feedback is not None and model is not None and np.any(keep):
            keep[keep]=~model.classify_robot_points(pts[keep],feedback,camera_origin=self.origin).removed_mask
        y,x,pts,uv=y[keep],x[keep],pts[keep],uv[keep]
        if not len(y):return []
        index=(y+y0)*frame.depth_m.shape[1]+x+x0
        if rgb_grid:
            # Registered stage of V1–V3: contiguous RGB footprint plus the
            # number of independently observed RGB pixels within each footprint.
            mask=np.zeros((rh,rw),np.uint8)
            mask[uv[:,1],uv[:,0]]=1
            footprint=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            footprint[green]=0
            count,labels,stats,_=cv2.connectedComponentsWithStats(footprint,8)
            groups=labels[uv[:,1],uv[:,0]]
        else:
            mask=np.zeros(depth.shape,np.uint8)
            mask[y,x]=1
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
            groups=labels[y,x]
        out=[]
        for label in np.flatnonzero(stats[1:,cv2.CC_STAT_AREA]>=(
                self.p.rgb_area_px if rgb_grid else self.p.min_points))+1:
            selected=groups==label
            cloud=pts[selected]
            if len(cloud)<self.p.min_points:continue
            if rgb_grid:
                seen=len(np.unique(uv[selected,1]*rw+uv[selected,0]))
                if seen<max(self.p.rgb_depth_pixels,self.p.rgb_depth_fraction*stats[label,cv2.CC_STAT_AREA]):continue
            local=(np.median(cloud,axis=0)-self.plane[:3,3])@self.plane[:3,:3]
            if cv2.pointPolygonTest(self.hull,tuple(local[:2]),True)<self.p.edge_m:continue
            if not rgb_grid:
                # Measure this component in RGB using the correspondence already
                # computed above. Filled external contours provide a local area
                # denominator; a hole in one object cannot invalidate its peer.
                pixels = uv[selected]
                offset = pixels.min(0)
                width, height = pixels.max(0)-offset+1
                patch = np.zeros((height+4,width+4),np.uint8)
                xx,yy = (pixels-offset+2).T
                patch[yy,xx] = 1
                valid = np.count_nonzero(patch)
                footprint = cv2.morphologyEx(patch,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
                contours,_ = cv2.findContours(footprint,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(footprint,contours,-1,1,cv2.FILLED)
                area = np.count_nonzero(footprint)
                if area < self.p.rgb_area_px or valid < max(self.p.rgb_depth_pixels,area*self.p.rgb_depth_fraction):
                    continue
            out.append(Region(index[selected],cloud,uv[selected]))
        return out

    def measure(self, region, frame, d2c, *, complete=False, model=None, feedback=None):
        points,index=region.points,region.indices
        if complete:
            mask=np.zeros(frame.depth_m.shape,bool);mask.flat[index]=True
            silhouette,_=selected_rgb_silhouette(frame,d2c,mask)
            if silhouette is not None:
                world=world_points(frame,d2c)
                local=(world-self.plane[:3,3])@self.plane[:3,:3]
                support,_=visible_depth_mask(frame,d2c,silhouette,local,self.surface.plane)
                added=support & ~mask
                if model is not None and added.any():
                    support[added]=~model.classify_robot_points(world[added],feedback,
                        camera_origin=self.origin).removed_mask
                index=np.flatnonzero(mask|support)
                points=world.reshape(-1,3)[index]
        local=(points-self.plane[:3,3])@self.plane[:3,:3]
        steps=plane_pixel_steps(self.camera_to_plane,frame.depth_intrinsics.matrix,np.median(local,axis=0))
        radius=max(.004,3.5*np.max(np.linalg.norm(steps[:,:2],axis=1)))
        d,_=cKDTree(local).query(local,k=5)
        keep=d[:,-1]<=radius
        points,index=points[keep],index[keep]
        if len(index)<self.p.min_points:return None
        try:
            poses,box=obb_grasps(points,self.plane,self.config.canonical_tcp,
                tcp_z_offset_m=self.config.raw.get('grasp_tcp_z_offset_m',0.))
        except VisibleOBBError:
            return None
        extent=tuple(box['extent_m'])
        if min(extent[:2])<self.p.min_width_m:return None
        center=self.plane[:3,:3]@np.array(box['center_world_m'])+self.plane[:3,3]
        box['above_diagnostic_length']=bool(max(extent[:2])>self.p.length_diagnostic_m)
        return Observation(index,points,center,tuple(poses),box,project_base(center[None],frame,d2c)[0],extent)

    def extract(self,frame,d2c,version,*,feedback=None,model=None):
        if version not in (1,2,3,4):raise ValueError('version must be 1..4')
        if not self.surface.matches_frame(frame,d2c):raise ValueError('changed camera calibration or epoch')
        if (feedback is None)!=(model is None):raise ValueError('self-filter requires paired state and model')
        regions=self.regions(frame,d2c,model=model,feedback=feedback,rgb_grid=version<4)
        if version==1 and len(regions)!=1:return []
        if version<4 and regions:
            # Map labelled registered regions to native components by exact
            # same-exposure pixels. No centre distance or overlap cutoff in V2+.
            native=self.regions(frame,d2c,model=model,feedback=feedback)
            seeds=np.zeros(frame.depth_m.size,np.int32)
            for label,r in enumerate(regions,1):seeds[r.indices]=label
            associated=[]
            for r in native:
                votes=np.bincount(seeds[r.indices],minlength=len(regions)+1)[1:]
                best=int(np.argmax(votes))
                if votes[best]==0 or np.count_nonzero(votes==votes[best])!=1:continue
                seed=regions[best]
                if version==1 and (votes[best]<.5*len(seed.indices) or
                      np.linalg.norm(np.median(r.points,0)-np.median(seed.points,0))>.03):continue
                associated.append(r)
            regions=associated
        return [obs for r in regions if (obs:=self.measure(r,frame,d2c,complete=version<=2,
                model=model, feedback=feedback)) is not None]
