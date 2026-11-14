"""Validate measured depth components before the existing tracker assigns IDs.

The RGB grid contains the already registered native depth. Its measured points
are transformed to the existing BeltSurface frame and passed to the original
DepthGeometryVisionPipeline; tracker measurements and Kalman settings stay intact.
"""
import cv2
import numpy as np
from ..perception import DepthGeometryVisionPipeline, GeometryPrototype
from vendor.nero7_speed_reference.nero7_perception.detection import color_masks


class MeasuredWorkpieceAdmission:
    def __init__(self, surface, maximum_height_m, *, include_speed_position=False,
                 maximum_target_length_m=.16):
        self.surface=surface
        self.include_speed_position=bool(include_speed_position)
        bounds=surface.bounds_xy_m
        # Same geometric limits as the deployed single_pick.detect_target.
        self.pipeline=DepthGeometryVisionPipeline(
            [GeometryPrototype(category_id=0,dimensions=(.05,.05,.05),symmetric=True)],
            support_plane_height=0.,roi_x=bounds[0],roi_y=bounds[1],
            minimum_height=.012,maximum_height=maximum_height_m,minimum_mask_pixels=70,
            minimum_footprint=.012,maximum_footprint=maximum_target_length_m,classification_enabled=False)

    def filter(self, detections, mask, depth, camera, color_to_base, frame, parameters,
               *, robot_model=None, robot_state=None):
        # CC labels come from this exposure's existing detector mask, never a
        # delayed geometry worker or the mask from an earlier exposure.
        count,labels,stats,centers=cv2.connectedComponentsWithStats(np.uint8(mask>0),8)
        plane=self.surface.plane_to_base
        accepted=[];decisions=[]
        for detection in detections:
            row={'rgb_uv':[float(detection['u']),float(detection['v'])],
                 'position_base_m':np.asarray(detection['centroid_base']).tolist(),
                 'accepted':False,'id_assigned':False}
            matches=np.flatnonzero(np.all(np.isclose(centers[1:],row['rgb_uv'],rtol=0,atol=1e-6),axis=1))+1
            if len(matches)!=1:
                row['reason']='当前曝光的深度连通区域无法唯一关联';decisions.append(row);continue
            label=int(matches[0]);x,y,w,h=map(int,stats[label,:4])
            component=labels[y:y+h,x:x+w]==label
            z=np.asarray(depth[y:y+h,x:x+w]);valid=component & np.isfinite(z) & (z>0)
            green,_=color_masks(cv2.cvtColor(frame.rgb[y:y+h,x:x+w],cv2.COLOR_RGB2BGR),parameters)
            valid &= ~green
            rr,cc=np.nonzero(valid)
            if len(rr)<self.pipeline.minimum_mask_pixels:
                row.update(reason='有效非带面深度点不足',valid_depth_points=len(rr));decisions.append(row);continue
            optical=camera.rays(cc+x,rr+y)*z[rr,cc,None]
            base=optical@color_to_base[:3,:3].T+color_to_base[:3,3]
            local=(base-plane[:3,3])@plane[:3,:3]
            row['height_range_m']=np.quantile(local[:,2],[.01,.99]).tolist()
            keep=(local[:,2]>=self.pipeline.minimum_height)&(local[:,2]<=self.pipeline.maximum_height)
            row['above_belt_depth_points']=int(keep.sum())
            if not keep.any():
                row['reason']='实测高度不在带面上方12mm至配置上限内';decisions.append(row);continue
            if robot_model is not None and robot_state is not None:
                self_result=robot_model.classify_robot_points(base[keep],robot_state,
                    camera_origin=color_to_base[:3,3])
                ids=np.flatnonzero(keep);keep[ids[self_result.removed_mask]]=False
                row['robot_points_removed']=int(np.count_nonzero(self_result.removed_mask))
            points=np.full((*z.shape,3),np.nan)
            points[rr,cc]=local
            allowed=np.zeros(z.shape,bool);allowed[rr[keep],cc[keep]]=True
            intrinsics=camera.K.copy();intrinsics[:2,2]-=[x,y]
            parts=self.pipeline.extract_components(None,z,intrinsics,np.eye(4),
                camera_convention='opencv',foreground_mask=allowed,
                points_world=points,roi_selects_components=True)
            parts=[part for part in parts if cv2.pointPolygonTest(self.surface.hull_xy,
                tuple(map(float,part.position[:2])),True)>.025]
            if len(parts)!=1:
                row.update(reason='未通过原有带面边界、连通点数或物理尺寸检查',
                           component_size_diagnostics=self.pipeline.last_component_workload,
                           depth_workpiece_components=len(parts));decisions.append(row);continue
            row.update(accepted=True,reason='实测带面高度与原有工件几何检查通过',
                valid_depth_points=int(parts[0].mask.sum()),height_m=float(parts[0].height)
                if hasattr(parts[0],'height') else float(np.quantile(local[keep,2],.98)))
            row.pop('id_assigned',None)  # the tracker owns any subsequent ID
            # Retain the exact original observation for the original tracker.
            if self.include_speed_position:
                # Reuse this exposure's already filtered, uniquely associated
                # component. Transform the actual registered RGB-D points back
                # to base before taking per-axis percentiles, matching the
                # simulation's 5--95% visible-cloud speed observation.
                component_points=np.asarray(parts[0].mask,dtype=bool)[rr,cc] & keep
                selected_base=base[component_points]
                selected_base=selected_base[np.isfinite(selected_base).all(axis=1)]
                if not len(selected_base):
                    row.update(accepted=False,reason='测速点云为空');decisions.append(row);continue
                lo,hi=np.percentile(selected_base,[5.,95.],axis=0)
                detection={**detection,'speed_position_base':(lo+hi)/2.,
                           'speed_position_point_count':len(selected_base)}
                row['speed_position_base_m']=detection['speed_position_base'].tolist()
            accepted.append(detection);decisions.append(row)
        return accepted,{'source':'measured_belt_depth_before_ID',
            'raw_candidates':len(detections),'accepted_candidates':len(accepted),
            'rejected_candidates':len(detections)-len(accepted),'decisions':decisions,
            'minimum_height_m':self.pipeline.minimum_height,
            'maximum_height_m':self.pipeline.maximum_height,'depth_modified':False}
