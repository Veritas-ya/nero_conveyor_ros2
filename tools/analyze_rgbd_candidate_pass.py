#!/usr/bin/env python3
"""Localize RGB candidates and registered-depth gaps in a recorded camera-only pass."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from compare_recorded_detection import read
from nero_core.two_second.config import CellConfig
from nero_core.two_second.measured_rgbd import fit_plane,green_belt_points
from nero_core.two_second.perception import make_meter
from mujoco_conveyor_sorting.runtime.belt_surface import BeltSurface
from mujoco_conveyor_sorting.runtime.depth_calibration import load_runtime_depth_calibration
from vendor.nero7_speed_reference.nero7_perception.detection import color_masks
from vendor.nero7_speed_reference.nero7_perception.geometry import roi_mask


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path);p.add_argument('--run',required=True,type=Path);p.add_argument('--stride',type=int,default=1);p.add_argument('--overlay-stride',type=int,default=10);a=p.parse_args()
 files=sorted((a.run/'rgbd_frames').glob('*.npz'));assert files
 c=CellConfig.load(a.config);cal=load_runtime_depth_calibration(c.raw['camera_calibration_file']);first,d2c=read(files[0],cal)
 surface=BeltSurface.from_frame(first,d2c,fit_plane=fit_plane,green_belt_points=green_belt_points);m=make_meter(c,surface);m._geometry(first,d2c)
 out=a.run/'candidates';out.mkdir(exist_ok=True);report=[];start=time.monotonic()
 for index,path in enumerate(files[::a.stride]):
  frame,d2c=read(path,cal);m._geometry(frame,d2c);depth=frame.depth_m;aligned=m._registered_depth(depth,d2c,m._native_selection(depth,d2c))
  u0,v0,u1,v1=m.roi_px;x0,y0=max(0,u0-2)&~1,max(0,v0-2)&~1;x1,y1=min(aligned.shape[1],u1+2),min(aligned.shape[0],v1+2)
  crop=np.s_[y0:y1,x0:x1];color=cv2.cvtColor(frame.rgb[crop],cv2.COLOR_RGB2BGR)
  if m.tracking_rgb_roi is not None:color[~m.tracking_rgb_roi[crop]]=0
  _,gray=color_masks(color,m.parameters);selected=roi_mask(gray.shape,[u0-x0,v0-y0,u1-x0,v1-y0]);mask=cv2.morphologyEx((gray&selected).astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
  count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8);valid=np.isfinite(aligned[crop])&(aligned[crop]>0);candidates=[]
  for label in range(1,count):
   area=int(stats[label,cv2.CC_STAT_AREA]);
   if area<m.parameters['min_area_px']:continue
   x,y,w,h=map(int,stats[label,:4]); region=(labels[y:y+h,x:x+w]==label).astype(np.uint8);border=m.parameters['depth_erode_px']
   if border:region=cv2.erode(region,np.ones((2*border+1,2*border+1),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
   region=region.astype(bool);inside=int(region.sum());num=int((region&valid[y:y+h,x:x+w]).sum());bad=num<m.parameters['depth_min_valid_pixels'] or num<m.parameters['depth_min_valid_ratio']*inside
   candidates.append({'bbox_rgb_px':[x+x0,y+y0,w,h],'component_pixels':area,'interior_pixels':inside,'valid_depth_pixels':num,'valid_depth_fraction':None if not inside else num/inside,'minimum_valid_pixels':m.parameters['depth_min_valid_pixels'],'minimum_valid_fraction':m.parameters['depth_min_valid_ratio'],'would_reject_frame':bool(bad)})
  row={'sequence':frame.header.sequence,'source_time':frame.header.source_time,'roi_rgb_px':m.roi_px,'candidates':candidates,'rejected':any(x['would_reject_frame'] for x in candidates)};report.append(row)
  if index%a.overlay_stride==0:
   image=cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR);cv2.rectangle(image,(u0,v0),(u1,v1),(255,255,0),2)
   for item in candidates:
    x,y,w,h=item['bbox_rgb_px'];red=item['would_reject_frame'];clr=(0,0,255) if red else (0,220,0);cv2.rectangle(image,(x,y),(x+w,y+h),clr,2);f=item['valid_depth_fraction'];cv2.putText(image,f'{0 if f is None else f:.0%}',(x,max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,clr,1,cv2.LINE_AA)
   cv2.imwrite(str(out/f'frame_{frame.header.sequence:09d}.jpg'),image,[cv2.IMWRITE_JPEG_QUALITY,80])
  if index%100==0:print('analyzed',index,flush=True)
 with (out/'candidates.jsonl').open('w') as f:
  for row in report:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 summary={'frames_analyzed':len(report),'frames_with_bad_candidate':sum(x['rejected'] for x in report),'total_bad_candidates':sum(sum(c['would_reject_frame'] for c in x['candidates']) for x in report),'overlay_stride':a.overlay_stride,'roi_rgb_px':m.roi_px,'wall_s':time.monotonic()-start,'camera_connections':0,'robot_commands':0,'note':'Reproduces RGB component and registered-depth checks; no claim candidate identity or all later detection filters.'}
 (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
