"""Project the measured belt upstream of entry, with optional object height.

This is the strategy's observation ROI, independent of the arm workspace.
It never changes native depth or fills an unknown RGB-D sample.
"""
import cv2
import numpy as np


def upstream_rgb_roi(surface, camera, color_to_base, direction, entry, height):
    if not np.isfinite(height) or height<0:
        raise ValueError('搜索投影高度必须为非负有限米数')
    direction=np.asarray(direction,dtype=float).reshape(-1)
    if len(direction)==2:direction=np.r_[direction,0.]
    if direction.shape!=(3,) or not np.isfinite(direction).all():
        raise ValueError('上游检测范围需要明确的传送带方向')
    direction=direction/np.linalg.norm(direction)
    transform=surface.plane_to_base
    polygon=np.asarray(surface.hull_xy,dtype=float).reshape(-1,2)
    normal=transform[:3,:2].T@direction
    limit=float(entry-transform[:3,3]@direction+max(0.,-direction[2]*height))
    clipped=[]
    for first,second in zip(polygon,np.roll(polygon,-1,axis=0)):
        a,b=first@normal-limit,second@normal-limit
        if a<=0:clipped.append(first)
        if (a<=0)!=(b<=0):clipped.append(first+(second-first)*a/(a-b))
    mask=np.zeros((camera.height,camera.width),np.uint8)
    if len(clipped)<3:return mask.astype(bool)
    polygon=np.asarray(clipped)
    # Sample curved distorted edges before rasterization. One-pixel expansion
    # covers integer rounding; it is not a grasp or association tolerance.
    t=np.linspace(0.,1.,33)
    edges=(polygon[:,None,:]*(1-t[None,:,None])+
           np.roll(polygon,-1,axis=0)[:,None,:]*t[None,:,None]).reshape(-1,2)
    base=np.c_[edges,np.zeros(len(edges))]@transform[:3,:3].T+transform[:3,3]
    volume=np.concatenate((base,base+[0.,0.,height]))
    optical=(volume-color_to_base[:3,3])@color_to_base[:3,:3]
    if np.any(optical[:,2]<=.05):
        if height==0:
            raise ValueError('实测带面搜索轮廓穿过相机近平面')
        # A near-plane crossing cannot be bounded safely by projected corners.
        return np.ones(mask.shape,bool)
    pixels=camera.project(optical)
    if not np.isfinite(pixels).all() or np.max(np.abs(pixels))>1e6:
        if height==0:
            raise ValueError('实测带面搜索轮廓无法有效投影')
        return np.ones(mask.shape,bool)
    cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(pixels).astype(np.int32)),1)
    return cv2.dilate(mask,np.ones((3,3),np.uint8)).astype(bool)
