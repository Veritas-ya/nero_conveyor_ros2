import numpy as np
import cv2
from experiments.graspgenx_sim.visual_velocity_ablation import cloud_translation,RGBOpticalFlow,ray_intersections


def test_cloud_translation_without_speed_prior():
    p=np.random.default_rng(2).uniform(-.04,.04,(500,3));d=np.array([.003,-.001,.0005]);v,info=cloud_translation(p,p+d,.05)
    assert info['reason']=='ok'
    np.testing.assert_allclose(v,d/.05,atol=1e-6)


def test_rgb_flow_uses_initialized_features_and_belt_projection():
    im=np.zeros((120,160,3),np.uint8)
    for x in (35,55,75,95,115):
        for y in (30,50,70,90):cv2.rectangle(im,(x,y),(x+5,y+5),(255,255,255),-1)
    mask=np.zeros((120,160),np.uint8);mask[20:110,20:140]=1
    K=np.array([[100.,0,80],[0,100.,60],[0,0,1]])
    T=np.diag([1.,-1.,-1.,1.]);T[2,3]=1.
    v=RGBOpticalFlow(); assert v.update(im,mask,1.,K,T,0.,3)[0] is None
    moved=np.roll(im,2,axis=1);empty=np.zeros_like(mask)
    out,info=v.update(moved,empty,1.1,K,T,0.,3)
    assert info['reason']=='ok' and info['features']>=4
    np.testing.assert_allclose(out[0],.2,atol=.01)


def test_fixture_initialization_uses_rgb_not_depth():
    from experiments.graspgenx_sim.visual_velocity_ablation import rgb_orange_fixture_mask
    image=np.zeros((80,80,3),np.uint8);image[20:45,25:50]=[200,80,20]
    mask=rgb_orange_fixture_mask(image)
    assert mask[30,30] and not mask[0,0] and mask.sum()==625
