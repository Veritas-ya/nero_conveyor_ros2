#!/usr/bin/env python3
"""Draw logged scene-collision state and margin with the real private model; no devices."""
import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
# Use the mplot3d shipped beside this Matplotlib when pip and system namespace
# packages coexist. Do not modify the user's installed Python environment.
import mpl_toolkits
matching_toolkit=Path(matplotlib.__file__).parent.parent/'mpl_toolkits'
if (matching_toolkit/'mplot3d').is_dir():
    mpl_toolkits.__path__=[str(matching_toolkit),*[p for p in mpl_toolkits.__path__ if p!=str(matching_toolkit)]]
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from scipy.spatial import ConvexHull,HalfspaceIntersection

from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model


def equal_limits(ax, vertices, factor=1.08):
    lo,hi=vertices.min(axis=0),vertices.max(axis=0)
    center=(lo+hi)/2;radius=max(hi-lo)*factor/2
    for axis,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):axis(c-radius,c+radius)
    ax.set_box_aspect((1,1,1))


def mesh(ax, vertices, color, alpha, *, edges=False, linewidth=.3):
    hull=ConvexHull(vertices)
    collection=Poly3DCollection(vertices[hull.simplices],facecolor=color,alpha=alpha,
        edgecolor=color if edges else 'none',linewidth=linewidth)
    ax.add_collection3d(collection)
    return hull


def describe(run, candidate, config):
    source=Path(run)/'summary.json';summary=json.loads(source.read_text())
    attempt=next(a for a in summary['details']['failures'] if a.get('candidate_index')==candidate-1
        and (a.get('details') or {}).get('kind')=='scene_collision')
    detail=attempt['details'];point=np.asarray(detail['points_base_m'][0],float)
    if detail['geometry']!='nero_link6_collision':
        raise ValueError('本示意图专用于nero_link6_collision；其他部位请使用通用碰撞预览')
    model=calibrated_model(config);model.set_state(detail['joints_rad'],detail['width_m'])
    geom=int(model.model.geom(detail['geometry']).id)
    rotation=np.asarray(model.data.geom_xmat[geom],float).reshape(3,3)
    translation=np.asarray(model.data.geom_xpos[geom],float)
    vertices=np.asarray(model.mesh_vertices[geom],float)
    equations=np.asarray(model.meshes[geom][0],float)
    local=(point-translation)@rotation
    signed=equations[:,:3]@local+equations[:,3]
    margin=float(detail['margin_m'])
    solved=minimize(lambda x:.5*np.sum((x-local)**2),vertices.mean(axis=0),jac=lambda x:x-local,
        constraints=[LinearConstraint(equations[:,:3],-np.inf,-equations[:,3])],method='SLSQP',
        options={'ftol':1e-15,'maxiter':500})
    if not solved.success or np.max(equations[:,:3]@solved.x+equations[:,3])>1e-10:
        raise ValueError('最近凸包点未收敛，不生成未验证的距离标签')
    padded=equations.copy();padded[:,3]-=margin
    expanded=HalfspaceIntersection(padded,vertices.mean(axis=0)).intersections
    original_world=vertices@rotation.T+translation
    expanded_world=expanded@rotation.T+translation
    nearest=solved.x@rotation.T+translation
    geometry=[]
    for g in model.meshes:
        world=np.asarray(model.mesh_vertices[g],float)@model.data.geom_xmat[g].reshape(3,3).T+model.data.geom_xpos[g]
        geometry.append((int(g),world))
    match=re.search(r'(\d{8})_(\d{6})',Path(run).name)
    run_time=Path(run).name if match is None else datetime.strptime('_'.join(match.groups()),'%Y%m%d_%H%M%S').strftime('%Y-%m-%d %H:%M:%S')
    inside=bool(np.all(signed<=0));within=bool(np.all(signed<=margin))
    interpretation=('点在第六连杆本体凸包内' if inside else '点未进入第六连杆本体凸包，但进入软件余量' if within else '当前模型未复现该记录的余量触发')
    report={'source_summary':str(source.resolve()),'source_sha256':sha256(source.read_bytes()).hexdigest(),
        'run_time_local':run_time,
        'config_sha256':config.source_sha256,'candidate_number':candidate,
        'sample_time_s':attempt['last_progress']['trajectory_time_s'],'geometry':detail['geometry'],
        'point_base_m':point.tolist(),'nearest_hull_point_base_m':nearest.tolist(),
        'nominal_convex_hull_euclidean_distance_m':float(np.linalg.norm(solved.x-local)),
        'maximum_signed_halfspace_value_m':float(signed.max()),'software_margin_m':margin,
        'inside_nominal_hull':inside, 'inside_software_margin':within,
        'margin_boundary_slack_m':float(margin-signed.max()),'joints_rad':detail['joints_rad'],
        'prediction_not_measured_collision':True,'full_scene_saved':False,'raw_rgb_saved':False,
        'hardware_connections':0,'commands_sent':0,
        'interpretation':interpretation+'；不证明现场无障碍或误检。'}
    return report,geometry,geom,original_world,expanded_world,point,nearest,rotation@equations[int(np.argmax(signed)),:3]


def draw(report, geometry, geom, original, expanded, point, nearest, normal, output):
    font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font.exists():
        fontManager.addfont(str(font));family=FontProperties(fname=font).get_name()
    else:family='DejaVu Sans'
    plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,'font.size':11,
        'svg.fonttype':'none'})
    blue='#2878b5';orange='#f0a23a';red='#d93d47';gray='#95a5b4';ink='#203142'
    fig=plt.figure(figsize=(16,9),facecolor='#f7f9fc')
    gs=fig.add_gridspec(2,2,left=.035,right=.965,bottom=.24,top=.86,width_ratios=[1.08,1],
        height_ratios=[3.0,1.0],hspace=.30,wspace=.10)
    full=fig.add_subplot(gs[:,0],projection='3d',computed_zorder=False)
    zoom=fig.add_subplot(gs[0,1],projection='3d',computed_zorder=False)
    band=fig.add_subplot(gs[1,1])
    for ax in (full,zoom):
        ax.set_facecolor('#f7f9fc');ax.set_proj_type('ortho')
        ax.tick_params(labelsize=8,pad=1)
        ax.set_xlabel('X / mm',labelpad=4);ax.set_ylabel('Y / mm',labelpad=4);ax.set_zlabel('Z / mm',labelpad=4)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.set_facecolor((.95,.97,.99,1));axis.pane.set_edgecolor('#dae3ed')
    for g,world in geometry:
        mesh(full,world*1000,blue if g==geom else gray,.86 if g==geom else .16)
    mesh(full,expanded*1000,orange,.12,edges=True,linewidth=.18)
    full.scatter(*point*1000,c=red,s=70,depthshade=False,zorder=50,edgecolor='white',linewidth=1)
    full.text(*(point*1000+np.array([15,30,25])),'告警触发点',color=red,fontsize=12,zorder=60)
    equal_limits(full,np.concatenate([world for _,world in geometry])*1000,1.05)
    full.view_init(elev=20,azim=-48)
    full.set_title('① 被拒绝的规划姿态（全臂）',fontsize=14,color=ink,pad=5)
    mesh(zoom,original*1000,blue,.58,edges=True,linewidth=.12)
    mesh(zoom,expanded*1000,orange,.12,edges=True,linewidth=.3)
    zoom.plot(*np.stack([nearest,point]).T*1000,color=red,linewidth=2,zorder=40)
    zoom.scatter(*nearest*1000,c=blue,s=25,depthshade=False,zorder=45)
    zoom.scatter(*point*1000,c=red,s=75,depthshade=False,zorder=50,edgecolor='white',linewidth=1)
    equal_limits(zoom,np.concatenate([expanded,point[None,:]])*1000,1.12)
    azim=np.degrees(np.arctan2(normal[1],normal[0]))+48
    elev=float(np.clip(np.degrees(np.arctan2(normal[2],np.hypot(normal[0],normal[1])))+20,-45,65))
    zoom.view_init(elev=elev,azim=azim)
    zoom.set_title('② 第六连杆局部：本体与软件余量',fontsize=14,color=ink,pad=5)
    distance=report['maximum_signed_halfspace_value_m']*1000;margin=report['software_margin_m']*1000
    band.axvspan(-3.,0,color=blue,alpha=.65);band.axvspan(0,margin,color=orange,alpha=.22)
    band.axvline(0,color=blue,linewidth=2);band.axvline(margin,color=orange,linewidth=2)
    band.scatter([distance],[.5],s=100,color=red,zorder=10,edgecolor='white')
    band.annotate(f'触发点：{distance:.3f} mm',xy=(distance,.5),xytext=(3.4,1.15),color=red,
        fontsize=11,arrowprops={'arrowstyle':'->','color':red},ha='center')
    band.text(-1.5,.5,'本体',ha='center',va='center',color='white',fontsize=12)
    band.text(3.4,.16,f'{margin:g} mm 软件余量',ha='center',color='#a36612',fontsize=11)
    band.text(margin+.15,.58,f'{margin:.3f} mm\n余量边界',color='#a36612',fontsize=10)
    band.set_xlim(-3,margin+2);band.set_ylim(-.15,1.45);band.set_yticks([])
    band.set_xticks([0,2,4,6,8]);band.set_xlabel('沿触发凸包面法向的外侧距离 / mm',labelpad=5)
    for name in ['top','right','left']:band.spines[name].set_visible(False)
    band.set_title(f'③ 进入余量约 {report["margin_boundary_slack_m"]*1000:.3f} mm',
        fontsize=12,color=ink,loc='left',pad=10)
    fig.suptitle('第六连杆碰撞告警：保存状态的几何复原',fontsize=23,fontweight='bold',color=ink,y=.972)
    fig.text(.045,.91,f'{report["run_time_local"]}  ·  候选 {report["candidate_number"]}  ·  轨迹 t = {report["sample_time_s"]:.3f} s  ·  规划预测，非实拍撞击',
        fontsize=12,color='#586c80')
    fig.legend(handles=[Line2D([0],[0],color=gray,lw=6,label='其余机器人模型'),
        Line2D([0],[0],color=blue,lw=6,label='第六连杆本体凸包'),
        Line2D([0],[0],color=orange,lw=3,label='8 mm 余量包络'),
        Line2D([0],[0],marker='o',color='none',markerfacecolor=red,markersize=9,label='日志中的场景触发点')],
        loc='lower center',bbox_to_anchor=(.5,.13),ncol=4,frameon=False)
    xyz=', '.join(f'{v*1000:.3f}' for v in report['point_base_m'])
    fig.text(.045,.085,f'触发点基座坐标 XYZ = ({xyz}) mm；到第六连杆本体凸包的最近距离约 {report["nominal_convex_hull_euclidean_distance_m"]*1000:.3f} mm。',fontsize=11,color=ink)
    fig.text(.045,.051,'结论仅针对保存点与模型：'+report['interpretation'],fontsize=11,color=ink)
    fig.text(.045,.020,'原始RGB和全量场景未保存，因此没有绘制现场背景；未连接相机、机械臂或CAN。',fontsize=10,color='#687c90')
    fig.savefig(output/'scene_collision.png',dpi=170,facecolor=fig.get_facecolor())
    fig.savefig(output/'scene_collision.svg',facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--candidate',type=int,default=1);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    config=CellConfig.load(args.config)
    args.output.mkdir(parents=True,exist_ok=False)
    result=describe(args.run,args.candidate,config)
    draw(*result,args.output)
    (args.output/'geometry_report.json').write_text(json.dumps(result[0],ensure_ascii=False,indent=2))
    print(f'[示意图] {args.output.resolve()}/scene_collision.png',flush=True)


if __name__=='__main__':main()
