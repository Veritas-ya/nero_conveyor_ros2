#!/usr/bin/env python3
"""Visualize an actual saved rendered RGB-D frame and GraspGen-X result.

This is an inspection tool, never a planner, collision certificate or controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.offline import plot

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'runs/vision_network_warm_20261027'


def segments(pose):
    # TCP +Y: jaw axis; TCP +Z: approach. Drawing is an 80 mm orientation
    # guide, not a measured jaw width or the exact physical finger meshes.
    p = np.array([[0,-.04,0], [0,-.04,-.04], [0,.04,-.04], [0,.04,0],
                  [0,0,0], [0,0,-.08]], dtype=float)
    return p @ pose[:3,:3].T + pose[:3,3]


def xyza(p):
    return dict(x=p[:,0], y=p[:,1], z=p[:,2])


def build(run, output):
    prefix = 'track_0000_try_01'
    inputs = run / 'inputs' / (prefix + '.npz')
    inference = run / 'inference' / (prefix + '.npz')
    meta_path = run / 'inference' / (prefix + '.json')
    metadata = json.loads(meta_path.read_text())
    data = np.load(inputs, allow_pickle=False)
    pred = np.load(inference, allow_pickle=False)
    points, poses, scores = data['points'], pred['poses'], pred['scores']
    accepted = metadata['accepted_candidates']
    if metadata['input_provenance']['cloud_coordinate_frame'] != 'world':
        raise ValueError('Expected saved world-coordinate cloud')
    if not accepted or poses.shape != (len(scores),4,4) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Missing accepted candidate or incompatible input')
    camera = metadata['input_visuals']['camera']
    depth = np.load(run / 'inputs' / (prefix + '_depth.npy'), allow_pickle=False)
    K = np.asarray(camera['intrinsics']); transform = np.asarray(camera['camera_to_world'])
    # Native depth projection, sparsely sampled only for display.  The target
    # network input below is preserved at full saved resolution.
    yy, xx = np.mgrid[0:depth.shape[0]:5, 0:depth.shape[1]:5]
    z = depth[yy,xx]; valid = np.isfinite(z) & (z > 0)
    x = (xx[valid]-K[0,2])*z[valid]/K[0,0]
    y = (yy[valid]-K[1,2])*z[valid]/K[1,1]
    gl = np.column_stack((x,-y,-z[valid])); context = gl@transform[:3,:3].T+transform[:3,3]
    center = np.median(points, axis=0)
    context = context[(np.abs(context[:,0]-center[0]) < .20) & (np.abs(context[:,1]-center[1]) < .20) & (np.abs(context[:,2]-center[2]) < .16)]
    selected = accepted[0]
    selected_pose = np.asarray(selected['T_world_tcp'])
    canonical_tcp = np.asarray(json.loads((ROOT / 'src/nero_core/experiments/graspgenx_sim/gripper_agx/transforms.json').read_text())['T_canonical_ee_site'])
    raw_tcp = poses @ canonical_tcp
    raw = segments(raw_tcp[int(selected['source_index'])])
    selected_segments = segments(selected_pose)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(**xyza(context),mode='markers',marker=dict(size=1.3,color='#aebbc8',opacity=.35),name='同帧场景（显示抽样）',hoverinfo='skip'))
    fig.add_trace(go.Scatter3d(**xyza(points),mode='markers',marker=dict(size=2.5,color=points[:,2],colorscale='Viridis',colorbar=dict(title='高度 z (m)')),name='网络输入目标点云',hovertemplate='x=%{x:.4f}<br>y=%{y:.4f}<br>z=%{z:.4f} m<extra></extra>'))
    fig.add_trace(go.Scatter3d(**xyza(selected_segments[:4]),mode='lines+markers',line=dict(color='#ff5a25',width=9),marker=dict(size=3),name='通过筛选的抓姿：夹爪示意'))
    fig.add_trace(go.Scatter3d(**xyza(selected_segments[4:]),mode='lines+markers',line=dict(color='#ffb000',width=6),marker=dict(size=3),name='选中抓姿进近方向（朝 TCP）'))
    for axis,color in enumerate(['red','green','blue']):
        p = np.vstack([selected_pose[:3,3], selected_pose[:3,3]+.045*selected_pose[:3,axis]])
        fig.add_trace(go.Scatter3d(**xyza(p),mode='lines',line=dict(color=color,width=5),name='TCP '+ 'XYZ'[axis]))
    fig.add_trace(go.Scatter3d(**xyza(raw[:4]),mode='lines+markers',line=dict(color='#e84eff',width=5),marker=dict(size=2),name='查看原始候选',visible=False))
    fig.add_trace(go.Scatter3d(**xyza(raw[4:]),mode='lines',line=dict(color='#e84eff',width=4),name='原始候选进近方向',visible=False,showlegend=False))
    fig.update_layout(title=f'同帧 RGB-D 点云与 GraspGen-X 抓姿 | t={float(data["capture_timestamp"]):.3f} s | 选中 #{selected["source_index"]}，得分 {selected["score"]:.3f}',
        scene=dict(xaxis_title='世界 X (m)',yaxis_title='世界 Y (m)',zaxis_title='世界 Z (m)',aspectmode='data'),
        margin=dict(l=0,r=0,b=0,t=65),legend=dict(x=0,y=1))
    html = plot(fig, output_type='div', include_plotlyjs=True, auto_open=False)
    # The saved transforms are canonical network poses, transformed using the
    # recorded gripper calibration; acceptance symmetry may change selected pose.
    paths = [segments(p).tolist() for p in raw_tcp]
    embedded = json.dumps({'paths':paths,'scores':scores.tolist(),'selected':int(selected['source_index'])},separators=(',',':'))
    head = '''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>仿真点云与 GraspGen-X 抓姿</title><style>body{font-family:system-ui,sans-serif;margin:0;background:#f7f9fc;color:#18283a}header{padding:12px 18px}p{margin:7px 0}input[type=range]{width:min(480px,70vw)}#plot{height:76vh}</style></head><body><header><h2>仿真点云与 GraspGen-X 抓姿</h2><p>鼠标拖动旋转，滚轮缩放；橙色是筛选通过的抓姿，紫色为选中的原始网络候选。图例可控制显示。</p><p>显示的是同一次保存的曝光帧；场景点云仅为显示而抽样。夹爪线框是 80 mm 的方向示意，不代表预测宽度。筛选通过不等于完整轨迹认证或抓取成功。</p><label><input type="checkbox" id="show"> 显示原始候选</label>　<label>候选编号 <input type="range" id="candidate" min="0" max="511" value="0"></label> <strong id="value"></strong></header><div id="plot">'''
    script = '''</div><script>const D=DATA;const graph=document.querySelector('.plotly-graph-div');const slider=document.getElementById('candidate');const show=document.getElementById('show');slider.max=D.paths.length-1;slider.value=D.selected;function update(){let i=Number(slider.value), p=D.paths[i], on=show.checked;document.getElementById('value').textContent='#'+i+' / '+(D.paths.length-1)+'，网络分数 '+D.scores[i].toFixed(4);Plotly.restyle(graph,{x:[p.slice(0,4).map(v=>v[0]),p.slice(4,6).map(v=>v[0])],y:[p.slice(0,4).map(v=>v[1]),p.slice(4,6).map(v=>v[1])],z:[p.slice(0,4).map(v=>v[2]),p.slice(4,6).map(v=>v[2])],visible:[on,on]},[7,8]);} slider.addEventListener('input',update);show.addEventListener('change',update);update();</script></body></html>'''.replace('DATA',embedded)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(head+html+script,encoding='utf-8')
    # A static preview makes the result readable when an interactive browser
    # or hardware-accelerated WebGL is unavailable.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_static = plt.figure(figsize=(10,7), dpi=160)
    ax = fig_static.add_subplot(111, projection='3d')
    ax.scatter(context[:,0],context[:,1],context[:,2],c='#b4bec8',s=.5,alpha=.18)
    ax.scatter(points[:,0],points[:,1],points[:,2],c=points[:,2],cmap='viridis',s=3)
    ax.plot(*selected_segments[:4].T,color='#e8521c',linewidth=3,label='Selected grasp (jaw guide)')
    ax.plot(*selected_segments[4:].T,color='#ecaa00',linewidth=3,label='Approach toward TCP')
    for axis,color in enumerate(('red','green','blue')):
        p=np.vstack([selected_pose[:3,3],selected_pose[:3,3]+.045*selected_pose[:3,axis]])
        ax.plot(*p.T,color=color,linewidth=2,label='TCP '+'XYZ'[axis])
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',zlabel='World Z (m)',
           title=f'Captured simulation RGB-D and GraspGen-X selected grasp #{selected["source_index"]}')
    ax.set_box_aspect((1,1,.55)); ax.view_init(elev=28,azim=-57)
    ax.legend(loc='upper left',fontsize=8)
    fig_static.text(.1,.015,'Jaw guide is 80 mm, not a predicted jaw width. Grasp selection is not a path certificate.',fontsize=8)
    fig_static.tight_layout(rect=(0,.03,1,1))
    fig_static.savefig(output.with_suffix('.png')); plt.close(fig_static)
    print(f'{output.resolve()} | target points={len(points)}, displayed scene={len(context)}, raw={len(poses)}, accepted={len(accepted)}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DEFAULT)
    parser.add_argument('--output',type=Path,default=ROOT/'runs/visualization/graspgenx_pointcloud.html')
    args=parser.parse_args();build(args.run,args.output)

if __name__ == '__main__': main()
