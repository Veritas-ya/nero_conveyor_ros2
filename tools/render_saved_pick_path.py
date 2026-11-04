#!/usr/bin/env python3
"""Render an archived pick without opening a camera, CAN, or robot controller."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.spatial.transform import Rotation

from nero_core.two_second.config import CellConfig
from nero_core.two_second.collision_view import restore_frame, project_base
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.smooth_reference import SmoothIntercept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plotly-source', type=Path, default=Path('runs/visualization/graspgenx_pointcloud.html'))
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/'summary.json').read_text())
    journal = [json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
    started = next(row for row in journal if row['event'] == 'run_started')
    run_time = datetime.fromisoformat(started['wall_time_utc']).astimezone(ZoneInfo('Asia/Shanghai')).strftime('%H:%M')
    expected = next((value for key, value in summary['source_hashes'].items()
                     if key.endswith('/config/two_second.hardware.legacy_observed.yaml')), None)
    actual = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if expected is None or expected != actual:
        raise ValueError('配置与事故时保存的哈希不一致，不能标为当时模型；请提供匹配配置')
    records = json.loads((run/'cycle_001/events.json').read_text())
    start = next(r['data'] for r in records if r['event'] == 'planning_submitted')
    plan = next(r['data'] for r in records if r['event'] == 'planning_delivery')
    if start['identity'] != plan['identity']:
        raise ValueError('保存的目标身份不一致')
    detail = plan['trajectory_diagnostics']
    pre, approach = float(detail['prealign_s']), float(detail['approach_s'])
    duration = pre + approach
    origin = float(plan['contact_monotonic_s']) - duration
    st, goal = np.asarray(start['start_tcp_base']), np.asarray(plan['selected_grasp_tcp_base'])
    quat = lambda p: Rotation.from_matrix(p[:3,:3]).as_quat()[[3,0,1,2]]
    ref = SmoothIntercept(start['capture_monotonic_s'], origin, duration, goal[:3,3],
        plan['velocity_base_m_s'], st[:3,3], quat(st), quat(goal),
        float(detail['follow_s']), float(detail['brake_s']), pre)
    predicted = np.asarray(plan['predicted_contact_tcp_base_m'])
    if not np.allclose(ref.contact_xyz, predicted, atol=1e-8, rtol=0):
        raise ValueError('重建参考的接触点与当时保存记录不一致')
    path_times = np.unique(np.r_[np.linspace(0, ref.full_duration_s, 581), pre, duration, ref.follow_end_s])
    path_xyz = np.asarray([ref.sample(t)[0] for t in path_times])
    with np.load(run/'cycle_001/execution_trace.npz', allow_pickle=False) as f:
        send_q = f['dispatched_q_rad'].copy()
        send_times = f['policy_sample_time_s'].copy()
        send_success = f['dispatch_succeeded'].copy()
        feedback_q = f['feedback_q_rad'].copy()
        feedback_times = f['feedback_joint_sample_monotonic_s'].copy()
        feedback_stage = f['feedback_sample_stage'].copy()
    if not send_success.all() or not np.isfinite(send_q).all():
        raise ValueError('轨迹发送记录不完整或包含未成功发送的值')
    valid = ((feedback_stage == 'smooth_intercept_follow_brake') & np.isfinite(feedback_times)
             & np.isfinite(feedback_q).all(axis=1))
    feedback_q, feedback_times = feedback_q[valid], feedback_times[valid]
    if not len(feedback_q):
        raise ValueError('没有有效的执行阶段反馈')
    model = calibrated_model(CellConfig.load(args.config))
    width = float(summary['startup_return_wait']['initial_state']['width_m'])
    def fk(qs):
        out = []
        for q in qs:
            model.set_state(q, width)
            out.append(model.tcp().copy())
        return np.asarray(out)
    send_pose, feedback_pose = fk(send_q), fk(feedback_q)
    model.set_state(feedback_q[-1], width)
    fingers = []
    for g, vertices in model.mesh_vertices.items():
        if 'finger' in model.model.geom(g).name:
            vertices = vertices @ model.data.geom_xmat[g].reshape(3,3).T + model.data.geom_xpos[g]
            fingers.append(vertices[::max(1, len(vertices)//500)])
    fingers = np.concatenate(fingers) if fingers else np.empty((0,3))
    with np.load(run/'preview/target.npz', allow_pickle=False) as data:
        meta = json.loads(str(data['metadata']))
        frame = restore_frame(meta, {'frame_rgb':data['frame_rgb'].copy(),
                                     'frame_depth_m':data['frame_depth_m'].copy()})
        objects = data['points'].copy()
    if (list(meta['identity']) != list(plan['identity']) or
            not np.isclose(meta['capture_s'], start['capture_monotonic_s'], atol=1e-6, rtol=0)):
        raise ValueError('目标画面与轨迹身份不一致')
    d2c = np.asarray(meta['depth_to_color_cv'])
    with np.load(run/'collisions/collision_0001.npz',allow_pickle=False) as data:
        scene = data['scene_points_base_m'].copy()
    collision_meta = json.loads((run/'collisions/collision_0001.json').read_text())
    if (collision_meta['frame_sequence'] != meta['header']['sequence'] or
            collision_meta['target_identity'] != list(plan['identity'])):
        raise ValueError('记录的带面点云与目标曝光不一致')
    # Scene comes from the rejected-candidate packet, but is the shared same-frame
    # belt point set. Do NOT display its candidate collision as the selected path.
    if len(scene) > 14000:
        scene = scene[np.linspace(0, len(scene)-1, 14000, dtype=int)]
    contact_cloud = objects + np.asarray(plan['velocity_base_m_s'])*(plan['contact_monotonic_s']-start['capture_monotonic_s'])
    args.output.mkdir(parents=True,exist_ok=True)
    font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=str(font_path)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
    fig, (ax, timeline) = plt.subplots(2,1,figsize=(14,10), gridspec_kw={'height_ratios':[3.5,1]},constrained_layout=True)
    ax.imshow(frame.rgb)
    def overlay(xyz,color,label,style='-',width=2):
        uv, visible=project_base(xyz,frame,d2c)
        masked=np.where(visible[:,None],uv,np.nan)
        ax.plot(masked[:,0],masked[:,1],style,color=color,lw=width,label=label)
        return uv,visible
    overlay(path_xyz,'#00e5ff','完整 TCP 参考（重建、非全部已执行）','--',2.1)
    overlay(send_pose[:,:3,3],'#ffb31c','已发送关节经模型 FK',width=4)
    overlay(feedback_pose[:,:3,3],'#ff385c','已记录关节反馈经模型 FK',width=3)
    for xyz,color,label in ((path_xyz[0],'#6bf282','开始'),(predicted,'#fefefe','预测接触点')):
        uv,ok=project_base(np.array([xyz]),frame,d2c)
        if ok[0]:
            ax.scatter(*uv[0],s=95,c=color,edgecolor='#111',linewidth=1.1,zorder=9)
            ax.annotate(label,xy=uv[0],xytext=(-60,12) if label=='开始' else (8,-12),textcoords='offset points',color='white',
                bbox={'facecolor':'black','alpha':.7,'edgecolor':'none','pad':3})
    ax.legend(loc='upper left',framealpha=.9,fontsize=9)
    ax.set_title(f"历史曝光帧 #{meta['header']['sequence']} 上的轨迹投影：叠加的是模型与预测，不是碰撞画面")
    ax.axis('off')
    segments=[(0,pre,'预转'),(pre,duration,'接近'),(duration,ref.follow_end_s,'跟随闭合'),(ref.follow_end_s,ref.full_duration_s,'减速')]
    for lo,hi,label in segments:
        timeline.axvspan(lo,hi,alpha=.18,color={'预转':'#9eb4d3','接近':'#00bcd4','跟随闭合':'#e6a74a','减速':'#a875d8'}[label])
        timeline.text((lo+hi)/2,.88,label,ha='center',transform=timeline.get_xaxis_transform())
    for axis,c in zip(range(3),('#1c73c9','#168a51','#9444aa')):
        timeline.plot(path_times,path_xyz[:,axis]*1000,color=c,lw=1.6,label='参考 '+'XYZ'[axis])
        timeline.scatter(send_times,send_pose[:,axis,3]*1000,c=c,s=4,alpha=.65)
    timeline.axvline(send_times[-1],color='#e63842',ls='--',lw=1.7,label='最后成功发送')
    timeline.set(xlabel='轨迹参考时间（秒）',ylabel='TCP 基座坐标（mm）',xlim=(0,ref.full_duration_s))
    timeline.grid(alpha=.2); timeline.legend(loc='lower left',ncol=3,fontsize=8)
    fig.suptitle(f'{run_time} 真机记录：规划参考、已发送段与反馈（均为模型重建）',fontsize=14)
    fig.savefig(args.output/'trajectory_on_rgb.png',dpi=150,bbox_inches='tight');plt.close(fig)
    def trace(xyz,name,color,mode='lines',dash='solid',width=5,size=2,visible=True,extra=None):
        a=np.asarray(xyz)
        result={'type':'scatter3d','mode':mode,'name':name,'x':a[:,0].tolist(),'y':a[:,1].tolist(),'z':a[:,2].tolist(),
            'visible':visible,'hovertemplate':name+'<br>X=%{x:.4f} m<br>Y=%{y:.4f} m<br>Z=%{z:.4f} m<extra></extra>'}
        if 'lines' in mode:result['line']={'color':color,'width':width,'dash':dash}
        if 'markers' in mode:result['marker']={'color':color,'size':size,'opacity':.75}
        if extra:result.update(extra)
        return result
    traces=[trace(scene,'带面点云（同帧，显示抽样）','#97a6a0','markers',size=1.5),
        trace(objects,'曝光时工件点云','#19cbcb','markers',size=2),
        trace(contact_cloud,'预测接触时工件（固定速度外推）','#bc82f5','markers',size=2,visible='legendonly'),
        trace(path_xyz[path_times<=duration],'接近 TCP 参考（重建）','#00a9ea','lines',dash='dash',width=6),
        trace(path_xyz[(path_times>=duration)&(path_times<=ref.follow_end_s)],'跟随 TCP 参考（重建）','#f2a52b','lines',dash='dash',width=6),
        trace(path_xyz[path_times>=ref.follow_end_s],'减速 TCP 参考（重建）','#a35cd2','lines',dash='dash',width=6),
        trace(send_pose[:,:3,3],'已发送关节的 TCP（名义模型）','#ff9b22','lines+markers',width=7,size=2),
        trace(feedback_pose[:,:3,3],'关节反馈的 TCP（名义模型）','#e73b50','lines+markers',width=7,size=2),
        trace(fingers,'末次反馈夹指模型（非实测）','#ef657b','markers',size=2),
        trace(np.array([path_xyz[0]]),'参考起点','#20bd56','markers',size=7),
        trace(np.array([predicted]),'预测接触 TCP（未执行到）','#121827','markers',size=7)]
    # Orientation cannot be inferred from a TCP point; show nominal model axes.
    for label,pose in [('起始',feedback_pose[0]),('末次反馈',feedback_pose[-1])]:
        for axis,col in enumerate(('#e64d3e','#2bbd66','#426de2')):
            end=pose[:3,3]+pose[:3,axis]*.055
            traces.append(trace(np.array([pose[:3,3],end]),label+' TCP '+ 'XYZ'[axis]+' 轴（模型）',col,'lines',width=4,
                                extra={'showlegend':axis==0}))
    source=args.plotly_source.read_text(encoding='utf-8')
    begin=source.find('<script>/**');end=source.find('</script>',begin)
    if begin<0 or end<0 or 'Plotly' not in source[begin:end]:
        raise ValueError('找不到项目现存离线 Plotly 资源')
    plotly_js=source[begin+len('<script>'):end]
    focus=np.vstack((objects, path_xyz, feedback_pose[:,:3,3], fingers))
    lo=focus.min(axis=0)-np.array([.08,.10,.06])
    hi=focus.max(axis=0)+np.array([.08,.10,.06])
    layout={'scene':{'xaxis':{'title':'基座 X (m)','range':[float(lo[0]),float(hi[0])]},
                     'yaxis':{'title':'基座 Y (m)','range':[float(lo[1]),float(hi[1])]},
                     'zaxis':{'title':'基座 Z (m)','range':[float(lo[2]),float(hi[2])]},'aspectmode':'data'},
            'margin':{'l':0,'r':0,'b':0,'t':12},'legend':{'orientation':'h'},'paper_bgcolor':'#f7f9fc'}
    payload=json.dumps({'data':traces,'layout':layout},ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真机轨迹离线复原</title><style>body{margin:0;background:#f7f9fc;color:#172635;font:15px system-ui,sans-serif}header{padding:10px 18px}h2,p{margin:6px 0}#plot{height:76vh}img{display:block;max-width:96%;margin:16px auto} .warn{color:#a12128}</style><header><h2>RUN_TIME 真机抓取：参考轨迹、已发送段与关节反馈</h2><p>拖动旋转、滚轮缩放、点击图例切换；坐标单位为米。点云来自曝光 #FRAME_SEQUENCE，已抽样显示。</p><p><b>虚线</b>：由当时记录重建的 TCP 参考（未保存完整关节轨迹）；<b>橙色</b>：已发送关节经名义模型计算的 TCP；<b>红色</b>：反馈关节经名义模型计算的 TCP。坐标轴及粉色夹指是模型，不是实物测量。</p><p class="warn">计划总长 TOTAL_DURATION 秒；最后成功发送约在参考 LAST_SEND 秒，未记录到计划接触时刻。画面不证明实际碰撞位置，也不验证现场标定。</p></header><div id="plot"></div><img src="trajectory_on_rgb.png" alt="参考和记录轨迹投影至历史RGB曝光及时间图"><script>''' + plotly_js + '''</script><script>const v='''+payload+''';Plotly.newPlot('plot',v.data,v.layout,{responsive:true,displaylogo:false});</script></html>'''
    html=html.replace('RUN_TIME',run_time).replace('FRAME_SEQUENCE',str(meta['header']['sequence']))
    html=html.replace('TOTAL_DURATION',f'{ref.full_duration_s:.2f}').replace('LAST_SEND',f'{send_times[-1]:.2f}')
    (args.output/'trajectory_3d.html').write_text(html,encoding='utf-8')
    report={'source':str(run),'config_sha256':actual,'identity':plan['identity'],'frame_sequence':meta['header']['sequence'],
        'source_note':'参考从记录的平滑策略参数重建；完整关节规划曲线未保存',
        'sensor_note':'关节位置经名义模型 FK；时间为 CAN 主机接收估计；不是真实 TCP 测量',
        'predicted_contact_tcp_base_m':predicted.tolist(),'reconstructed_contact_tcp_base_m':ref.contact_xyz.tolist(),
        'duration_s':ref.full_duration_s,'prealign_s':pre,'last_sent_reference_s':float(send_times[-1]),
        'successful_send_count':len(send_q),'feedback_samples':len(feedback_q),
        'camera_connections':0,'can_connections':0,'commands_sent':0}
    (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    main()
