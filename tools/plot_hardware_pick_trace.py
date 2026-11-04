#!/usr/bin/env python3
"""Plot reconstructed Cartesian reference and FK of saved joint feedback; no devices."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.transform import Rotation
from nero_core.two_second.config import CellConfig
from nero_core.two_second.planning import calibrated_model
from nero_core.two_second.smooth_reference import SmoothIntercept


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();events=[json.loads(line) for line in (a.run/'events.jsonl').read_text().splitlines()]
 records=[e['data']['record'] for e in events if e['event']=='execution_event']
 start=next(e['data'] for e in records if e['event']=='planning_submitted')
 plan=next(e['data'] for e in records if e['event']=='planning_delivery')
 detail=plan['trajectory_diagnostics'];pre=float(detail['prealign_s']);approach=float(detail['approach_s']);duration=pre+approach
 origin=float(plan['contact_monotonic_s'])-duration
 st=np.asarray(start['start_tcp_base']);goal=np.asarray(plan['selected_grasp_tcp_base'])
 quat=lambda pose:Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
 ref=SmoothIntercept(start['capture_monotonic_s'],origin,duration,goal[:3,3],plan['velocity_base_m_s'],st[:3,3],quat(st),quat(goal),detail['follow_s'],detail['brake_s'],pre)
 with np.load(a.run/'cycle_001/execution_trace.npz',allow_pickle=False) as d:
  q=d['feedback_q_rad'].copy();ts=d['feedback_joint_sample_monotonic_s'].copy();dispatch=d['dispatch_start_monotonic_s'].copy();policy=d['policy_sample_time_s'].copy()
 sel=(ts>=dispatch[0]-.03)&np.isfinite(q).all(axis=1)&np.isfinite(ts)
 config=CellConfig.load(a.config);model=calibrated_model(config)
 # Encoder joint positions mapped with nominal model and TCP, not measured TCP.
 feedback=[]
 for row in q[sel]:
  model.set_state(row,config.settings.open_width_m);feedback.append(model.tcp()[:3,3].copy())
 feedback=np.array(feedback); t_feedback=ts[sel]-origin
 tsamp=np.linspace(0.,ref.full_duration_s,581);reference=np.array([ref.sample(t)[0] for t in tsamp])
 font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 if font_path.exists():
  font_manager.fontManager.addfont(str(font_path))
  plt.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=str(font_path)).get_name(),'DejaVu Sans']
 plt.rcParams.update({'axes.unicode_minus':False,'font.size':10})
 fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
 colors=[('#e7f1fc',0,pre,'原位转向'),('#e7f7eb',pre,duration,'接近'),('#fff3dd',duration,ref.follow_end_s,'跟随闭合'),('#f5eafb',ref.follow_end_s,ref.full_duration_s,'减速')]
 for axis in range(3):
  ax=axes.flat[axis]
  for color,lo,hi,label in colors: ax.axvspan(lo,hi,color=color,alpha=.9)
  ax.plot(tsamp,reference[:,axis]*1000,color='#1871b4',lw=2,label='规划 TCP 参考')
  ax.plot(t_feedback,feedback[:,axis]*1000,color='#d94e35',lw=1.8,label='关节反馈经模型 FK')
  ax.axvline(policy[-1],color='#7a3632',linestyle=':',lw=1.2)
  ax.set_title('基座系 '+ 'XYZ'[axis]+'（mm）');ax.set_xlabel('自轨迹起点的时间（s）');ax.grid(alpha=.25)
 ax=axes.flat[3]
 ax.plot(reference[:,0]*1000,reference[:,1]*1000,color='#1871b4',lw=2,label='规划 TCP 参考')
 ax.plot(feedback[:,0]*1000,feedback[:,1]*1000,color='#d94e35',lw=2,label='关节反馈经模型 FK')
 ax.scatter(reference[0,0]*1000,reference[0,1]*1000,c='#1871b4',s=30);ax.scatter(feedback[-1,0]*1000,feedback[-1,1]*1000,c='#d94e35',s=40)
 ax.set_xlabel('基座 X（mm）');ax.set_ylabel('基座 Y（mm）');ax.set_title('俯视参考路径（XY）');ax.axis('equal');ax.grid(alpha=.25)
 axes.flat[0].legend(loc='best',fontsize=9);axes.flat[3].legend(loc='best',fontsize=9)
 fig.suptitle('22:43 实机记录：规划参考与已记录关节反馈（模型重建）',fontsize=14)
 fig.text(.5,-.005,'背景阶段：蓝=转向，绿=接近，黄=跟随闭合，紫=减速。红线结束后的运动仅为计划；未记录实物碰撞传感器。',ha='center',fontsize=9)
 a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160,bbox_inches='tight');plt.close(fig)
 print('输出',a.output,'反馈点数',len(feedback),'最后指令参考时间',round(policy[-1],3),'规划总长',ref.full_duration_s,'未连接设备')

if __name__=='__main__':main()
