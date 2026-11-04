"""Independent two-second OBB/RGB-D controller; MuJoCo only, no hardware."""
import argparse,json,os,time,traceback
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import yaml
from experiments.graspgenx_sim.run import config_overrides,serializable
from experiments.graspgenx_sim.nero7_perception import Nero7FixedSimulation,Nero7SensorControlEnvironment
from experiments.graspgenx_sim.sensor_control import SensorControlConfig
from experiments.graspgenx_sim.camera import cloud_for_track
from experiments.graspgenx_sim.obb_grasp import obb_top_down_grasp
from experiments.graspgenx_sim.live_cloud_view import draw_live_cloud
from experiments.graspgenx_sim.trajectory_overlay import draw_tcp_path
from experiments.two_second_grasp.controller import (Intercept, predict_entry_time,
    EntryNotReady, APPROACH_DURATION_S, DEFAULT_FREEZE_UPSTREAM_DISTANCE_M)
from experiments.two_second_grasp.placement import plan_leg, plan_joint_route, plan_return, evaluate_placement
from experiments.two_second_grasp.smooth_reference import SmoothDiagnosticReference
from experiments.two_second_grasp.smooth_execution import plan_smooth, closure_gate, check_loaded_brake, plan_smooth_leg, plan_transfer_candidates
from experiments.two_second_grasp.simulation_report import summarize, summarize_retention
from experiments.two_second_grasp.random_trials import make_trials, spawn_trial, bounds
from nero_core.two_second.feedback_gates import GRIP_CONFIRMATION_DWELL_S
from nero_core.two_second.motion_profile import select_motion_profile
from nero_core.two_second.smooth_reference import SmoothIntercept
from mujoco_conveyor_sorting.controllers.ik_solver import DampedLeastSquaresIK
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


class IndependentControllerSimulation(Nero7FixedSimulation):
    def get_predicted_grasp_pose(self, object_id, observation=None):
        # The legacy lifecycle asks for this optional historical metric after
        # contact. This controller owns a frozen prediction independently; it
        # must not query the unrelated legacy predictor or synthesize velocity.
        return None


def plan(env, ref, q, qpos, dt, width, cancelled):
    """Use private MuJoCo data; validate sampled robot/static geometry only."""
    data=mujoco.MjData(env.model);data.qpos[:]=qpos
    ik=DampedLeastSquaresIK(env.model,data,[f'joint{i}' for i in range(1,8)],'ee_site',env.ik.config)
    lower=np.asarray(env.robot_controller.command_joint_lower_limits);upper=np.asarray(env.robot_controller.command_joint_upper_limits)
    ik.lower_limits=lower.copy();ik.upper_limits=upper.copy()
    samples=[];seed=q.copy();stamps=np.arange(0.,ref.duration_s+.60001,dt)
    start=time.perf_counter()
    for t in stamps:
        if cancelled():return {'error':'planning_cancelled'}
        xyz,quat=ref.sample(t);sol=ik.solve(xyz,quat,qpos_seed=seed,cancelled=cancelled)
        if not sol.success:return {'error':'ik_failed','time_s':float(t),'position_error_m':sol.position_error,'rotation_error_rad':sol.orientation_error,'planning_wall_s':time.perf_counter()-start}
        nextq=np.asarray(sol.joint_positions).copy()
        n=max(1,int(np.ceil(np.max(np.abs(nextq-seed))/.008)))
        for u in np.linspace(0.,1.,n+1):
            qq=seed+(nextq-seed)*u;data.qpos[:]=qpos;data.qpos[env.arm_qpos_adrs]=qq
            mujoco.mj_forward(env.model,data)
            if not env._arm_state_data_is_safe(qq,data,approach_axis_local=np.array([0.,0.,1.]),minimum_downward_alignment=0.,minimum_tcp_height=env.belt_height,minimum_joint_margin=env.robot_controller.config.joint_command_margin,maximum_jacobian_condition=None):
                return {'error':'static_scene_or_joint_collision','time_s':float(t),'planning_wall_s':time.perf_counter()-start}
        samples.append(nextq);seed=nextq
    joints=np.asarray(samples)
    speeds=np.abs(np.diff(joints,axis=0)/dt)
    limits=np.asarray(env.robot_controller.max_joint_velocity)
    if np.any(speeds>limits+1e-6):return {'error':'joint_velocity_exceeded','peak_rad_s':float(np.max(speeds)),'planning_wall_s':time.perf_counter()-start}
    return {'samples':joints,'stamps':stamps,'planning_wall_s':time.perf_counter()-start,'max_joint_speed_rad_s':float(np.max(speeds))}


def plan_alternatives(env, references, q, qpos, dt, width, cancelled, *, conditions=None, delivery_budget_s=None, target_cloud_world=None):
    from mujoco_conveyor_sorting.runtime.single_pick import PickRejected
    attempts=[]
    begun=time.perf_counter()
    moving_target=None
    if conditions is not None and target_cloud_world is not None:
        from experiments.two_second_grasp.moving_target_check import SimulationMovingTargetCheck
        moving_target=SimulationMovingTargetCheck(conditions.mirror,target_cloud_world,env.nero_world_to_base,width)
    total_deadline=(float('inf') if conditions is None else begun+conditions.config.raw['planning_budget_s'])
    first_start=references[0].start_s if references else None
    for index, ref in enumerate(references):
        send_deadline=(float('inf') if conditions is None else
            begun+delivery_budget_s+(ref.start_s-first_start))
        candidate_deadline=(float('inf') if conditions is None else min(total_deadline,
            send_deadline,time.perf_counter()+conditions.config.raw['candidate_planning_budget_s']))
        if time.perf_counter()>=candidate_deadline:
            attempts.append({'candidate':index,'duration_s':ref.duration_s,
                             'error':'candidate_send_deadline_expired'})
            continue
        abort=lambda:cancelled() or time.perf_counter()>=candidate_deadline
        planner=plan_smooth if isinstance(ref,SmoothIntercept) else plan
        first_hz=None if conditions is None else conditions.config.raw['planning_knot_hz']
        fallback_hz=first_hz if conditions is None else conditions.config.raw.get('planning_knot_hz_fallback',first_hz)
        rates=[first_hz] if first_hz==fallback_hz else [first_hz,fallback_hz]
        for knot_hz in rates:
            try:
                result=planner(env,ref,q,qpos,dt,width,abort,
                    **({'pose_solver':conditions,'moving_target':moving_target,'knot_hz_override':knot_hz} if conditions is not None else {}))
            except PickRejected as error:
                result={'error':str(error),'details':error.details}
            attempts.append({'candidate':index,'duration_s':ref.duration_s,
                'join_s':(ref.prealign_s-ref.rotate_first_s if hasattr(ref,'rotate_first_s') else None),
                'ik_knot_hz':knot_hz,
                'candidate_budget_expired':time.perf_counter()>=candidate_deadline,
                **{k:v for k,v in result.items() if k not in ('samples','stamps','path')}})
            if 'error' not in result and not abort():
                result['reference']=ref
                result['attempts']=attempts
                result['all_candidates_wall_s']=time.perf_counter()-begun
                return result
            if result.get('error') not in ('deployed_ik_failed','smooth_joint_curve_rejected','sampled_fk_reference_mismatch') or abort():break
        if cancelled() or time.perf_counter()>=total_deadline:break
    return {'error':'all_obb_orientations_or_durations_failed', 'attempts':attempts,
            'all_candidates_wall_s':time.perf_counter()-begun}


def main():
    process_started_wall=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=180.);p.add_argument('--count',type=int,default=5,help='连续投放数量');p.add_argument('--seed',type=int,default=1543,help='随机 T-LESS 生成种子');p.add_argument('--start-reserve-s',type=float,default=1.5);p.add_argument('--belt-speed',type=float,default=.1,help='仿真传送带速度 m/s；控制输入仍为视觉测速');p.add_argument('--vision-max-speed',type=float,default=.30,help='仿真测速允许的最大值 m/s，不作为速度测量值');p.add_argument('--min-speed-span-s',type=float,default=.45,help='冻结前要求的视觉速度观测时长秒数');p.add_argument('--min-track-age-s',type=float,default=0.,help='冻结前要求同一视觉轨迹累计观测时长');p.add_argument('--freeze-upstream-distance-m',type=float,default=DEFAULT_FREEZE_UPSTREAM_DISTANCE_M,help='上游冻结观测的最大距离 m');p.add_argument('--bin-id',type=int,default=1,help='仿真投放料框 ID，未知类别采用此明确规则');p.add_argument('--viewer',action='store_true');p.add_argument('--hold-viewer',action='store_true');p.add_argument('--profile',choices=('legacy','smooth'),default='legacy',help='smooth 接入平滑参考，默认保留旧入口');p.add_argument('--approach-s',type=float,default=2.,help='smooth 模式接近持续秒数');p.add_argument('--brake-s',type=float,default=1.5,help='确认夹持后的减速秒数');p.add_argument('--sim-acceleration-limit',type=float,help='仅覆盖仿真关节加速度 rad/s²，留空沿用仿真配置');p.add_argument('--follow-s',type=float,default=.8,help='smooth 闭合跟随秒数，保留 0.2s 夹持确认，硬件对齐模式读取共享提前量')
    p.add_argument('--sim-velocity-limit',type=float,help='仅覆盖仿真关节速度 rad/s，便于与真机软件包络对照')
    p.add_argument('--hardware-conditions',type=Path,help='读取真机配置并对齐控制频率、限制、夹持及到位条件，仍使用明确的合成现场')
    p.add_argument('--simulation-use-hardware-standby',action='store_true',help='仅在仿真初始化时使用真机配置的等待位关节角；现场和料框几何仍未对齐')
    p.add_argument('--sim-gravity-compensation',action='store_true',help='仅仿真：机器人模型重力前馈；不能视为已确认厂家控制器行为')
    p.add_argument('--sim-arm-integral-time-s',type=float,help='仅仿真内环模型：位置积分时间常数；无真机辨识，不改变轨迹或关节状态')
    p.add_argument('--sim-lift-m',type=float,default=.20,help='仅仿真：抓取后抬升高度 m，默认 0.20；不修改真机配置')
    p.add_argument('--belt-speed-range',type=float,nargs=2,metavar=('MIN','MAX'),help='每件独立随机带速 m/s；视觉仍自行测速')
    p.add_argument('--spawn-y-range',type=float,nargs=2,metavar=('MIN','MAX'),help='每件随机横向世界Y位置 m')
    yaw_options=p.add_mutually_exclusive_group()
    yaw_options.add_argument('--spawn-yaw-deg',type=float,help='工件生成时绕世界Z轴旋转角度（度）；仅改变初始姿态')
    yaw_options.add_argument('--spawn-yaw-range-deg',type=float,nargs=2,metavar=('MIN','MAX'),help='每件工件独立随机的初始Z轴角度（度）')
    p.add_argument('--fixed-prediction-speed-m-s',type=float,help='预测使用用户指定的沿世界X方向带速 m/s；观测定位和OBB仍来自相机')
    p.add_argument('--sim-visual-speed-estimator',choices=('constant_fit','kalman','corrected_kalman'),default='corrected_kalman',help='仅仿真：修正位置卡尔曼、恒速拟合或原始卡尔曼测速')
    p.add_argument('--sim-speed-position-source',choices=('percentile_box','legacy_center'),default='percentile_box',help='仅仿真：恒速拟合的位置取点云5%%~95%%包围中心，或原二维轮廓中心与深度中位数')
    p.add_argument('--sim-observation-diagnostics',action='store_true',help='仅仿真：记录目标二维中心、深度与候选三维点用于离线分析')
    p.add_argument('--category-ids',type=int,nargs='+',help='使用指定的 T-LESS 工件类别编号，按种子随机排序')
    p.add_argument('--bin-placement',choices=('fixed','free_space'),default='fixed',help='仿真料框固定落点或按已放物件占用选择空闲落点')
    p.add_argument('--perception-view',action='store_true',help='打开只读仿真 RGB-D 感知画面')
    p.add_argument('--synchronized-approach',action='store_true',
        help='实验模式：先连续接入，再同速跟随后闭爪')
    p.add_argument('--synchronized-duration-s',type=float,default=.2,
        help='闭爪前在抓取高度同向同速跟随的时间，默认 0.2 秒；闭爪跟随时间另计')
    p.add_argument('--sim-join-candidates-s',type=float,nargs='+',help='仅仿真：按优先顺序尝试同步接入时间')
    p.add_argument('--sim-follow-s',type=float,help='仅仿真：闭爪后同速跟随时长，至少 0.4 秒')
    p.add_argument('--sim-experimental-short-follow-s',type=float,
        help='仅仿真实验：闭爪后提前进入减速；仍须在抬升前完成原夹持确认')
    p.add_argument('--sim-brake-s',type=float,help='仅仿真：跟随结束后的减速时长')
    p.add_argument('--rotate-before-approach-s',type=float,help='仅仿真：接入前在等待位置转到抓姿的时长；同步接入默认 0.5 秒，设为 0 时旋转与接入同步')
    p.add_argument('--early-visual-orientation',action=argparse.BooleanOptionalAction,default=None,help='仅仿真：先从上游点云预转，再用新观测冻结抓取位置与视觉速度；同步接入默认启用')
    p.add_argument('--early-rotation-s',type=float,default=.5,help='仅仿真：提前转向期望时长，关节约束可能使其延长')
    p.add_argument('--early-min-lead-s',type=float,default=3.2,help='仅仿真：距最终冻结位置约多少秒时选择稳定的上游点云')
    p.add_argument('--pipeline',action='store_true',help='仅仿真：夹持后并行预规划抬升，投放期间提前生成下一件')
    p.add_argument('--pipeline-feed-clearance-s',type=float,default=4.8,help='仅仿真：投放开始至机械臂可再次抓取的估计时间，用于按带速安排进料')
    p.add_argument('--sim-arm-command-advance-s',type=float,default=.1,help='仅仿真：关节轨迹采样提前量；夹爪提前量仍为 0.1 秒')
    p.add_argument('--sim-gripper-command-advance-s',type=float,default=.1,help='仅仿真：闭爪命令提前量，默认 0.1 秒')
    p.add_argument('--sim-close-ramp-s',type=float,default=0.,help='仅仿真：平滑闭爪命令的持续时间，0 为原瞬时命令')
    p.add_argument('--sim-return-accel-limit',type=float,help='仅仿真：返回轨迹规划时的加速度上限 rad/s²')
    p.add_argument('--sim-physics-timestep-s',type=float,help='仅仿真：物理积分步长，必须整除控制周期')
    p.add_argument('--sim-j6-frictionloss',type=float,help='仅仿真诊断：覆盖 J6 关节摩擦力矩，单位 Nm')
    p.add_argument('--sim-interpolate-arm-targets',action='store_true',help='仅仿真：在控制更新之间插值电机位置目标')
    p.add_argument('--trace-physics',action='store_true',help='记录 MuJoCo 每个物理子步的关节加速度及峰值接触')
    p.add_argument('--trace-physics-window-s',type=float,nargs=2,help='仅仿真：额外保存此仿真时钟区间内的逐子步数据')
    args=p.parse_args()
    if args.early_visual_orientation is None:
        # Explicit legacy rotation timing retains the prior one-freeze path.
        args.early_visual_orientation=bool(args.synchronized_approach and args.rotate_before_approach_s is None)
    if args.rotate_before_approach_s is None:
        args.rotate_before_approach_s=.5 if args.synchronized_approach else 0.
    if args.early_visual_orientation:
        if not args.synchronized_approach or args.hardware_conditions is None or args.profile!='smooth':
            p.error('early visual orientation requires smooth synchronized simulation with hardware conditions')
        if not np.isfinite([args.early_rotation_s,args.early_min_lead_s]).all() or min(args.early_rotation_s,args.early_min_lead_s)<=0:
            p.error('early rotation duration and upstream lead must be positive')
        # Residual differences from the later frame are handled continuously
        # during the join; do not schedule another mandatory in-place rotation.
        args.rotate_before_approach_s=0.
    conditions=None
    prealign_s=0.
    if args.hardware_conditions is not None:
        if args.profile!='smooth':p.error('hardware conditions require smooth')
        from experiments.two_second_grasp.hardware_conditions import HardwareConditions
        conditions=HardwareConditions(args.hardware_conditions)
        c=conditions.config
        args.approach_s=c.raw['approach_duration_candidates_s'][0]
        args.follow_s=c.settings.full_plan_close_window_s;args.brake_s=c.raw['brake_duration_s']
        prealign_s=c.raw.get('prealign_orientation_s',0.)
        args.start_reserve_s=c.raw['planning_reserve_s']+prealign_s;args.freeze_upstream_distance_m=c.raw['freeze_upstream_distance_m']
        args.min_speed_span_s=c.raw['minimum_speed_span_s'];args.vision_max_speed=c.raw['maximum_measured_speed_m_s']
    if args.synchronized_approach and (args.profile != 'smooth' or conditions is None or prealign_s <= 0):
        p.error('synchronized-approach requires smooth hardware conditions and positive join duration')
    if not np.isfinite(args.synchronized_duration_s) or args.synchronized_duration_s < 0:
        p.error('synchronized duration must be nonnegative')
    if not np.isfinite(args.rotate_before_approach_s) or args.rotate_before_approach_s < 0:
        p.error('rotate-before-approach duration must be nonnegative')
    if args.rotate_before_approach_s and not args.synchronized_approach:
        p.error('rotate-before-approach requires synchronized approach')
    if args.sim_join_candidates_s is not None:
        candidates=np.asarray(args.sim_join_candidates_s,float)
        if (not args.synchronized_approach or not np.isfinite(candidates).all()
                or np.any(candidates<=0) or len(np.unique(candidates))!=len(candidates)):
            p.error('simulation join candidates must be unique positive values for synchronized approach')
    if args.pipeline and (args.profile != 'smooth' or conditions is None):
        p.error('pipeline requires smooth hardware-conditions simulation')
    if not np.isfinite(args.pipeline_feed_clearance_s) or args.pipeline_feed_clearance_s <= 0:
        p.error('pipeline feed clearance estimate must be positive')
    if not np.isfinite(args.sim_arm_command_advance_s) or args.sim_arm_command_advance_s < 0:
        p.error('simulation arm advance must be nonnegative')
    if not np.isfinite(args.sim_gripper_command_advance_s) or args.sim_gripper_command_advance_s < 0:
        p.error('simulation gripper advance must be nonnegative')
    if not np.isfinite(args.sim_close_ramp_s) or args.sim_close_ramp_s < 0:
        p.error('simulation close ramp must be nonnegative')
    if args.sim_return_accel_limit is not None and (not np.isfinite(args.sim_return_accel_limit) or args.sim_return_accel_limit <= 0 or conditions is None):
        p.error('simulation return acceleration limit requires conditions and a positive value')
    if args.sim_physics_timestep_s is not None and (not np.isfinite(args.sim_physics_timestep_s) or args.sim_physics_timestep_s<=0):
        p.error('simulation physics timestep must be positive')
    if args.sim_j6_frictionloss is not None and (not np.isfinite(args.sim_j6_frictionloss) or args.sim_j6_frictionloss<0):
        p.error('simulation joint friction must be nonnegative')
    if args.trace_physics_window_s is not None and (not args.trace_physics or not np.isfinite(args.trace_physics_window_s).all() or args.trace_physics_window_s[0]>=args.trace_physics_window_s[1]):
        p.error('physics trace window requires tracing and increasing finite times')
    if args.sim_follow_s is not None:
        if not np.isfinite(args.sim_follow_s) or args.sim_follow_s < .4:p.error('simulation follow duration must be at least 0.4 seconds')
        args.follow_s=args.sim_follow_s
    if args.sim_experimental_short_follow_s is not None:
        short_follow=args.sim_experimental_short_follow_s
        if (args.sim_follow_s is not None or conditions is None or not args.synchronized_approach
                or conditions.config.raw.get('closure_policy','feedback') != 'scheduled_attempt'
                or not np.isfinite(short_follow) or not 0 < short_follow < .4):
            p.error('experimental short follow needs synchronized scheduled-attempt simulation, 0 < duration < 0.4, and no sim-follow-s')
        args.follow_s=short_follow
    if args.sim_brake_s is not None:
        if not np.isfinite(args.sim_brake_s) or args.sim_brake_s <= 0:p.error('simulation braking duration must be positive')
        args.brake_s=args.sim_brake_s
    if args.sim_arm_integral_time_s is not None and (conditions is None or not np.isfinite(args.sim_arm_integral_time_s) or args.sim_arm_integral_time_s<=0):p.error('simulation integral model needs hardware-conditions and a positive time constant')
    if args.sim_lift_m is not None and (not np.isfinite(args.sim_lift_m) or args.sim_lift_m <= 0):p.error('simulation lift must be positive')
    if args.sim_gravity_compensation and conditions is None:p.error('gravity compensation requires explicit hardware-conditions mode')
    if args.simulation_use_hardware_standby and conditions is None:p.error('simulation-use-hardware-standby requires hardware-conditions')
    scheduled_attempt=conditions is not None and conditions.config.raw.get('closure_policy','feedback')=='scheduled_attempt'
    if args.count < 1 or args.count > 5:p.error('count must be between 1 and 5')
    if args.viewer and not os.environ.get('DISPLAY'):p.error('--viewer needs DISPLAY')
    if args.perception_view and not os.environ.get('DISPLAY'):p.error('--perception-view needs DISPLAY')
    if args.hold_viewer and not args.viewer:p.error('--hold-viewer needs --viewer')
    if not np.isfinite(args.start_reserve_s) or args.start_reserve_s<=0:p.error('start reserve must be positive')
    if not np.isfinite(args.belt_speed) or not 0 < args.belt_speed <= 1.:p.error('belt speed must be in (0, 1] m/s')
    if args.spawn_yaw_deg is not None and not np.isfinite(args.spawn_yaw_deg):p.error('spawn yaw must be finite')
    if args.fixed_prediction_speed_m_s is not None and (not np.isfinite(args.fixed_prediction_speed_m_s) or not 0 < args.fixed_prediction_speed_m_s <= 1.):p.error('fixed prediction speed must be in (0, 1] m/s')
    if not np.isfinite(args.vision_max_speed) or args.vision_max_speed <= 0:p.error('vision max speed must be positive')
    if not np.isfinite(args.min_speed_span_s) or not 0 <= args.min_speed_span_s <= .6:p.error('speed observation span must be between 0 and 0.6s')
    if not np.isfinite(args.min_track_age_s) or args.min_track_age_s < 0:p.error('track age must be nonnegative')
    if not np.isfinite(args.freeze_upstream_distance_m) or args.freeze_upstream_distance_m <= 0:p.error('upstream distance must be positive')
    if (not np.isfinite([args.approach_s,args.brake_s,args.follow_s]).all()
            or min(args.approach_s,args.brake_s,args.follow_s)<=0
            or (args.follow_s<.4 and args.sim_experimental_short_follow_s is None)):
        p.error('smooth durations must be positive; follow below 0.4 requires explicit simulation experiment')
    if args.profile=='legacy' and args.approach_s!=2.:p.error('--approach-s applies only to smooth')
    if args.sim_acceleration_limit is not None and (not np.isfinite(args.sim_acceleration_limit) or args.sim_acceleration_limit<=0):p.error('acceleration must be positive')
    if args.sim_velocity_limit is not None and (not np.isfinite(args.sim_velocity_limit) or args.sim_velocity_limit<=0):p.error('velocity must be positive')
    base=Path(__file__).resolve().parents[1]/'src/nero_core/experiments/graspgenx_sim';cfg=yaml.safe_load((base/'nero7_config.yaml').read_text());cfg['experiment'].update(seed=args.seed,belt_speed_m_s=args.belt_speed,feed_interval_s=1000.,objects=args.count)
    if args.category_ids is not None:
        catalog=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/nero_core/mujoco_conveyor_sorting/configs/tless_objects.yaml').read_text())
        supported={int(s['category_id']) for s in catalog['object_specs']}
        if len(set(args.category_ids)) != len(args.category_ids) or not set(args.category_ids) <= supported or len(args.category_ids)<args.count:
            p.error('category-ids 必须是已有 T-LESS 类别，互不重复且数量不少于 count')
        cfg['experiment']['active_category_ids']=args.category_ids
    category_sequence=list(np.random.default_rng(args.seed).permutation(cfg['experiment']['active_category_ids'])[:args.count])
    try:
        bounds(args.spawn_y_range,'spawn-y-range')
        yaw_range=None if args.spawn_yaw_range_deg is None else np.deg2rad(args.spawn_yaw_range_deg)
        trials=make_trials(category_sequence,args.seed,args.belt_speed,speed_range=args.belt_speed_range,yaw_range=yaw_range)
        if args.spawn_yaw_deg is not None:
            for trial in trials:trial['spawn_yaw_rad']=float(np.deg2rad(args.spawn_yaw_deg))
    except ValueError as error:p.error(str(error))
    spawn_records=[]
    overrides=config_overrides(cfg);overrides['robot']['robot']['initialize_at_wait_pose']=True
    if args.category_ids is not None:
        scene=yaml.safe_load((Path(__file__).resolve().parents[1]/'src/nero_core/mujoco_conveyor_sorting/configs/scene.yaml').read_text())
        assignments={int(spec['category_id']):int(spec['bin_id']) for spec in catalog['object_specs']}
        for bin_item in scene['bins']:
            bin_item['category_ids']=[cid for cid in args.category_ids if assignments[cid]==int(bin_item['id'])]
        overrides['scene']['bins']=scene['bins']
    if args.spawn_y_range is not None:
        # Keep the native spawn RNG order for yaw/material/friction: changing
        # lateral bounds must not silently skip its draw and alter other inputs.
        overrides['scene']['randomization']['object_lateral_position']={'enabled':True,'range':args.spawn_y_range}
    overrides['scene'].setdefault('workspace',{}).update(waiting_ee_position=cfg['control']['simulation_standby_tcp_world_m'],waiting_ee_quaternion_wxyz=cfg['control']['simulation_standby_quaternion_wxyz'])
    if args.sim_acceleration_limit is not None:overrides['robot']['robot']['max_joint_acceleration']=[args.sim_acceleration_limit]*7
    if args.sim_velocity_limit is not None:overrides['robot']['robot']['max_joint_velocity']=[args.sim_velocity_limit]*7
    if conditions is not None:conditions.apply(overrides)
    if args.simulation_use_hardware_standby:
        overrides['robot']['robot']['initialize_at_wait_pose']=False
        overrides['robot']['robot']['home_joint_positions']=conditions.config.standby.tolist()
    env=IndependentControllerSimulation(capture_path=base/cfg['camera']['capture_reference'].split('experiments/graspgenx_sim/')[-1],config_overrides=overrides,grasp_assist=False,perception_mode='rendered_rgbd',grasp_event_source='physical_contacts',seed=args.seed,action_type='joint_position',preserve_control_profile=True,maximum_measured_speed_m_s=args.vision_max_speed,simulation_max_height_m=.08 if conditions is None else conditions.config.raw['maximum_target_height_m'],simulation_camera_world_xyz=cfg['camera']['early_view_world_xyz_m'],visual_workspace_x_world=cfg['tracking']['early_visual_workspace_x_world_m'])
    original_j6_friction=float(env.model.dof_frictionloss[env.arm_dof_adrs[5]])
    if args.sim_j6_frictionloss is not None:
        env.model.dof_frictionloss[env.arm_dof_adrs[5]]=args.sim_j6_frictionloss
    if args.sim_physics_timestep_s is not None:
        steps=int(round(env.control_dt/args.sim_physics_timestep_s))
        if steps < 1 or not np.isclose(steps*args.sim_physics_timestep_s,env.control_dt,atol=1e-12,rtol=0):
            p.error('simulation physics timestep must divide the control period')
        env.model.opt.timestep=args.sim_physics_timestep_s
        env.frame_skip=steps
    if conditions is not None:conditions.bind(env,gravity_compensation=args.sim_gravity_compensation)
    sim_servo=None
    if args.sim_arm_integral_time_s is not None:
        from experiments.two_second_grasp.sim_servo import IntegralPositionServo
        sim_servo=IntegralPositionServo(env,args.sim_arm_integral_time_s)
    sensor=Nero7SensorControlEnvironment(env,SensorControlConfig(maximum_track_age_s=1.));pool=ThreadPoolExecutor(max_workers=1);viewer=None;future=None;last=-1;cloud=None;pose=None;capture=None;ref=None;trajectory=None;pending=None;path=None;rows=[];cancel=[False];lastdraw=0.;closed=False;target_id=None;phase='intercept';leg_future=None;leg=None;leg_start=None;hold_q=None;release_start=None;object_runtime=None;object_z_at_close=None;placement_rows=[];attempts=[];current_logical_id=None;category_sequence=list(np.random.default_rng(args.seed).permutation(cfg['experiment']['active_category_ids'])[:args.count]);category_cursor=0;evaluation_trace=[];last_evaluation_s=-1.;track_first_seen={};previous_pose=None;previous_pose_s=None;grip_since=None;brake_checked=False;command_rows=[];began_wall=time.perf_counter()
    prefetched_lift=None;prefetch_identity=None;using_prefetched_lift=False;queued_next=None
    prefetched_return=None;return_identity=None;using_prefetched_return=False
    prefetched_transfer=None;transfer_identity=None;transfer_selection=None;using_prefetched_transfer=False
    early_state='idle' if args.early_visual_orientation else 'disabled'
    early_future=None;early_trajectory=None;early_identity=None;early_latest_identity=None;early_latest_s=None;early_start_s=None;early_finished_s=None;early_start_wait_s=None;early_cancel=None;early_invalidated=False
    from experiments.two_second_grasp.early_observation_quality import EarlyObservationQuality
    early_quality=EarlyObservationQuality()
    phase_events=[];last_phase=None;physics_trace=None
    live_speed=None;live_speed_samples=[];visual_position_samples=[];truth_speed=None
    from experiments.two_second_grasp.perception_panel import visual_speed_world, speed_hud_values
    from experiments.two_second_grasp.frozen_cloud_display import FrozenCloudDisplay
    from experiments.two_second_grasp.constant_visual_speed import ConstantVisualSpeed
    from experiments.two_second_grasp.speed_observation import percentile_cloud_center
    from experiments.two_second_grasp.visual_point_kalman import VisualPointKalman
    frozen_display=FrozenCloudDisplay()
    constant_speed=ConstantVisualSpeed(min_samples=8,min_span_s=args.min_speed_span_s)
    corrected_kalman=VisualPointKalman(max_velocity=args.vision_max_speed)
    live_fit=None;live_corrected=None
    result={'controller':'independent_two_second_cartesian','requested_count':args.count,'random_seed':args.seed,'random_tless_category_sequence':category_sequence,'completed_count':0,'attempts':attempts,'simulation_only':True,'network_called':False,'configured_drop_bin_id':args.bin_id,'bin_assignment':'explicit_configured_bin_no_classification','grasp_source':'single_frame_visible_obb','speed_source':'nero7_xy_kalman_from_rendered_rgbd','truth_used_for_control':True,'truth_use_scope':'MuJoCo contact/lift gates and rigid attached object pose for simulation-only loaded collision search; never visual interception input','hardware_connections':0,'status':'waiting','commands_sent':0,'physical_grasp_verified':False,'duration_s':2.0,'command_advance_s':.1,'simulation_belt_set_m_s':args.belt_speed,'vision_max_speed_m_s':args.vision_max_speed,'simulation_detection_max_height_m':.08,'minimum_speed_span_s':args.min_speed_span_s,'freeze_upstream_distance_m':args.freeze_upstream_distance_m,'trigger_mode':'predicted_entry_from_frozen_upstream_visual_observation','visual_diagnostics':{'frames':0,'frames_with_measurements':0,'frames_with_one_qualified_speed':0,'no_observed_state':0,'stale_observation':0,'outside_grasp_region':0,'insufficient_cloud':0,'speed_span_not_ready':0,'too_far_upstream':0,'observed_x_min_m':None,'observed_x_max_m':None},'checks':'sampled IK, static MuJoCo geometry and joint position/velocity; not a full dynamic or moving-scene certificate'}
    try:
        if args.sim_visual_speed_estimator=='constant_fit':
            result['speed_source']='constant_velocity_fit_from_rendered_rgbd_positions'
        elif args.sim_visual_speed_estimator=='corrected_kalman':
            result['speed_source']='simulation_visual_point_kalman'
        result['sim_speed_position_source']=args.sim_speed_position_source
        result.update(environment_construction_wall_s=time.perf_counter()-process_started_wall,profile=args.profile,duration_s=args.approach_s if args.profile=='smooth' else 2.,prealign_s=prealign_s,spawn_yaw_deg=args.spawn_yaw_deg,spawn_yaw_range_deg=args.spawn_yaw_range_deg,brake_s=args.brake_s if args.profile=='smooth' else 0.,follow_s=args.follow_s if args.profile=='smooth' else .6,simulation_acceleration_limits_rad_s2=env.robot_controller.max_joint_acceleration,simulation_velocity_limits_rad_s=env.robot_controller.max_joint_velocity,grasp_assist=False,pipeline_enabled=args.pipeline,phase_events=phase_events,pipeline_events=[],sim_arm_command_advance_s=args.sim_arm_command_advance_s,sim_gripper_command_advance_s=args.sim_gripper_command_advance_s,sim_close_ramp_s=args.sim_close_ramp_s)
        result['early_visual_orientation']={'enabled':args.early_visual_orientation,'requested_duration_s':args.early_rotation_s,'upstream_lead_s':args.early_min_lead_s,'observations':[],'quality_rejections':{}}
        result['experimental_short_follow']=(args.sim_experimental_short_follow_s is not None)
        result['sim_j6_frictionloss']={'original_Nm':original_j6_friction,'used_Nm':float(env.model.dof_frictionloss[env.arm_dof_adrs[5]]),'diagnostic_override':args.sim_j6_frictionloss is not None}
        result['object_scenario']={'object_kind':'tless_mesh'}
        if args.fixed_prediction_speed_m_s is not None:
            result['speed_source']='operator_fixed_belt_speed';result['fixed_prediction_speed_m_s']=args.fixed_prediction_speed_m_s
            result['visual_speed_used_for_prediction']=False
        if args.profile=='smooth':result['checks']='full emitted spline q/v/a extrema; sampled robot/static geometry; actual-grip-gated payload brake check; physical feedback and containment'
        if args.profile=='smooth' and conditions is not None:
            result['checks']='full spline q/v/a extrema; shared dense FK grid and moving visual-cloud 4mm predicate; simulated static geometry; physical grip-gated payload brake'
        if conditions is not None:
            from nero_core.two_second.command_history import CommandHistory
            dispatch_history=CommandHistory(conditions.config)
            result['condition_audit']=conditions.report(env,args.sim_gravity_compensation)
            result['closure_policy']=conditions.config.raw.get('closure_policy','feedback')
            result['simulation_detection_max_height_m']=conditions.config.raw['maximum_target_height_m']
        result.update(scenario_trials=trials,spawn_records=spawn_records,
            bin_placement_mode=args.bin_placement,
            simulation_belt_set_m_s=args.belt_speed if args.belt_speed_range is None else None,
            simulation_belt_speed_range_m_s=args.belt_speed_range,simulation_spawn_y_range_m=args.spawn_y_range)
        if args.bin_placement=='free_space':result['truth_use_scope']+='; simulation-only bin footprint selection from actual placed geometry'
        env.reset(seed=args.seed,options={'belt_speed':trials[0]['belt_set_m_s'],'initial_objects':0})
        if args.sim_interpolate_arm_targets:
            from experiments.two_second_grasp.target_interpolator import SimulationTargetInterpolator
            env._physics_target_interpolator=SimulationTargetInterpolator(env)
            result['simulation_target_interpolation']=env._physics_target_interpolator.report()
        if args.trace_physics:
            from experiments.two_second_grasp.physics_trace import PhysicsAccelerationTrace
            physics_trace=PhysicsAccelerationTrace(env,lambda:phase,args.trace_physics_window_s)
            env._physics_substep_observer=physics_trace
        current_logical_id,spawn_record=spawn_trial(env,trials[0]);spawn_records.append(spawn_record);category_cursor=1
        print('第1件随机场景：'+json.dumps(serializable(spawn_record),ensure_ascii=False),flush=True)
        if current_logical_id is None:raise RuntimeError('first random T-LESS spawn failed')
        start_q=np.asarray(env.get_joint_positions()).copy();wait_q=start_q.copy();start_pose=env._ee_pose().copy();dt=float(env.control_dt)
        if conditions is not None:
            result['condition_audit']['standby_joint_source']=('hardware_config' if args.simulation_use_hardware_standby else 'simulation_wait_pose_ik')
            result['condition_audit']['standby_max_difference_rad']=float(np.max(np.abs(start_q-conditions.config.standby)))
        original_recycle=env.recycle_terminal_objects
        retained_placements=set()
        env.recycle_terminal_objects=lambda *,older_than=.4:original_recycle(older_than=max(8.,older_than),retain_object_ids=retained_placements)
        result.update(initial_joints_rad=start_q,initial_tcp_world_m=start_pose[:3],grasp_region_world_x_m=env.grasp_region,simulation_reset_pose_only=True,drop_pose_world=env.get_bin_drop_pose(args.bin_id))
        if args.viewer:
            from mujoco import viewer as mujoco_viewer
            viewer=mujoco_viewer.launch_passive(env.model,env.data,show_left_ui=False,show_right_ui=False)
            with viewer.lock():
                viewer.cam.lookat[:]=[-.25,-.12,.40];viewer.cam.distance=2.7;viewer.cam.azimuth=110.;viewer.cam.elevation=-32.
            print('冻结点云：紫色为提前转向，青色为抓取规划；橙色与黄色分别表示选定方向。轨迹：绿色接近、青色跟随、橙色减速。',flush=True)
        if args.perception_view:
            import cv2
            from experiments.two_second_grasp.perception_panel import render_panel
            cv2.namedWindow('MuJoCo RGB-D perception', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('MuJoCo RGB-D perception', 1280, 520)
            cv2.waitKey(1)
            print('已打开仿真 RGB-D 感知窗口；黄色为检测轮廓。',flush=True)
        last_panel_wall = 0.
        while env.data.time<args.seconds and (viewer is None or viewer.is_running()):
            queued=pending;step_started_wall=time.perf_counter()
            if conditions is not None and pending is not None:pending['arm_time_parameterized']=True
            env.step(pending);pending=None;observation=sensor.get_observation();now=float(env.data.time)
            if phase != last_phase:
                phase_events.append({'phase':phase,'sim_s':now,'wall_s':time.perf_counter()-began_wall,
                                     'logical_id':current_logical_id})
                last_phase=phase
            if 'first_step_and_sensor_wall_s' not in result:result['first_step_and_sensor_wall_s']=time.perf_counter()-step_started_wall
            if args.profile=='smooth' and queued is not None:
                command=env.robot_controller._last_command
                if conditions is not None and command is not None:
                    motion=select_motion_profile(conditions.config,
                        payload=phase in ('planning_lift','lift','planning_transfer','transfer'))
                    dispatch_history.record(now,dt,np.asarray(command.joint_target),{},limits=motion)
                if command is not None:command_rows.append({'sim_s':now,'phase':phase,'requested_q_rad':command.requested_arm,'applied_q_rad':command.joint_target,'applied_qd_rad_s':command.joint_target_velocity,'clipped':command.clipped,'time_parameterized':command.time_parameterized,'actual_q_rad':env.get_joint_positions().copy(),'actual_qd_rad_s':env.get_joint_velocities().copy(),'actual_qdd_rad_s2':env.data.qacc[env.arm_dof_adrs].copy()})
            if now-last_evaluation_s>=.5:
                eval_runtime=next((r for r in env._objects if r.logical_id==current_logical_id),None)
                evaluation_trace.append({'sim_s':now,'trial_index':category_cursor,'belt_speed_m_s':env.belt_speed,'object_position_world_m':None if eval_runtime is None else env.data.xpos[eval_runtime.body_id].copy(),'state':None if eval_runtime is None else eval_runtime.state,'camera_measurement_track_ids':list(env._nero_records)})
                last_evaluation_s=now
            for frame in tuple(env._camera_frames):
                meta=frame.get('nero_capture')
                if meta is None or meta['sequence']<=last:continue
                last=meta['sequence']
                live_speed=visual_speed_world(frame,env.nero_base_to_world[:3,:3],track_id=target_id)
                runtime=next((r for r in env._objects if r.logical_id==current_logical_id),None)
                truth_speed=(None if runtime is None else float(env.data.qvel[runtime.qvel_adr]))
                # Diagnostic only: capture the unfiltered visual centroid from the
                # same image/track, including frames before speed qualification.
                # Simulator ground truth is deliberately kept out of this record.
                live_fit=None;live_corrected=None
                for identity, entry in meta['measurements'].items():
                    capture_stamp, observed = sensor._observed_state_at(identity, float(frame['timestamp']))
                    if observed is None or abs(capture_stamp-float(frame['timestamp'])) > 1e-6:
                        continue
                    report = entry.get('velocity_measurement') or {}
                    if entry.get('track_generation') is None:
                        continue
                    fit_identity=(int(meta['clock_epoch']),int(identity),int(entry['track_generation']))
                    speed_position=(percentile_cloud_center(frame,identity)
                        if args.sim_speed_position_source=='percentile_box' else np.asarray(observed['position']))
                    if speed_position is None:
                        continue
                    constant_speed.update(fit_identity,capture_stamp,speed_position)
                    corrected_kalman.update(fit_identity,capture_stamp,speed_position)
                    if (target_id is None and len(meta['measurements'])==1) or int(identity)==target_id:
                        live_fit=constant_speed.estimate(fit_identity)
                        live_corrected=corrected_kalman.estimate(fit_identity)
                    visual_position_samples.append({
                        'frame_sequence': int(meta['sequence']), 'frame_time_s': capture_stamp,
                        'clock_epoch': int(meta['clock_epoch']),
                        'track_id': int(identity), 'track_generation': entry.get('track_generation'),
                        'position_world_m': np.asarray(observed['position']).tolist(),
                        'speed_position_world_m': np.asarray(speed_position).tolist(),
                        'speed_position_source': args.sim_speed_position_source,
                        'speed_valid': bool(report.get('speed_valid')),
                        'velocity_base_m_s': report.get('velocity_base_m_s'),
                        'span_s': report.get('span_s'), 'sample_count': report.get('sample_count')})
                    if args.sim_observation_diagnostics:
                        from experiments.two_second_grasp.observation_diagnostics import inspect_observation
                        visual_position_samples[-1]['camera_diagnostics']=inspect_observation(frame,identity)
                        if 'observation_diagnostic_calibration' not in result:
                            result['observation_diagnostic_calibration']={key:frame['top_camera'][key]
                                for key in ('intrinsics','camera_to_world_opencv')}
                if live_speed is not None:
                    live_speed_samples.append({'frame_sequence':int(meta['sequence']),
                        'frame_time_s':live_speed['frame_time_s'],'track_id':live_speed['track_id'],
                        'velocity_world_m_s':live_speed['velocity_world_m_s'].tolist(),
                        'simulation_object_x_velocity_m_s':truth_speed})
                if args.perception_view and time.monotonic()-last_panel_wall >= .12:
                    speed_lines=speed_hud_values(live_speed,now_s=now,
                        frozen_velocity=result.get('velocity_world_m_s'),belt_speed=env.belt_speed,truth_speed=truth_speed)
                    speed_lines=(*speed_lines, 'Constant fit: --' if live_fit is None else
                        f"Constant fit: {live_fit['velocity_world_m_s'][0]*100:+.2f} cm/s ({live_fit['sample_count']} frames)")
                    speed_lines=(*speed_lines, 'Corrected Kalman: --' if live_corrected is None else
                        f"Corrected Kalman: {live_corrected['velocity_world_m_s'][0]*100:+.2f} cm/s")
                    panel=render_panel(frame,status=result['status'],completed=result['completed_count'],
                        requested=args.count,fixed_speed_m_s=args.fixed_prediction_speed_m_s,speed_lines=speed_lines,
                        visual_speed_source=args.sim_visual_speed_estimator)
                    cv2.imshow('MuJoCo RGB-D perception',panel)
                    cv2.waitKey(1)
                    last_panel_wall=time.monotonic()
                diag=result['visual_diagnostics'];diag['frames']+=1;diag['frames_with_measurements']+=int(bool(meta['measurements']))
                valid=[(k,v) for k,v in meta['measurements'].items() if v['velocity_measurement'].get('speed_valid') and v['velocity_measurement'].get('motion_source')=='nero7_xy_kalman']
                for key, measurement in meta['measurements'].items():
                    identity=(key,measurement.get('track_generation'))
                    track_first_seen.setdefault(identity,float(frame['timestamp']))
                if len(valid)==1:diag['frames_with_one_qualified_speed']+=1
                if args.early_visual_orientation and phase=='intercept' and ref is None and len(valid)==1:
                    early_tid,early_entry=valid[0]
                    identity=(int(meta['clock_epoch']),int(early_tid),int(early_entry['track_generation']))
                    early_latest_identity=identity;early_latest_s=float(frame['timestamp'])
                    if early_identity is not None and identity!=early_identity:
                        if not early_invalidated:
                            frozen_display.clear()
                            result['early_visual_orientation']['observations'][-1].update(invalidated_reason='target_identity_changed',replacement_identity=identity,invalidation_s=now)
                            if early_cancel is not None:early_cancel[0]=True
                            if early_future is not None:early_future.cancel()
                            early_invalidated=True
                            print('目标视觉身份已变化，丢弃旧预转结果并重新获取姿态。',flush=True)
                        if early_state!='executing':
                            early_state='idle';early_identity=None;early_future=None;early_trajectory=None;early_cancel=None;early_invalidated=False;early_quality.clear()
                    if early_state=='idle':
                        early_stamp=float(frame['timestamp'])
                        _,early_observed=sensor._observed_state_at(early_tid,early_stamp)
                        early_cloud=cloud_for_track(frame,early_tid)
                        quality=early_quality.update(identity=identity,stamp=early_stamp,
                            mask=(np.asarray(frame['top_instance_labels'])==int(early_tid)),depth=frame['top_depth'],cloud=early_cloud)
                        if early_observed is not None and quality['ready']:
                            from nero_core.two_second.synchronized_reference import freeze_window_for_speed
                            early_velocity=env.nero_base_to_world[:3,:3]@np.asarray(early_entry['velocity_measurement']['velocity_base_m_s'],float)
                            if args.sim_visual_speed_estimator=='constant_fit':
                                early_fit=constant_speed.estimate(identity)
                                if early_fit is None or abs(early_fit['last_stamp_s']-early_stamp)>1e-6:continue
                                early_velocity=early_fit['velocity_world_m_s'].copy()
                            elif args.sim_visual_speed_estimator=='corrected_kalman':
                                early_fit=corrected_kalman.estimate(identity)
                                if early_fit is None or abs(early_fit['last_stamp_s']-early_stamp)>1e-6:continue
                                early_velocity=early_fit['velocity_world_m_s'].copy()
                            if early_velocity[0]<=0:continue
                            early_window=freeze_window_for_speed(conditions.config,float(early_velocity[0]))
                            freeze_boundary=float(env.grasp_region[0])-early_window['distance_m']
                            selection_boundary=freeze_boundary-float(early_velocity[0])*args.early_min_lead_s
                            if float(early_observed['position'][0])<selection_boundary:
                                counts=result['early_visual_orientation']['quality_rejections'];counts['waiting_closer_view']=counts.get('waiting_closer_view',0)+1
                                continue
                            from experiments.two_second_grasp.early_orientation import plan_early_orientation
                            early_identity=identity;early_state='planning';early_cancel=[False]
                            # Connect to the already commanded rest endpoint;
                            # the measured joints can differ by arrival tolerance.
                            early_q=np.asarray(env.robot_controller._last_joint_target).copy()
                            early_pose=env.ik.forward_pose(early_q)
                            camera_world=np.asarray(frame['top_camera']['camera_to_world'])[:3,3]
                            early_record={'identity':identity,'capture_s':early_stamp,'frame_sequence':int(meta['sequence']),
                                'point_count':len(early_cloud),'point_cloud_world_m':early_cloud.tolist(),
                                'observed_center_world_m':np.asarray(early_observed['position']).tolist(),
                                'quality':quality,'camera_distance_m':float(np.linalg.norm(np.asarray(early_observed['position'])-camera_world)),
                                'selection_boundary_world_x_m':selection_boundary,'selection_velocity_world_m_s':early_velocity.tolist()}
                            result['early_visual_orientation']['observations'].append(early_record)
                            frozen_display.freeze('early',identity,early_stamp,meta['sequence'],early_cloud,
                                obb_top_down_grasp(early_cloud)['box'])
                            early_future=pool.submit(plan_early_orientation,env,np.asarray(env.data.qpos).copy(),early_q,early_pose,early_cloud.copy(),dt,args.early_rotation_s,conditions,lambda token=early_cancel:cancel[0] or token[0])
                            print('上游连续观测稳定，冻结目标点云并后台规划提前转向。',flush=True)
                        elif not quality['ready']:
                            counts=result['early_visual_orientation']['quality_rejections'];reason=quality['reason'];counts[reason]=counts.get(reason,0)+1
                if args.early_visual_orientation and early_state!='complete':continue
                if ref is not None or len(valid)!=1:continue
                tid,entry=valid[0];stamp=float(frame['timestamp']);speed_report=entry['velocity_measurement']
                if args.early_visual_orientation and (early_identity!=(int(meta['clock_epoch']),int(tid),int(entry['track_generation'])) or stamp<=early_finished_s):continue
                track_age=stamp-track_first_seen[(tid,entry.get('track_generation'))]
                if track_age < args.min_track_age_s or float(speed_report.get('span_s',0.)) < args.min_speed_span_s:diag['speed_span_not_ready']+=1;continue
                _,observed=sensor._observed_state_at(tid,stamp)
                if observed is None:diag['no_observed_state']+=1;continue
                if now-stamp>1.:diag['stale_observation']+=1;continue
                pos=np.asarray(observed['position']);
                diag['observed_x_min_m']=float(pos[0]) if diag['observed_x_min_m'] is None else min(diag['observed_x_min_m'],float(pos[0]));diag['observed_x_max_m']=float(pos[0]) if diag['observed_x_max_m'] is None else max(diag['observed_x_max_m'],float(pos[0]))
                velocity=env.nero_base_to_world[:3,:3]@np.asarray(entry['velocity_measurement']['velocity_base_m_s'],float)
                kalman_velocity=velocity.copy()
                fit_identity=(int(meta['clock_epoch']),int(tid),int(entry['track_generation']))
                frozen_fit=constant_speed.estimate(fit_identity)
                frozen_corrected=corrected_kalman.estimate(fit_identity)
                if args.sim_visual_speed_estimator=='constant_fit' and args.fixed_prediction_speed_m_s is None:
                    if frozen_fit is None or abs(frozen_fit['last_stamp_s']-stamp)>1e-6:continue
                    velocity=frozen_fit['velocity_world_m_s'].copy()
                elif args.sim_visual_speed_estimator=='corrected_kalman' and args.fixed_prediction_speed_m_s is None:
                    if frozen_corrected is None or abs(frozen_corrected['last_stamp_s']-stamp)>1e-6:continue
                    velocity=frozen_corrected['velocity_world_m_s'].copy()
                if args.fixed_prediction_speed_m_s is not None:
                    velocity=np.array([args.fixed_prediction_speed_m_s,0.,0.])
                try:
                    if args.synchronized_approach:
                        # Match the hardware admission rule; the exact join is
                        # scheduled only after the OBB grasp has been computed.
                        from nero_core.two_second.synchronized_reference import freeze_window_for_speed, admit_synchronized_observation
                        freeze_window=freeze_window_for_speed(conditions.config,float(velocity[0]))
                        entry_s=admit_synchronized_observation(stamp,float(pos[0]),float(velocity[0]),env.grasp_region,now,freeze_window['distance_m'])
                        result['effective_freeze_window']=freeze_window
                    else:
                        entry_s=predict_entry_time(stamp,float(pos[0]),float(velocity[0]),float(env.grasp_region[0]),now,args.start_reserve_s,
                            maximum_upstream_distance_m=args.freeze_upstream_distance_m)
                except EntryNotReady as error:
                    diag['too_far_upstream' if error.reason=='too_far_upstream' else 'outside_grasp_region']+=1;continue
                except ValueError:diag['outside_grasp_region']+=1;continue
                cloud=cloud_for_track(frame,tid)
                if len(cloud)<100:diag['insufficient_cloud']+=1;continue
                box=obb_top_down_grasp(cloud);pose=box['tcp'];target_id=int(tid);capture=stamp
                final_display_identity=(int(meta['clock_epoch']),int(tid),int(entry['track_generation']))
                frozen_display.freeze('final',final_display_identity,stamp,meta['sequence'],cloud,box['box'])
                quat=Rotation.from_matrix(pose[:3,:3]).as_quat()[[3,0,1,2]]
                result['initial_obb_candidate_quaternion_wxyz']=quat.tolist()
                if args.synchronized_approach:
                    from nero_core.two_second.synchronized_reference import SynchronizedIntercept, schedule_forward_join
                    join_candidates=args.sim_join_candidates_s or [prealign_s]
                    schedule_failures=[];timing=None;initial_join=None
                    for join_s in join_candidates:
                        try:
                            timing=schedule_forward_join(capture_s=stamp,grasp_position=pose[:3,3],velocity=velocity,
                                standby_position=start_pose[:3],direction=[1.,0.,0.],now_s=now,
                                planning_reserve_s=conditions.config.raw['planning_reserve_s'],
                                advance_s=conditions.config.raw['command_advance_s'],join_s=join_s,
                                synchronized_s=args.synchronized_duration_s,follow_s=args.follow_s,follow_bounds=env.grasp_region)
                            initial_join=join_s;break
                        except ValueError as error:
                            schedule_failures.append({'join_s':join_s,'reason':str(error)})
                    result['synchronized_schedule_rejections']=schedule_failures
                    if timing is None:
                        result.update(status='synchronized_window_infeasible',error='no candidate join time fits grasp interval');break
                    ref=SynchronizedIntercept(stamp,timing['join_start_s']-args.rotate_before_approach_s,
                        args.rotate_before_approach_s+initial_join+args.synchronized_duration_s,pose[:3,3],velocity,
                        start_pose[:3],start_pose[3:],quat,args.follow_s,args.brake_s,
                        args.rotate_before_approach_s+initial_join,args.rotate_before_approach_s)
                    result['synchronized_schedule']=timing
                    result['intercept_strategy']='forward_join_then_matched_velocity'
                    result['synchronized_duration_s']=args.synchronized_duration_s
                    result['rotate_before_approach_s']=args.rotate_before_approach_s
                else:
                    ref=(SmoothDiagnosticReference(stamp,entry_s-prealign_s,args.approach_s+prealign_s,pose[:3,3],velocity,start_pose[:3],start_pose[3:],quat,brake_s=args.brake_s,follow_s=args.follow_s,prealign_s=prealign_s) if args.profile=='smooth' else Intercept(stamp,entry_s,APPROACH_DURATION_S,pose[:3,3],velocity,start_pose[:3],start_pose[3:],quat))
                result.update(status='planning',capture_s=stamp,observation_age_s=now-stamp,preparation_sim_s=now,predicted_entry_sim_s=entry_s,track_id=target_id,observed_center_world_m=pos,observed_obb_tcp_world_m=pose[:3,3],velocity_world_m_s=velocity,velocity_sample_count=speed_report.get('sample_count'),visual_track_age_s=track_age,velocity_span_s=speed_report.get('span_s'),velocity_speed_std_m_s=speed_report.get('speed_std_m_s'),scheduled_start_s=ref.start_s,scheduled_contact_s=ref.contact_s,predicted_contact_tcp_world_m=ref.contact_xyz)
                result['frozen_speed_comparison']={'kalman_world_m_s':kalman_velocity,'constant_fit':frozen_fit,
                    'corrected_kalman':frozen_corrected,
                    'control_source':result['speed_source'],'speed_position_source':args.sim_speed_position_source,
                    'capture_s':stamp,'track_identity':fit_identity}
                if args.sim_visual_speed_estimator=='constant_fit' and args.fixed_prediction_speed_m_s is None:
                    result['velocity_sample_count']=frozen_fit['sample_count']
                    result['velocity_span_s']=frozen_fit['span_s']
                    result['velocity_speed_std_m_s']=None
                elif args.sim_visual_speed_estimator=='corrected_kalman' and args.fixed_prediction_speed_m_s is None:
                    result['velocity_sample_count']=frozen_corrected['sample_count']
                    result['velocity_span_s']=frozen_corrected['span_s']
                    result['velocity_speed_std_m_s']=frozen_corrected['velocity_std_m_s']
                if args.early_visual_orientation:
                    result['early_visual_orientation']['observations'][-1].update(final_capture_s=stamp,final_identity=early_identity,final_frame_sequence=int(meta['sequence']),final_point_cloud_world_m=cloud.tolist(),final_observed_center_world_m=pos.tolist(),final_obb_quaternion_wxyz=quat.tolist(),final_velocity_world_m_s=velocity.tolist())
                if conditions is None and not env.grasp_region[0]<=ref.contact_xyz[0]<=env.grasp_region[1]:result['status']='predicted_contact_outside_grasp_region';break
                if conditions is None and args.profile=='smooth' and not env.grasp_region[0]<=ref.contact_xyz[0]+velocity[0]*ref.follow_s<=env.grasp_region[1]:result['status']='predicted_follow_outside_grasp_region';break
                references=[]
                if args.early_visual_orientation:
                    from experiments.two_second_grasp.early_orientation import final_orientation_candidates
                    early_record=result['early_visual_orientation']['observations'][-1]
                    orientation_candidates=final_orientation_candidates(cloud,early_record['planning']['candidate']['quaternion_wxyz'],early_record['quality']['near_square'])
                else:
                    orientation_candidates=[]
                    for axis in ('short','long'):
                        original=obb_top_down_grasp(cloud,closing_axis=axis)['tcp']
                        for flip in (0.,np.pi):
                            candidate=original.copy();candidate[:3,:3]=original[:3,:3] @ Rotation.from_euler('z',flip).as_matrix()
                            orientation_candidates.append({'pose':candidate,'source':axis})
                for candidate_entry in orientation_candidates:
                    candidate=candidate_entry['pose']
                    candidate_quat=Rotation.from_matrix(candidate[:3,:3]).as_quat()[[3,0,1,2]]
                    if args.synchronized_approach:
                        for join_s in join_candidates:
                            try:
                                candidate_timing=schedule_forward_join(capture_s=stamp,grasp_position=candidate[:3,3],velocity=velocity,
                                    standby_position=start_pose[:3],direction=[1.,0.,0.],now_s=now,
                                    planning_reserve_s=conditions.config.raw['planning_reserve_s'],
                                    advance_s=conditions.config.raw['command_advance_s'],join_s=join_s,
                                    synchronized_s=args.synchronized_duration_s,follow_s=args.follow_s,follow_bounds=env.grasp_region)
                            except ValueError:
                                continue
                            references.append(replace(ref,start_s=candidate_timing['join_start_s']-args.rotate_before_approach_s,
                                duration_s=args.rotate_before_approach_s+join_s+args.synchronized_duration_s,
                                prealign_s=args.rotate_before_approach_s+join_s,
                                capture_tcp_world=candidate[:3,3],target_quaternion_wxyz=candidate_quat))
                    else:
                        references.append(replace(ref,capture_tcp_world=candidate[:3,3],target_quaternion_wxyz=candidate_quat))
                if conditions is not None:
                    if not args.synchronized_approach:
                        references=[replace(r,duration_s=d+prealign_s) for d in conditions.config.raw['approach_duration_candidates_s'] for r in references]
                    references=[r for r in references if env.grasp_region[0]<=r.contact_xyz[0]<=env.grasp_region[1] and env.grasp_region[0]<=r.contact_xyz[0]+r.velocity_world[0]*r.follow_s<=env.grasp_region[1]]
                from experiments.two_second_grasp.orientation_audit import preferred_obb_axis
                if args.early_visual_orientation:
                    early_quat=np.asarray(early_record['planning']['candidate']['quaternion_wxyz'])
                    references.sort(key=lambda r:((min(range(len(join_candidates)),key=lambda i:abs(join_candidates[i]-(r.prealign_s-r.rotate_first_s))) if args.synchronized_approach else r.duration_s),
                        -abs(float(np.dot(early_quat,r.target_quaternion_wxyz))),preferred_obb_axis(r.target_quaternion_wxyz,pose)))
                    early_record['final_orientation_candidates']=[{'source':c['source'],'rotation_difference_deg':float(np.rad2deg(c['rotation_difference_rad']))} for c in orientation_candidates]
                else:
                    references.sort(key=lambda r:((min(range(len(join_candidates)),key=lambda i:abs(join_candidates[i]-(r.prealign_s-r.rotate_first_s))) if args.synchronized_approach else r.duration_s),preferred_obb_axis(r.target_quaternion_wxyz,pose),
                        -abs(float(np.dot(r.start_quaternion_wxyz,r.target_quaternion_wxyz)))))
                result['visible_obb_extent_m']=box['box']['extent_m']
                print('OBB尺寸 m：'+json.dumps(result['visible_obb_extent_m']),flush=True)
                future=pool.submit(plan_alternatives,env,references,start_q,np.asarray(env.data.qpos).copy(),dt,float(sensor._jaw_state['opening_width']),lambda:cancel[0],conditions=conditions,delivery_budget_s=max(0.,ref.start_s-now-.1),target_cloud_world=cloud.copy())
                speed_label='用户设定速度' if args.fixed_prediction_speed_m_s is not None else '视觉速度'
                print('抓取区上游已冻结'+speed_label+'和 OBB，并提前规划；预计入区',round(entry_s,3),'接触',round(ref.contact_s,3),flush=True)
            if result['status'] in ('synchronized_window_infeasible','predicted_contact_outside_grasp_region','predicted_follow_outside_grasp_region','early_target_identity_changed'):break
            if early_state=='planning' and early_future is not None and early_future.done():
                early_plan=early_future.result();early_future=None
                early_record=result['early_visual_orientation']['observations'][-1]
                early_record['planning']={k:v for k,v in early_plan.items() if k!='path'}
                if 'error' in early_plan:
                    result.update(status='early_orientation_planning_failed',error=early_plan['error']);break
                frozen_display.select_pose('early',early_identity,early_plan['candidate']['quaternion_wxyz'])
                early_trajectory=early_plan;early_state='ready';early_start_wait_s=now
            if early_state=='ready':
                early_start_q=early_trajectory['path'].sample(0.)[0]
                pending={'arm':early_start_q,'gripper':1.,'arm_time_parameterized':True};result['commands_sent']+=1
                early_fresh=(early_latest_identity==early_identity and early_latest_s is not None and 0<=now-early_latest_s<=.15)
                early_ready=conditions.arrival(('early_orientation_start',early_identity),early_start_q,early_start_wait_s,now,tolerance_rad=.005)
                if not early_fresh or not early_ready:continue
                early_record=result['early_visual_orientation']['observations'][-1]
                from experiments.two_second_grasp.smooth_execution import check_curve_geometry
                early_check=check_curve_geometry(env,early_trajectory['path'],np.asarray(env.data.qpos).copy(),lambda:cancel[0])
                early_record['start_geometry_recheck']=early_check
                if 'error' in early_check:
                    result.update(status='early_orientation_geometry_rejected',error=early_check['error']);break
                early_start_s=now+.1;early_state='executing'
                early_record['execution_start_s']=early_start_s
                print('提前转向轨迹已认证，开始转向；相机继续观测。',flush=True)
            if early_state=='executing':
                early_t=np.clip(now-early_start_s+args.sim_arm_command_advance_s,0.,early_trajectory['duration_s'])
                pending={'arm':early_trajectory['path'].sample(early_t)[0],'gripper':1.,'arm_time_parameterized':True};result['commands_sent']+=1
                if now>=early_start_s+early_trajectory['duration_s'] and conditions.arrival(('early_orientation',current_logical_id),early_trajectory['path'].q1,early_start_s+early_trajectory['duration_s'],now,tolerance_rad=.005):
                    start_q=early_trajectory['path'].q1.copy();start_pose=env.ik.forward_pose(start_q);early_finished_s=now
                    result['early_visual_orientation']['observations'][-1]['execution_finished_s']=now
                    if early_invalidated:
                        early_state='idle';early_identity=None;early_trajectory=None;early_cancel=None;early_invalidated=False
                        print('旧目标预转已停止在终点，重新等待新目标。',flush=True)
                    else:
                        early_state='complete'
                        print('提前转向完成，等待新一帧冻结位置、抓姿与视觉速度。',flush=True)
            if future is not None and future.done() and trajectory is None:
                planned=future.result();result.update({('candidate_ik_attempts' if k=='attempts' else k):v for k,v in planned.items() if k not in ('samples','stamps','reference','path')});future=None
                if 'error' in planned:result['status']='planning_failed';break
                if now+max(float(env.control_latency),.1 if args.profile=='smooth' else 0.)>planned['reference'].start_s:result['status']='planning_delivery_missed_start';result['delivery_sim_s']=now;break
                ref=planned['reference'];trajectory=planned;result['duration_s']=ref.duration_s;result['scheduled_contact_s']=ref.contact_s;result['status']='waiting_for_start';result['delivery_sim_s']=now;result['predicted_contact_tcp_world_m']=ref.contact_xyz;result['selected_target_quaternion_wxyz']=ref.target_quaternion_wxyz
                if args.early_visual_orientation:
                    early_record=result['early_visual_orientation']['observations'][-1]
                    early_quat=np.asarray(early_record['planning']['candidate']['quaternion_wxyz'])
                    early_rotation=Rotation.from_quat(early_quat[[1,2,3,0]])
                    final_rotation=Rotation.from_quat(ref.target_quaternion_wxyz[[1,2,3,0]])
                    early_record['selected_final_quaternion_wxyz']=ref.target_quaternion_wxyz.tolist()
                    early_record['early_vs_selected_final_rotation_deg']=float(np.rad2deg((early_rotation.inv()*final_rotation).magnitude()))
                if args.synchronized_approach:
                    result['selected_join_s']=ref.prealign_s-ref.rotate_first_s
                    result['scheduled_start_s']=ref.start_s
                    result['synchronized_schedule']={'join_start_s':ref.start_s+ref.rotate_first_s,
                        'sync_start_s':ref.start_s+ref.prealign_s,'contact_s':ref.contact_s,
                        'sync_start_coordinate_m':float(ref.sample(ref.prealign_s)[0][0]),
                        'contact_coordinate_m':float(ref.contact_xyz[0]),
                        'follow_end_coordinate_m':float(ref.contact_xyz[0]+ref.velocity_world[0]*ref.follow_s),
                        'speed_along_belt_m_s':float(ref.velocity_world[0])}
                from experiments.two_second_grasp.orientation_audit import capture_pose, orientation_error_deg
                pose=capture_pose(ref)
                frozen_display.select_pose('final',final_display_identity,ref.target_quaternion_wxyz)
                result['selected_candidate_index_in_sorted_list']=planned['attempts'][-1]['candidate']
                result['selected_vs_initial_obb_rotation_deg']=orientation_error_deg(ref.target_quaternion_wxyz,result['initial_obb_candidate_quaternion_wxyz'])
                path=[('approach' if t<=(ref.prealign_s if args.synchronized_approach else ref.duration_s) else ('follow' if t<=ref.duration_s+ref.follow_s else 'brake'),t,ref.sample(t)[0]) for t in np.linspace(0,ref.full_duration_s if args.profile=='smooth' else 2.,100)]
                print('平滑轨迹已就绪，等待运动开始。' if args.profile=='smooth' else '直线轨迹已就绪，等待运动开始。',flush=True)
            if trajectory is not None and phase=='intercept':
                if args.profile=='smooth' and env.has_severe_collision():result['status']='smooth_intercept_severe_collision';break
                elapsed=now-ref.start_s
                if elapsed>=(-args.sim_arm_command_advance_s if args.profile=='smooth' else 0.):
                    t=max(0.,elapsed+args.sim_arm_command_advance_s);idx=min(len(trajectory['samples'])-1,int(round(t/dt)))
                    measured=env._ee_pose().copy()
                    if args.profile=='smooth':
                        close_t=max(0.,elapsed+args.sim_gripper_command_advance_s)
                        eligible,gate=closure_gate(ref,elapsed,measured,previous_pose,previous_pose_s,now,command_elapsed=close_t)
                        if not closed:result['last_closure_gate']=gate
                        should_close=close_t>=ref.duration_s-1e-9 if scheduled_attempt else eligible
                        if should_close and not closed:closed=True;result['close_command_queued_sim_s']=now;result['close_gate']=gate;result['close_feedback_would_allow']=bool(eligible);print('到达计划接触时刻，开始闭合并跟随。' if scheduled_attempt else '位置、方向与相对速度通过，开始闭合并跟随。',flush=True)
                        if conditions is not None and closed and conditions.confirmation is None:conditions.begin_grip(now)
                        previous_pose=measured.copy();previous_pose_s=now
                        commanded=trajectory['path'].sample(t)[0]
                    else:
                        if elapsed>=2. and not closed:closed=True;result['close_command_queued_sim_s']=now;print('两秒到达，开始闭合并跟随。',flush=True)
                        commanded=trajectory['samples'][idx]
                    grip_command=-1. if closed else 1.
                    if closed and args.sim_close_ramp_s:
                        from experiments.two_second_grasp.pipeline import close_ramp_action
                        grip_command=close_ramp_action(now-result['close_command_queued_sim_s'],args.sim_close_ramp_s)
                    pending={'arm':commanded, 'gripper':grip_command,'arm_time_parameterized':args.profile=='smooth'};result['commands_sent']+=1
                    desired,desired_quat=ref.sample(max(0.,elapsed) if args.profile=='smooth' else min(elapsed,2.6))
                    from experiments.two_second_grasp.orientation_audit import orientation_error_deg
                    orientation_error=orientation_error_deg(measured[3:],desired_quat)
                    if closed and 'closure_orientation_audit' not in result:
                        result['closure_orientation_audit']={'sim_s':now,'actual_quaternion_wxyz':measured[3:].tolist(),
                            'reference_quaternion_wxyz':desired_quat.tolist(),'selected_target_quaternion_wxyz':ref.target_quaternion_wxyz.tolist(),
                            'actual_vs_reference_deg':orientation_error,
                            'actual_vs_selected_target_deg':orientation_error_deg(measured[3:],ref.target_quaternion_wxyz),
                            'comparison':'same simulation time; gripper command advance is independently configured',
                            'gripper_command_advance_s':args.sim_gripper_command_advance_s}
                    runtime=next((r for r in env._objects if r.logical_id==current_logical_id),None)
                    if closed and object_runtime is None and runtime is not None:object_runtime=runtime;object_z_at_close=float(env.data.qpos[runtime.qpos_adr+2])
                    forces=env._finger_object_contact_forces(runtime) if runtime is not None else (0.,0.)
                    jaw=env._gripper_observation()
                    rows.append({'sim_s':now,'elapsed_s':elapsed,'actual_tcp_world_m':measured[:3].tolist(),'reference_tcp_world_m':desired.tolist(),'actual_quaternion_wxyz':measured[3:].tolist(),'reference_quaternion_wxyz':desired_quat.tolist(),'orientation_error_deg':orientation_error,'error_m':float(np.linalg.norm(measured[:3]-desired)),'gripper_width_m':float(jaw['opening_width']),'motor_holding':bool(sensor._jaw_state.get('is_holding',False)),'left_force_n':float(forces[0]),'right_force_n':float(forces[1])})
                    follow_end=ref.duration_s+(ref.follow_s if args.profile=='smooth' else .6)
                    if args.profile=='smooth':
                        try:
                            held=(conditions.grip_feedback(sensor._jaw_state,now) if conditions is not None else closed and sensor._jaw_state.get('is_holding',False) and forces[0]>0 and forces[1]>0)
                        except PickRejected as error:
                            if not scheduled_attempt:raise
                            result['grip_retention_lost']={'sim_s':now,'reason':str(error)}
                            conditions.retention.held_width=None
                            if conditions.confirmation is not None:
                                conditions.confirmation.since=None;conditions.confirmation.confirmed=False
                            held=False
                        grip_since=(now if grip_since is None else grip_since) if held else None
                        if elapsed+args.sim_arm_command_advance_s>=follow_end and not brake_checked:
                            result['closure_feedback']={'motor_holding':bool(held),'left_force_n':float(forces[0]),'right_force_n':float(forces[1]),'continuous_confirmation_s':0. if grip_since is None else now-grip_since}
                            if not held or conditions is None and now-grip_since<GRIP_CONFIRMATION_DWELL_S-1e-9:
                                if not scheduled_attempt:result['status']='smooth_grip_not_confirmed_before_brake';break
                                result['brake_loaded_check']={'status':'not_performed_grip_unconfirmed'}
                            else:
                                brake_check=check_loaded_brake(env,trajectory['path'],ref,object_runtime,np.asarray(env.data.qpos).copy(),lambda:cancel[0]);result['brake_loaded_check']=brake_check
                                if 'error' in brake_check:result['status']='smooth_loaded_brake_rejected';break
                            brake_checked=True;print('执行已规划的连续减速段。',flush=True)
                        if (args.pipeline and brake_checked and held and object_runtime is not None
                                and prefetched_lift is None and prefetch_identity is None):
                            if result['brake_loaded_check'].get('status')=='not_performed_grip_unconfirmed':
                                result['brake_loaded_check']=check_loaded_brake(env,trajectory['path'],ref,object_runtime,np.asarray(env.data.qpos).copy(),lambda:cancel[0])
                            if 'error' in result['brake_loaded_check']:
                                result['status']='smooth_loaded_brake_rejected';break
                            from experiments.two_second_grasp.pipeline import predicted_loaded_snapshot
                            future_q=trajectory['path'].q1.copy()
                            future_qpos,future_pose=predicted_loaded_snapshot(env,np.asarray(env.data.qpos),object_runtime,future_q)
                            lift_goal=future_pose.copy()
                            lift_goal[2]+=args.sim_lift_m if args.sim_lift_m is not None else conditions.settings.lift_m
                            prefetch_identity=(current_logical_id,capture)
                            prefetched_lift=pool.submit(plan_smooth_leg,env,'lift',future_qpos,future_q,lift_goal,object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            result['pipeline_events'].append({'event':'lift_prefetch_submitted','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id})
                            print('减速期间并行准备抬升轨迹。',flush=True)
                        if brake_checked and not held and not scheduled_attempt:result['status']='smooth_brake_grip_lost';break
                    finish=ref.full_duration_s if args.profile=='smooth' else 2.6
                    ready=elapsed>=finish
                    if ready and conditions is not None:ready=conditions.arrival((result['completed_count'],'intercept'),trajectory['path'].q1,ref.start_s+finish,now)
                    if ready:
                        result['closure_feedback']={'motor_holding':bool(sensor._jaw_state.get('is_holding',False)), 'jaw_width_m':float(jaw['opening_width']), 'jaw_observer':sensor._jaw_state, 'physical_state':None if object_runtime is None else object_runtime.state,'left_force_n':float(forces[0]),'right_force_n':float(forces[1])}
                        if (not held if conditions is not None else not sensor._jaw_state.get('is_holding',False) or forces[0]<=0 or forces[1]<=0):
                            result['status']='attempt_completed_grip_unconfirmed' if scheduled_attempt else 'grasp_not_confirmed';break
                        phase='planning_lift';result['status']=phase;hold_q=(trajectory['path'].q1 if args.profile=='smooth' else np.asarray(env.get_joint_positions()).copy());actual_pose=env._ee_pose().copy()
                        goal=actual_pose.copy();goal[2]=(actual_pose[2]+args.sim_lift_m if args.sim_lift_m is not None else actual_pose[2]+conditions.settings.lift_m if conditions is not None else max(actual_pose[2]+.20,float(env.get_bin_drop_pose(args.bin_id)[2])+.08))
                        result['simulation_lift_override_m']=args.sim_lift_m
                        using_prefetched_lift=bool(prefetched_lift is not None and prefetch_identity==(current_logical_id,capture))
                        leg_future=(prefetched_lift if using_prefetched_lift else
                            pool.submit(plan_smooth_leg,env,'lift',np.asarray(env.data.qpos).copy(),hold_q,goal,object_runtime,dt,lambda:cancel[0],pose_solver=conditions) if args.profile=='smooth' else
                            pool.submit(plan_leg,env,np.asarray(env.data.qpos).copy(),hold_q,actual_pose,goal,2.,dt,lambda:cancel[0],runtime=object_runtime))
                        print('夹持确认，等待预规划抬升。' if using_prefetched_lift else '夹持确认，规划抬升。',flush=True)
            if phase in ('planning_lift','planning_transfer'):
                pending={'arm':hold_q,'gripper':-1.};result['commands_sent']+=1
                if leg_future is not None and leg_future.done():
                    planned=leg_future.result();leg_future=None
                    if phase=='planning_lift' and using_prefetched_lift:
                        from experiments.two_second_grasp.smooth_execution import check_curve_geometry
                        validation_started=time.perf_counter()
                        fresh=np.asarray(env.data.qpos).copy();fresh[env.arm_qpos_adrs]=hold_q
                        mismatch=float(np.max(abs(np.asarray(env.get_joint_positions())-hold_q)))
                        recheck=({'error':'start_feedback_changed'} if mismatch>.005 else
                            check_curve_geometry(env,planned['path'],fresh,lambda:cancel[0],runtime=object_runtime,allow_initial_belt_support=True) if 'error' not in planned else {'error':'prefetch_failed'})
                        result['pipeline_events'].append({'event':'lift_prefetch_recheck','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id,'result':recheck,'wall_duration_s':time.perf_counter()-validation_started})
                        using_prefetched_lift=False;prefetched_lift=None
                        if 'error' in recheck:
                            actual_pose=env._ee_pose().copy();goal=actual_pose.copy()
                            goal[2]+=args.sim_lift_m if args.sim_lift_m is not None else conditions.settings.lift_m
                            leg_future=pool.submit(plan_smooth_leg,env,'lift',np.asarray(env.data.qpos).copy(),hold_q,goal,object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            print('预规划抬升未通过实时复核，按当前状态重新规划。',flush=True)
                            continue
                        print('预规划抬升已按当前夹持状态复核。',flush=True)
                    if phase=='planning_transfer' and using_prefetched_transfer:
                        from experiments.two_second_grasp.pipeline import matching_drop_candidate
                        from experiments.two_second_grasp.smooth_execution import check_curve_geometry
                        validation_started=time.perf_counter()
                        mismatch=float(np.max(abs(np.asarray(env.get_joint_positions())-hold_q)))
                        recheck={'error':'prefetch_failed'} if 'error' in planned else None
                        if recheck is None and mismatch>.005:
                            recheck={'error':'start_feedback_changed'}
                        if recheck is None and args.bin_placement=='free_space':
                            selected=planned.get('bin_candidate_index')
                            old_goals=transfer_selection['candidate_tcp_poses_world']
                            fresh_goals=result['bin_space_selection']['candidate_tcp_poses_world']
                            match=(None if selected is None or selected>=len(old_goals) else
                                   matching_drop_candidate(old_goals[selected],fresh_goals))
                            if match is None:recheck={'error':'bin_candidate_changed'}
                            else:planned['bin_candidate_index']=match
                        if recheck is None:
                            recheck=check_curve_geometry(env,planned['path'],np.asarray(env.data.qpos).copy(),lambda:cancel[0],runtime=object_runtime)
                        result['pipeline_events'].append({'event':'transfer_prefetch_recheck','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id,'result':recheck,'wall_duration_s':time.perf_counter()-validation_started})
                        using_prefetched_transfer=False;prefetched_transfer=None;transfer_identity=None;transfer_selection=None
                        if 'error' in recheck:
                            qpos=np.asarray(env.data.qpos).copy()
                            if args.bin_placement=='free_space':
                                leg_future=pool.submit(plan_transfer_candidates,env,qpos,hold_q,result['bin_space_selection']['candidate_tcp_poses_world'],object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            else:
                                leg_future=pool.submit(plan_smooth_leg,env,'transfer',qpos,hold_q,env.get_bin_drop_pose(args.bin_id),object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            print('预规划搬运未通过实际状态复核，重新规划。',flush=True)
                            continue
                        print('预规划搬运已按抬升后的实际状态复核。',flush=True)
                    if phase=='planning_transfer' and args.bin_placement=='free_space' and 'bin_candidate_index' in planned:
                        chosen=planned['bin_candidate_index']; selection=result['bin_space_selection']
                        selection['selected_candidate_index']=chosen
                        selection['selected_tcp_pose_world']=selection['candidate_tcp_poses_world'][chosen]
                        selection['selected_payload_center_xy_m']=selection['candidate_payload_centers_xy_m'][chosen]
                        selection['selected_yaw_offset_deg']=selection['candidate_yaw_offset_deg'][chosen]
                        if chosen:
                            print('首选落点不可达，已验证备用落点：'+json.dumps(selection['selected_payload_center_xy_m']),flush=True)
                    result[phase+'_result']={k:v for k,v in planned.items() if k not in ('samples','times','path')}
                    if 'error' in planned:result['status']=phase+'_failed';break
                    actual_start=np.asarray(env.get_joint_positions()).copy()
                    start_error=actual_start-np.asarray(hold_q)
                    start_tolerance=.005 if conditions is not None else .05
                    if np.max(abs(start_error))>start_tolerance:
                        result['leg_start_feedback_mismatch']={'phase':phase,'actual_joints_rad':actual_start,
                            'planned_start_joints_rad':np.asarray(hold_q).copy(),
                            'joint_errors_rad':start_error,'max_abs_error_rad':float(np.max(abs(start_error))),
                            'tolerance_rad':start_tolerance,'sim_s':now}
                        result['status']=phase+'_start_feedback_changed';break
                    leg=planned;leg_start=now+(.1 if args.profile=='smooth' else 0.);phase='lift' if phase=='planning_lift' else 'transfer';result['status']=phase
                    print('执行阶段:',phase,flush=True)
                    if phase=='lift' and args.profile=='smooth' and conditions is not None:
                        from experiments.two_second_grasp.pipeline import predicted_loaded_snapshot
                        future_q=leg['path'].q1.copy()
                        future_qpos,_=predicted_loaded_snapshot(env,np.asarray(env.data.qpos),object_runtime,future_q)
                        try:
                            transfer_goal=env.get_bin_drop_pose(args.bin_id)
                            if args.bin_placement=='free_space':
                                from experiments.two_second_grasp.bin_space import select_drop_pose_candidates
                                _,transfer_selection=select_drop_pose_candidates(env,object_runtime,retained_placements,args.bin_id,transfer_goal)
                                prefetched_transfer=pool.submit(plan_transfer_candidates,env,future_qpos,future_q,transfer_selection['candidate_tcp_poses_world'],object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            else:
                                prefetched_transfer=pool.submit(plan_smooth_leg,env,'transfer',future_qpos,future_q,transfer_goal,object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                            transfer_identity=(current_logical_id,capture)
                            result['pipeline_events'].append({'event':'transfer_prefetch_submitted','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id})
                            print('抬升开始，并行规划抬升终点至料框的搬运路径。',flush=True)
                        except ValueError as error:
                            result['pipeline_events'].append({'event':'transfer_prefetch_unavailable','sim_s':now,'logical_id':current_logical_id,'reason':str(error)})
            elif phase in ('lift','transfer'):
                if env.has_severe_collision():result['status']=phase+'_severe_collision';break
                elapsed_leg=now-leg_start
                t=min(leg['duration_s'],max(0.,elapsed_leg+.1));idx=min(len(leg['times'])-1,int(round(t/leg['duration_s']*(len(leg['times'])-1))))
                pending={'arm':leg['path'].sample(t)[0] if args.profile=='smooth' else leg['samples'][idx],'gripper':-1.,'arm_time_parameterized':args.profile=='smooth'};result['commands_sent']+=1
                if object_runtime is not None:
                    objz=float(env.data.qpos[object_runtime.qpos_adr+2]);f=env._finger_object_contact_forces(object_runtime)
                    actual_tcp=env._ee_pose()
                    tcp_rotation=Rotation.from_quat(actual_tcp[3:][[1,2,3,0]]).as_matrix()
                    object_position=env.data.xpos[object_runtime.body_id].copy()
                    object_rotation=env.data.xmat[object_runtime.body_id].reshape(3,3)
                    effort=float(abs(np.asarray(sensor._jaw_state['motor_effort']).reshape(-1)[0]))
                    other_contacts=[]
                    object_geoms=set(object_runtime.geom_ids)
                    for contact_index,contact in enumerate(env.data.contact[:env.data.ncon]):
                        a,b=int(contact.geom1),int(contact.geom2)
                        if (a in object_geoms)==(b in object_geoms):continue
                        other=b if a in object_geoms else a
                        if other in env.finger_geom_ids:continue
                        contact_force=np.zeros(6)
                        mujoco.mj_contactForce(env.model,env.data,contact_index,contact_force)
                        if contact_force[0]>0:
                            other_contacts.append({'geometry':env.model.geom(other).name,
                                'normal_force_n':float(contact_force[0]),'distance_m':float(contact.dist)})
                    placement_rows.append({'sim_s':now,'phase':phase,'object_z_m':objz,
                        'left_force_n':float(f[0]),'right_force_n':float(f[1]),
                        'motor_holding':bool(sensor._jaw_state.get('is_holding',False)),
                        'gripper_width_m':float(sensor._jaw_state['opening_width']), 'motor_force_n':effort,
                        'confirmed_width_m':None if conditions is None else conditions.retention.held_width,
                        'actual_tcp_world_m':actual_tcp[:3].copy(),
                        'object_position_world_m':object_position,
                        'object_in_tcp_xyz_m':tcp_rotation.T@(object_position-actual_tcp[:3]),
                        'object_in_tcp_rotation_vector_rad':Rotation.from_matrix(tcp_rotation.T@object_rotation).as_rotvec(),
                        'nonfinger_object_contacts':other_contacts,
                        'object_pose_source':'mujoco_independent_diagnostic_only'})
                if not (conditions.grip_feedback(sensor._jaw_state,now) if conditions is not None else sensor._jaw_state.get('is_holding',False)):result['status']=phase+'_grasp_feedback_lost';break
                ready=elapsed_leg>=leg['duration_s']+(.0 if conditions is not None else .3)
                if ready and conditions is not None:
                    # Transfer planning requires its initial feedback within
                    # 0.005 rad of this endpoint. Wait for that same existing
                    # threshold before submitting a loaded transfer plan.
                    ready=conditions.arrival((result['completed_count'],phase),leg['path'].q1,
                        leg_start+leg['duration_s'],now,**({'tolerance_rad':.005} if phase=='lift' else {}))
                if ready:
                    if phase=='lift':
                        rise=objz-object_z_at_close if object_runtime is not None else 0.
                        result['lift_object_rise_m']=rise
                        if rise<.02 or f[0]<=0 or f[1]<=0:result['status']='physical_lift_not_verified';break
                        result['physical_grasp_verified']=True
                        phase='planning_transfer';result['status']=phase;hold_q=(leg['path'].q1 if args.profile=='smooth' else np.asarray(env.get_joint_positions()).copy());actual_pose=env._ee_pose().copy();goal=env.get_bin_drop_pose(args.bin_id)
                        if args.bin_placement=='free_space':
                            from experiments.two_second_grasp.bin_space import select_drop_pose_candidates
                            goal,result['bin_space_selection']=select_drop_pose_candidates(env,object_runtime,retained_placements,args.bin_id,goal)
                            print('同框空闲落点：'+json.dumps(result['bin_space_selection']['selected_payload_center_xy_m']),flush=True)
                        if conditions is None:goal[2]=actual_pose[2];goal[3:]=actual_pose[3:]  # historical experiment only
                        using_prefetched_transfer=bool(prefetched_transfer is not None and transfer_identity==(current_logical_id,capture))
                        if using_prefetched_transfer:
                            leg_future=prefetched_transfer
                        elif args.profile=='smooth' and args.bin_placement=='free_space':
                            leg_future=pool.submit(plan_transfer_candidates,env,np.asarray(env.data.qpos).copy(),hold_q,result['bin_space_selection']['candidate_tcp_poses_world'],object_runtime,dt,lambda:cancel[0],pose_solver=conditions)
                        else:
                            leg_future=(pool.submit(plan_smooth_leg,env,'transfer',np.asarray(env.data.qpos).copy(),hold_q,goal,object_runtime,dt,lambda:cancel[0],pose_solver=conditions) if args.profile=='smooth' else pool.submit(plan_joint_route,env,np.asarray(env.data.qpos).copy(),hold_q,goal,object_runtime,dt,lambda:cancel[0]))
                        print('实际抬升已验证，等待并复核预规划搬运。' if using_prefetched_transfer else '实际抬升已验证，搜索载物避障路径。',flush=True)
                    else:
                        phase='release';result['status']=phase;release_start=now;hold_q=(leg['path'].q1 if args.profile=='smooth' else np.asarray(env.get_joint_positions()).copy());print('到达料框上方，张开夹爪。',flush=True)
            elif phase=='release':
                # Early orientation cannot execute while the prior object is
                # being released/returned. Feed after return in this mode so
                # the next visual target still has an upstream approach.
                if args.pipeline and not args.early_visual_orientation and category_cursor < args.count and queued_next is None and np.isclose(trials[category_cursor]['belt_set_m_s'],env.belt_speed,rtol=0,atol=1e-12):
                    from nero_core.two_second.synchronized_reference import freeze_window_for_speed
                    from experiments.two_second_grasp.pipeline import feed_delay as calculate_feed_delay
                    freeze_distance=freeze_window_for_speed(conditions.config,float(env.belt_speed))['distance_m']
                    spawn_x=float(spawn_records[0]['spawn_position_world_m'][0])
                    feed_delay,travel_to_freeze=calculate_feed_delay(spawn_x,float(env.grasp_region[0])-freeze_distance,env.belt_speed,args.pipeline_feed_clearance_s)
                    if now-release_start>=feed_delay:
                        queued_id,queued_record=spawn_trial(env,trials[category_cursor])
                        queued_next=(queued_id,queued_record)
                        spawn_records.append(queued_record)
                        result['pipeline_events'].append({'event':'next_object_spawned_during_release','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':queued_id,'feed_delay_s':feed_delay,'estimated_clearance_s':args.pipeline_feed_clearance_s,'estimated_travel_to_freeze_s':travel_to_freeze})
                        print('投放期间按带速安排下一件，视觉继续观测。',flush=True)
                pending={'arm':hold_q,'gripper':1.};result['commands_sent']+=1
                if conditions is not None:
                    env.model.actuator_forcerange[conditions.gripper_actuator]=[-1.,1.]
                if object_runtime is not None:
                    pos=np.asarray(env.data.qpos[object_runtime.qpos_adr:object_runtime.qpos_adr+3]).copy();speed=float(np.linalg.norm(env.data.qvel[object_runtime.qvel_adr:object_runtime.qvel_adr+3]));f=env._finger_object_contact_forces(object_runtime)
                    placement_rows.append({'sim_s':now,'phase':phase,'object_position_world_m':pos,'object_speed_m_s':speed,'left_force_n':float(f[0]),'right_force_n':float(f[1]),'physical_checks':evaluate_placement(env,object_runtime,args.bin_id)})
                open_ready=(True if conditions is None else conditions.opening.observe(float(sensor._jaw_state['opening_width']),now))
                if conditions is not None and not open_ready and now-release_start>6.:result['status']='opening_feedback_not_reached';break
                if args.pipeline and open_ready and prefetched_return is None and return_identity is None:
                    return_identity=current_logical_id
                    prefetched_return=pool.submit(plan_smooth_leg,env,'return',np.asarray(env.data.qpos).copy(),hold_q,wait_q,None,dt,lambda:cancel[0],pose_solver=conditions,acceleration_override=args.sim_return_accel_limit)
                    result['pipeline_events'].append({'event':'return_prefetch_submitted','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id})
                    print('张爪后并行准备返回轨迹，继续确认工件稳定。',flush=True)
                if now-release_start>=2. and open_ready:
                    tail=[r for r in placement_rows if r['phase']=='release' and r['sim_s']>=now-.5]
                    required=('inside_xy','below_rim','above_floor','supported_by_bin')
                    stable=bool(len(tail)>=20 and tail[-1]['sim_s']-tail[0]['sim_s']>=.48 and all(all(r['physical_checks'][k] for k in required) and not r['physical_checks']['touching_finger'] and r['physical_checks']['linear_speed_m_s']<=.08 and r['physical_checks']['angular_speed_rad_s']<=1. for r in tail))
                    result['placement_verification']={'scope':'simulation_collision_bounds_bin_support_speed_no_finger_contact_0.5s', 'samples':len(tail),'stable':stable,'last_checks':tail[-1]['physical_checks'] if tail else None}
                    if not stable and now-release_start<6.:continue
                    if not stable:result['status']='placement_not_verified';break
                    result['completed_count']+=1
                    retained_placements.add(int(current_logical_id))
                    attempts.append({'index':result['completed_count'],'logical_id':int(current_logical_id),'category_id':int(category_sequence[category_cursor-1]),'track_id':target_id,'capture_s':result['capture_s'],'velocity_world_m_s':result['velocity_world_m_s'],'lift_object_rise_m':result.get('lift_object_rise_m'),'intercept_planning_wall_s':result.get('planning_wall_s'),'planning_lift_result':result.get('planning_lift_result'),'planning_transfer_result':result.get('planning_transfer_result'),'placement_verification':result['placement_verification'],'return_verified':False})
                    attempts[-1]['retention_diagnostics']=summarize_retention(placement_rows)
                    attempts[-1].update(spawn_record=next(s for s in spawn_records if s['logical_id']==current_logical_id),
                        frozen_speed_comparison=result.get('frozen_speed_comparison'),
                        initial_obb_candidate_quaternion_wxyz=result.get('initial_obb_candidate_quaternion_wxyz'),
                        selected_target_quaternion_wxyz=result.get('selected_target_quaternion_wxyz'),
                        selected_vs_initial_obb_rotation_deg=result.get('selected_vs_initial_obb_rotation_deg'),
                        selected_candidate_index_in_sorted_list=result.get('selected_candidate_index_in_sorted_list'),
                        closure_orientation_audit=result.get('closure_orientation_audit'),
                        bin_space_selection=result.get('bin_space_selection'),
                        observed_center_world_m=result.get('observed_center_world_m'),
                        predicted_contact_tcp_world_m=result.get('predicted_contact_tcp_world_m'),
                        selected_join_s=result.get('selected_join_s'),
                        synchronized_schedule=result.get('synchronized_schedule'),
                        synchronized_schedule_rejections=result.get('synchronized_schedule_rejections'),
                        candidate_ik_attempts=result.get('candidate_ik_attempts'))
                    phase='planning_return';result['status']=phase;hold_q=(leg['path'].q1 if args.profile=='smooth' else np.asarray(env.get_joint_positions()).copy())
                    using_prefetched_return=bool(prefetched_return is not None and return_identity==current_logical_id)
                    leg_future=(prefetched_return if using_prefetched_return else
                        pool.submit(plan_smooth_leg,env,'return',np.asarray(env.data.qpos).copy(),hold_q,wait_q,None,dt,lambda:cancel[0],pose_solver=conditions,acceleration_override=args.sim_return_accel_limit) if args.profile=='smooth' else
                        pool.submit(plan_return,env,np.asarray(env.data.qpos).copy(),hold_q,wait_q,dt,lambda:cancel[0]))
                    print('物料框稳定放置，准备返回等待位。',flush=True)
            if phase=='planning_return':
                pending={'arm':hold_q,'gripper':1.};result['commands_sent']+=1
                if leg_future is not None and leg_future.done():
                    returned=leg_future.result();leg_future=None;result['planning_return_result']={k:v for k,v in returned.items() if k not in ('samples','times','path')}
                    if using_prefetched_return:
                        from experiments.two_second_grasp.smooth_execution import check_curve_geometry
                        validation_started=time.perf_counter()
                        fresh=np.asarray(env.data.qpos).copy();fresh[env.arm_qpos_adrs]=hold_q
                        mismatch=float(np.max(abs(np.asarray(env.get_joint_positions())-hold_q)))
                        recheck=({'error':'start_feedback_changed'} if mismatch>.005 else
                            check_curve_geometry(env,returned['path'],fresh,lambda:cancel[0]) if 'error' not in returned else {'error':'prefetch_failed'})
                        result['pipeline_events'].append({'event':'return_prefetch_recheck','sim_s':now,'wall_s':time.perf_counter()-began_wall,'logical_id':current_logical_id,'result':recheck,'wall_duration_s':time.perf_counter()-validation_started})
                        using_prefetched_return=False;prefetched_return=None
                        if 'error' in recheck:
                            leg_future=pool.submit(plan_smooth_leg,env,'return',np.asarray(env.data.qpos).copy(),hold_q,wait_q,None,dt,lambda:cancel[0],pose_solver=conditions,acceleration_override=args.sim_return_accel_limit)
                            print('预规划返回未通过实时复核，按当前场景重新规划。',flush=True)
                            continue
                        print('预规划返回已按当前料框场景复核。',flush=True)
                    if 'error' in returned:result['status']='planning_return_failed';break
                    if np.max(abs(np.asarray(env.get_joint_positions())-hold_q))>(.005 if conditions is not None else .05):result['status']='return_start_feedback_changed';break
                    leg=returned;leg_start=now+(.1 if args.profile=='smooth' else 0.);phase='return';result['status']=phase;print('执行返回等待位。',flush=True)
            elif phase=='return':
                elapsed_return=now-leg_start;t=min(leg['duration_s'],max(0.,elapsed_return+.1));idx=min(len(leg['times'])-1,int(round(t/leg['duration_s']*(len(leg['times'])-1))))
                pending={'arm':leg['path'].sample(t)[0] if args.profile=='smooth' else leg['samples'][idx],'gripper':1.,'arm_time_parameterized':args.profile=='smooth'};result['commands_sent']+=1
                ready=elapsed_return>=leg['duration_s']+(.0 if conditions is not None else .5)
                if ready and conditions is not None:
                    # The following wait-pose verification uses .005 rad too.
                    ready=conditions.arrival((result['completed_count'],'return'),leg['path'].q1,
                        leg_start+leg['duration_s'],now,tolerance_rad=.005)
                if ready:
                    err=float(np.max(np.abs(np.asarray(env.get_joint_positions())-wait_q)));attempts[-1]['return_joint_error_rad']=err
                    if err>(.005 if conditions is not None else .03):result['status']='return_feedback_not_reached';break
                    attempts[-1]['return_verified']=True;attempts[-1]['returned_sim_s']=now;attempts[-1]['planning_return_result']=result.get('planning_return_result')
                    if result['completed_count']>=args.count:
                        final_checks=[]
                        for completed in attempts:
                            placed=next((r for r in env._objects if r.logical_id==completed['logical_id']),None)
                            check=None if placed is None else evaluate_placement(env,placed,args.bin_id)
                            final_checks.append({'logical_id':completed['logical_id'],'category_id':completed['category_id'],'checks':check})
                        result['final_bin_checks']=final_checks
                        required=('inside_xy','below_rim','above_floor','supported_by_bin')
                        all_present=all(x['checks'] is not None and all(x['checks'][k] for k in required) and not x['checks']['touching_finger'] and x['checks']['linear_speed_m_s']<=.08 and x['checks']['angular_speed_rad_s']<=1. for x in final_checks)
                        result['status']='all_placements_and_returns_completed' if all_present else 'final_bin_contents_not_verified'
                        break
                    print(f"第{category_cursor}件抓取、投放、返回已完成。",flush=True)
                    if queued_next is None:
                        current_logical_id,spawn_record=spawn_trial(env,trials[category_cursor]);spawn_records.append(spawn_record)
                    else:
                        current_logical_id,spawn_record=queued_next;queued_next=None
                    category_cursor+=1
                    print(f'第{category_cursor}件随机场景：'+json.dumps(serializable(spawn_record),ensure_ascii=False),flush=True)
                    if current_logical_id is None:result['status']='next_spawn_failed';break
                    start_q=(env.robot_controller._last_joint_target.copy() if args.profile=='smooth' else np.asarray(env.get_joint_positions()).copy());start_pose=(env.ik.forward_pose(start_q) if args.profile=='smooth' else env._ee_pose().copy())
                    ref=None;trajectory=None;future=None;phase='intercept';pending=None;closed=False;previous_pose=None;previous_pose_s=None;grip_since=None;brake_checked=False;target_id=None;object_runtime=None;object_z_at_close=None;pose=None;cloud=None;path=None;rows=[];placement_rows=[];prefetched_lift=None;prefetch_identity=None;using_prefetched_lift=False;prefetched_return=None;return_identity=None;using_prefetched_return=False;prefetched_transfer=None;transfer_identity=None;transfer_selection=None;using_prefetched_transfer=False;early_state='idle' if args.early_visual_orientation else 'disabled';early_future=None;early_trajectory=None;early_identity=None;early_latest_identity=None;early_latest_s=None;early_start_s=None;early_finished_s=None;early_start_wait_s=None;early_cancel=None;early_invalidated=False;early_quality.clear()
                    if conditions is not None:
                        conditions.confirmation=None;conditions.retention.held_width=None;conditions.retention.force_lost_since=None
                        from nero_core.two_second.feedback_gates import OpenConfirmation
                        conditions.opening=OpenConfirmation(conditions.settings.open_width_m)
                        env.model.actuator_forcerange[conditions.gripper_actuator]=[-conditions.settings.force_n,conditions.settings.force_n]
                    track_first_seen.clear()
                    frozen_display.clear()
                    # Per-object results must not survive into the next target.
                    for key in ('capture_s','observation_age_s','preparation_sim_s','predicted_entry_sim_s','track_id','observed_center_world_m','observed_obb_tcp_world_m','velocity_world_m_s','velocity_sample_count','visual_track_age_s','velocity_span_s','velocity_speed_std_m_s','scheduled_start_s','scheduled_contact_s','predicted_contact_tcp_world_m','planning_wall_s','max_joint_speed_rad_s','candidate_ik_attempts','delivery_sim_s','selected_target_quaternion_wxyz','close_command_queued_sim_s','closure_feedback','planning_lift_result','lift_object_rise_m','planning_transfer_result','placement_verification','planning_return_result','last_closure_gate','close_gate','brake_loaded_check','peak_joint_acceleration_rad_s2','geometry_scope','geometry_states','ik_and_qva_wall_s','geometry_wall_s'):
                        result.pop(key,None)
                    result['physical_grasp_verified']=False
                    for key in ('initial_obb_candidate_quaternion_wxyz','selected_vs_initial_obb_rotation_deg','selected_candidate_index_in_sorted_list','closure_orientation_audit','frozen_speed_comparison'):
                        result.pop(key,None)
                    result.pop('bin_space_selection',None);result.pop('visible_obb_extent_m',None)
                    result['status']='waiting_next_object';print(f"返回完成，生成第 {category_cursor} 件 T-LESS 工件。",flush=True)
            if viewer and time.monotonic()-lastdraw>.08:
                with viewer.lock():
                    viewer.user_scn.ngeom=0
                    if frozen_display.early is not None:
                        snapshot=frozen_display.early
                        draw_live_cloud(viewer.user_scn,snapshot['points'],snapshot['pose'],max_points=1400,
                            obb=snapshot['box'],clear=False,point_rgba=(.85,.2,1.,.9),
                            grasp_rgba=(1.,.35,.05,1.),reserve_geoms=min(1600,viewer.user_scn.maxgeom//2))
                    if frozen_display.final is not None:
                        snapshot=frozen_display.final
                        draw_live_cloud(viewer.user_scn,snapshot['points'],snapshot['pose'],max_points=1400,
                            obb=snapshot['box'],clear=False,point_rgba=(.05,.9,1.,.9),
                            grasp_rgba=(1.,.9,.05,1.),reserve_geoms=min(120,viewer.user_scn.maxgeom//4))
                    if path:draw_tcp_path(viewer.user_scn,path)
                    speed_lines=speed_hud_values(live_speed,now_s=now,
                        frozen_velocity=result.get('velocity_world_m_s'),belt_speed=env.belt_speed,truth_speed=truth_speed)
                    fit_line='--' if live_fit is None else f"{live_fit['velocity_world_m_s'][0]*100:+.2f} cm/s ({live_fit['sample_count']} frames)"
                    kalman_line='--' if live_corrected is None else f"{live_corrected['velocity_world_m_s'][0]*100:+.2f} cm/s"
                    viewer.set_texts((None,None,'Status\nVisual measurement\nFrozen prediction\nBelt setting\nObject truth (sim only)\nConstant visual fit\nCorrected Kalman\nPurple: early frozen\nCyan: grasp frozen',
                        f"{result['status']}\n"+'\n'.join([*speed_lines,fit_line,kalman_line,*frozen_display.labels()])))
                viewer.sync();lastdraw=time.monotonic()
            time.sleep(.002)
        if result['status']!='all_placements_and_returns_completed':
            if env.data.time>=args.seconds:result.update(status='simulation_time_budget_exhausted',interrupted_phase=phase)
            elif viewer is not None and not viewer.is_running():result.update(status='viewer_closed',interrupted_phase=phase)
    except KeyboardInterrupt:result.update(status='cancelled_by_keyboard_interrupt')
    except Exception as exc:result.update(status='error',error=f'{type(exc).__name__}: {exc}',details=getattr(exc,'details',{}),traceback=traceback.format_exc())
    finally:
        cancel[0]=True;pool.shutdown(wait=True,cancel_futures=True)
        if sim_servo is not None:
            result['simulation_inner_servo']=sim_servo.report();sim_servo.close()
        result['total_wall_s']=time.perf_counter()-began_wall;result['final_sim_s']=float(env.data.time);result['command_rows']=command_rows;result['action_queue_overruns']=env._action_queue_overruns
        result['rows']=rows;result['placement_rows']=placement_rows;result['evaluation_trace']=evaluation_trace
        result['live_visual_speed_samples']=live_speed_samples
        result['visual_position_samples']=visual_position_samples
        result['retention_diagnostics']=summarize_retention(placement_rows)
        result['evaluation_object_states']=[{'logical_id':r.logical_id,'category_id':r.category_id,'state':r.state,'position_world_m':env.data.xpos[r.body_id].copy()} for r in env._objects if r.logical_id is not None] if env is not None else []
        if result['status']=='waiting_next_object':result['status']='timed_out_waiting_next_visual_target'
        if rows:
            at_contact=min(rows,key=lambda x:abs(x['elapsed_s']-result['duration_s']))
            result.update(max_tracking_error_m=max(x['error_m'] for x in rows),contact_feedback=at_contact,last_feedback=rows[-1],bilateral_contact_seen=any(x['left_force_n']>0 and x['right_force_n']>0 for x in rows))
        if physics_trace is not None:
            env._physics_substep_observer=None
            result['physics_substep_trace']=physics_trace.report()
        log_started_wall=time.perf_counter()
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(serializable(result),ensure_ascii=False,indent=2)+'\n')
        compact=summarize(result);compact['detailed_log_save_wall_s']=time.perf_counter()-log_started_wall
        args.output.with_suffix('.summary.json').write_text(json.dumps(serializable(compact),ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(serializable(compact if args.profile=='smooth' else {k:v for k,v in result.items() if k not in ('rows','placement_rows','evaluation_trace','command_rows')}),ensure_ascii=False),flush=True)
        if viewer:
            with viewer.lock():
                speed_lines=speed_hud_values(live_speed,now_s=float(env.data.time),paused=True,
                    frozen_velocity=result.get('velocity_world_m_s'),belt_speed=env.belt_speed,truth_speed=truth_speed)
                viewer.set_texts((None,None,'Final status\nCompleted\nLast visual\nFrozen prediction\nBelt setting\nObject truth (sim only)\nPurple: early frozen\nCyan: grasp frozen',
                    f'{result["status"]}\n{result["completed_count"]}/{args.count}\n'+'\n'.join([*speed_lines,*frozen_display.labels()])))
            viewer.sync()
            if args.hold_viewer:
                while viewer.is_running():
                    viewer.sync()
                    if args.perception_view:cv2.waitKey(1)
                    time.sleep(.1)
            viewer.close()
        if args.perception_view:cv2.destroyWindow('MuJoCo RGB-D perception')
        env.close()
    return 0 if result['status']=='all_placements_and_returns_completed' else 1
if __name__=='__main__':raise SystemExit(main())
