"""Human-readable offline results; all numbers come from executed diagnostics."""
from __future__ import annotations

from pathlib import Path

from .sim_diagnostic import write_document, write_rows


def _number(value, digits=4):
    return '—' if value is None else f'{value:.{digits}f}'


def write_report(output, result, config):
    output = Path(output)
    cases = result.get('phase_c', {}).get('cases')
    if not cases and result.get('phase_b', {}).get('executed'):
        cases = {'C0_rigid': {group: result['phase_b'].get(group)
                              for group in ('real_replay', 'synthetic_control')}}
    rows = []
    for name, groups in (cases or {}).items():
        for group in ('real_replay', 'synthetic_control'):
            item = groups.get(group)
            if item is None:
                continue
            cal = item['calibration']
            external = cal['external_error'] or {}
            def maximum(key):
                return (item.get(key) or {}).get('max')
            rows.append({'case': name, 'group': group,
                         'samples_total': item['samples_total'], 'detected': item['detected_count'],
                         'valid_samples': item['valid_calibration_samples'],
                         'pair_count': item['pair_count'], 'valid_pair_count': item['valid_pair_count'],
                         'detection_success_rate': item['detection_success_rate'],
                         'detection_failed_count': item['detection_failed_count'],
                         'out_of_view_count': item['out_of_view_count'],
                         'occluded_frame_count': item['occluded_frame_count'],
                         'occlusion_unknown_frame_count': item['occlusion_unknown_frame_count'],
                         'calibration_status': cal['status'],
                         'production_quality_accepted': cal.get('solution', {}).get('accepted'),
                         'pnp_rotation_max_deg': maximum('pnp_rotation_error_deg'),
                         'pnp_translation_max_mm': maximum('pnp_translation_error_mm'),
                         'reprojection_max_px': maximum('reprojection_rms_px'),
                         'relative_angle_difference_max_deg': maximum('relative_angle_difference_deg'),
                         'relative_angle_difference_rms_deg': (item.get('relative_angle_difference_deg') or {}).get('rms'),
                         'external_rotation_error_deg': external.get('rotation_error_deg'),
                         'external_translation_error_mm': external.get('translation_error_mm'),
                         'held_out_max_mm': maximum('held_out_position_error_mm'),
                         'actual_mount_min_deg': min(item['actual_mount_angles_deg']),
                         'actual_mount_max_deg': max(item['actual_mount_angles_deg'])})
    write_rows(output/'comparison.csv', rows)

    real = result.get('historical_real_pairs', [])
    comparisons = []
    for name, groups in (cases or {}).items():
        simulated = {(p['from_id'], p['to_id']): p for p in (groups.get('real_replay') or {}).get('pairs', [])}
        for pair in real:
            other = simulated.get((pair['from_id'], pair['to_id']), {})
            comparisons.append({'case': name, 'from_id': pair['from_id'], 'to_id': pair['to_id'],
                                'real_robot_rotation_deg': pair['robot_rotation_deg'],
                                'real_vision_rotation_deg': pair['vision_rotation_deg'],
                                'real_robot_minus_vision_deg': pair['robot_minus_vision_deg'],
                                'sim_status': other.get('status', 'not_available'),
                                'sim_robot_rotation_deg': other.get('robot_rotation_deg'),
                                'sim_vision_rotation_deg': other.get('vision_rotation_deg'),
                                'sim_robot_minus_vision_deg': other.get('robot_minus_vision_deg')})
    write_rows(output/'real_vs_sim_pairs.csv', comparisons)

    a = result['phase_a']
    a_error = a['calibration']['external_error'] or {}
    lines = ['# 眼在手外标定仿真诊断实跑报告', '',
             f"阶段 A：{'通过' if a['passed'] else '失败'}，{a['samples']} 组非单位 X/Y 的纯位姿数据。"
             f"外参旋转误差 {a_error.get('rotation_error_deg', float('nan')):.3g}°，"
             f"平移误差 {a_error.get('translation_error_mm', float('nan')):.3g} mm，"
             f"配对相对转角最大差 {a['relative_angle_difference_max_deg']:.3g}°。这只检查数学与接口约定，不证明真机精度。", '',
             '变换统一为 `F=T_base_flange`、`C=T_camera_board`、`X=T_base_camera`、`Y=T_flange_board`，'
             '`F Y = X C`。反馈转为米/弧度，`Rz(yaw) Ry(pitch) Rx(roll)`。所有旋转误差均取旋转矩阵的相对旋转角。', '',
             f"沿用生产求解器 `{config['solver']}`；没有为仿真更换算法或降低最小样本数。"]
    if result.get('phase_b', {}).get('executed'):
        b = result['phase_b']
        lines += ['', f"阶段 B：{'通过' if b['passed'] else '未通过'}。图像实际经过已有角点检测→ITERATIVE PnP→Park 手眼求解；"
                  '真值只生成图像并评价，未替代 PnP 或选择其解。独立真值验收阈值见 config.json，生产验收参数不变。', '',
                  '棋盘为最新 session 的 **11×8 内角点、20 mm 单格**，方格区域240×180 mm，内角点跨度200×140 mm。'
                  '使用现有 NERO robot.xml，反馈法兰为离线 SDK/FK 已核对的 **link7**，不是下游 gripper_flange。', '',
                  '仿真只沿用真实 K 的 fx/fy；实际渲染 K 的中心为(639.5,359.5)，1280×720，D=0。'
                  '没有裁剪、缩放或去畸变，也没有把真实非零 D 套在无畸变渲染上。'
                  'MuJoCo相机→OpenCV采用 diag(1,-1,-1)。'
                  f"投影自检最大像素差 **{b['camera_projection']['maximum_pixel_error']:.4f} px**，详见 camera_projection_check.json。", '',
                  '相机在全部 B/C 实验中始终固定。每张图像与法兰反馈都来自同一个 mj_forward 状态。'
                  '红色原点标记仅根据图像确定普通棋盘的行列编号，模拟人工核对同一个实物角点，不访问棋盘真值。', '',
                  '最新真实测试的6组七关节 rad 值逐项重放，共15个相同编号起终点配对。'
                  '原数据中部分关节超出现有 XML 限位，本诊断原样保留并逐帧记录；不夹紧、不驱动、不做动力学积分。'
                  '6张不足真实求解器要求的8张，回放组外参标为 not_solved。另用24个明确标为 synthetic_control 的姿态求外参，不能称为原真实实验复现。']
        if real:
            worst = max(real, key=lambda p: p['abs_angle_difference_deg'])
            baseline = next((p for p in (b.get('real_replay') or {}).get('pairs', [])
                             if p['from_id'] == worst['from_id'] and p['to_id'] == worst['to_id']), {})
            lines += ['', f"真实记录最大差配对 {worst['from_id']}→{worst['to_id']}：机器人 "
                      f"{worst['robot_rotation_deg']:.4f}°，视觉 {worst['vision_rotation_deg']:.4f}°，"
                      f"绝对差 {worst['abs_angle_difference_deg']:.4f}°。"
                      f"相同关节起终点在理想刚性仿真的差为 {_number(baseline.get('abs_angle_difference_deg'))}°。"
                  '相机位置及安装 Y 为人工设置的仿真场景，没有实测真实 X/Y，不能称为完整真机场景复现。']

        lines += ['', '24姿态控制组实跑结果：', '',
                  '| 安装条件 | 检测/有效样本 | 配对转角最大差(°) | 外参旋转误差(°) | 外参平移误差(mm) | 独立测试点最大误差(mm) | 生产验收 |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for row in rows:
            if row['group'] == 'synthetic_control':
                lines.append(f"| {row['case']} | {row['detected']}/{row['valid_samples']} | "
                             f"{_number(row['relative_angle_difference_max_deg'])} | {_number(row['external_rotation_error_deg'])} | "
                             f"{_number(row['external_translation_error_mm'])} | {_number(row['held_out_max_mm'])} | "
                             f"{'接受' if row['production_quality_accepted'] else '不接受'} |")
        lines += ['', '表中的“生产验收”检查拟合后固定安装Y的内部一致性残差，不是相对X_gt或独立测试点的绝对误差。'
                  '因此通过生产5 mm内部阈值，不等于外参真值平移误差小于5 mm；C2的小幅扰动可能被外参偏差部分吸收，仍通过内部验收。']
        lines += ['', '原6组关节配对回放：', '',
                  '| 安装条件 | 检测/总数 | 配对转角最大差(°) | 每帧PnP旋转最大误差(°) | 每帧PnP平移最大误差(mm) |',
                  '|---|---:|---:|---:|---:|']
        for row in rows:
            if row['group'] == 'real_replay':
                lines.append(f"| {row['case']} | {row['detected']}/{row['samples_total']} | "
                             f"{_number(row['relative_angle_difference_max_deg'])} | {_number(row['pnp_rotation_max_deg'])} | "
                             f"{_number(row['pnp_translation_max_mm'])} |")
        lines += ['', '每帧PnP旋转/平移误差、像素重投影RMS在各组 per_frame.csv；配对角度在 pairs.csv；'
                  '最终外参和安装残差在 result.json；8个未参与拟合的工作区点误差在 held_out_points.csv。'
                  '它们是不同指标，不能互相替代。测试点以X_gt生成准确的相机系三维坐标，只评价外参变换误差，不包含深度测量误差。'
                  '未检测和出视野帧保留原图、原因和记录，不静默删除。遮挡由与实际渲染相同的可见几何组射线检查，未知值明确记录，未用于PnP。']
        failures = [r for r in rows if r['detection_failed_count']]
        for row in failures:
            lines += ['', f"{row['case']}/{row['group']}：{row['detection_failed_count']}张检测失败，"
                      f"有效配对{row['valid_pair_count']}/{row['pair_count']}；"
                      f"出视野{row['out_of_view_count']}张，几何遮挡{row['occluded_frame_count']}张，"
                      f"遮挡未知{row['occlusion_unknown_frame_count']}张。失败不能自动归因为遮挡。"]
    c = result.get('phase_c', {})
    if c.get('executed'):
        axis = config['mount_cases']['flex_axis_flange']
        lines += ['', f"阶段 C 已实际运行。C0直接使用B的原样本；C1整批固定装歪{config['mount_cases']['constant_tilt_deg']:g}°，仍是恒定Y的刚性安装。"
                  f"C2采用可复现等效扰动：绕**法兰坐标系轴{axis}**，以名义棋盘内角点矩形中心为转动中心，"
                  f"`theta = amplitude * dot(R_base_flange[:,0], [0,0,-1])`，幅度设置为{config['mount_cases']['flex_amplitudes_deg']}°。"
                  '每帧真实施加角度、转轴、中心保存在 samples.json，实际角度不一定达到设定幅度。'
                  '这些设置是敏感性实验，不是测量到的支架变形；注入3°不保证出现3°配对差。', '',
                  '固定装歪允许被恒定Y吸收，不能据此解释随姿态产生的转角不一致。'
                  '随姿态改变安装关系会破坏刚性约束；若能产生类似量级，只说明该机制可能解释现象，不能确认真机根因。']
        largest = max((row for row in rows if row['group'] == 'real_replay' and row['case'].startswith('C2_')),
                      key=lambda row: row['relative_angle_difference_max_deg'] or 0, default=None)
        if largest and real:
            real_max = max(p['abs_angle_difference_deg'] for p in real)
            lines += ['', f"本轮真实关节回放中，{largest['case']}的实际安装角范围"
                      f"[{largest['actual_mount_min_deg']:.4f}, {largest['actual_mount_max_deg']:.4f}]°，"
                      f"最大配对转角差{largest['relative_angle_difference_max_deg']:.4f}°；"
                      f"原真实记录最大差为{real_max:.4f}°。当前轴与姿态依赖规律尚未完整复现这项真机差异，未拟合扰动规律追求数值吻合。"]
            worst = max(real, key=lambda pair: pair['abs_angle_difference_deg'])
            same_pair = next((pair for pair in c['cases'][largest['case']]['real_replay']['pairs']
                              if pair['from_id'] == worst['from_id'] and pair['to_id'] == worst['to_id']), None)
            if same_pair and same_pair['status'] == 'ok':
                lines += ['', f"进一步核对相同起终点：{worst['from_id']}→{worst['to_id']}在"
                          f"{largest['case']}下差{same_pair['abs_angle_difference_deg']:.4f}°，"
                          f"原记录差{worst['abs_angle_difference_deg']:.4f}°；不能用另一个配对的最大值代替这个配对的复现。"]
    else:
        lines += ['', f"阶段 C 尚未运行：{c.get('reason', '本次只请求阶段A')}。"]
    if result.get('error'):
        lines += ['', f"本次异常：{result['error']}。未完成阶段不标为通过。"]
    lines += ['', 'A/B通过仅说明所测试理想条件下当前流程成立，真实机器人反馈精度、时间同步、实际支架变化仍未由仿真测量。'
              '全部仿真外参仅用于诊断，**不能直接替代真实相机外参**。未连接真实CAN/相机，未修改固件、生产配置、原始数据或真实标定。', '',
              '交付：config.json与源YAML、summary.json、comparison.csv、real_vs_sim_pairs.csv；'
              '各组images/corners/axes图像；sampling_demo.mp4为实际渲染采样序列，非连续动力学运动。'
              'sources.json记录输入文件哈希及本次运行后完整性。', '',
              '参考：[MuJoCo手眼项目](https://github.com/kejianhong/Hand_Eye_Calibration_in_MuJoCo)，'
              '[ROS2仿真项目](https://github.com/shengyangzhuang/handeye_calibration_ros2_sim)，'
              '[OpenCV官方手眼测试](https://github.com/opencv/opencv/blob/4.x/modules/calib3d/test/test_calibration_hand_eye.cpp)。'
              '参考工程的机器人、ChArUco/ArUco和眼在手上场景与本项目不同，本次复用了本地NERO和真实普通棋盘标定接口。', '']
    (output/'report_zh.md').write_text('\n'.join(lines), encoding='utf-8')
    write_document(output/'completion.json', {'phase_a_executed': a['executed'], 'phase_a_passed': a['passed'],
                                            'phase_b_executed': result.get('phase_b', {}).get('executed', False),
                                            'phase_b_passed': result.get('phase_b', {}).get('passed', False),
                                            'phase_c_executed': c.get('executed', False),
                                            'uncompleted': [name for name in ('phase_a', 'phase_b', 'phase_c')
                                                            if not result.get(name, {}).get('executed', False)],
                                            'real_robot_accuracy_proven': False,
                                            'real_support_deformation_measured': False,
                                            'simulation_extrinsics_deployable_to_real_camera': False})
    return rows
