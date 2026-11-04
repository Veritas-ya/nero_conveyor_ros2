# OBB 独立抓姿的只读仿真

`tools/view_obb_only.py` 使用 MuJoCo 渲染的单帧目标 RGB-D 点云，使用视觉跟踪的目标 ID 和卡尔曼速度就绪条件，不加载 GraspGen-X。通过带面约束 OBB 求可见点中心，选择向下接近、夹爪闭合方向沿 OBB 水平短边的一种姿态；TCP 置于该中心。OBB 轴有符号及朝向歧义，近正方形物体尤其容易出现跳变；物体被遮挡时，可见中心也不等于实体中心。橙色夹爪是方向示意而不是实际夹爪模型。

窗口颜色：青色为可见目标点云，绿色为 OBB，橙色为 OBB 计算出的 TCP 和方向。结果记录张开手指真实模型顶点到带面的最低间隙及现有阈值；此检查不等于 IK、整条轨迹或接触认证。无网络评分、无规划交付和无机械臂或夹爪命令。视觉驱动的 full_preplanned 物理执行链尚未接通，不能声称已抓起或投放。

```bash
cd /home/ubuntu22/nero_conveyor_ros2
PYTHONPATH=src/nero_core:src/nero_bringup MUJOCO_GL=egl \
python tools/view_obb_only.py --seconds 12 --hold-viewer --output runs/visualization/obb_only.json
```

本次运行使用真实 RGB-D 仿真观测和 0.1 m/s 仿真带速，非硬件；实际运行结果见 `runs/visualization/obb_only.json`。本次通过张开手指带面间隙检查，不代表全场景碰撞或物理抓取通过。

## OBB 接入 full_preplanned 规划诊断

使用同一捕获帧的可见 OBB，按短边优先、长边备选生成两个俯抓方向，并用原先夹爪几何、预抓取与拦截 IK、端点安全检查和 `full_preplanned` 完整规划。两种方向都使用 OBB 中心；夹爪的 180 度对称变体由现有筛选器处理。排序键 `0` 是共用筛选接口所需的占位值，禁用该来源的网络评分阈值；它不是神经网络置信度。通过候选保留 `visible_obb_geometry_only` 来源和方向记录。

```bash
PYTHONPATH=src/nero_core:src/nero_bringup MUJOCO_GL=egl \
python tools/probe_measured_network_scene.py --grasp-source obb --seconds 35 \
  --viewer --hold-viewer --output runs/visualization/obb_full_plan_live.json
```

该命令仍为影子规划：它没有向仿真执行器发送运动。成功记录表示规划器认证了包括条件抬升在内的轨迹，不等于已经确认夹持或物理抓起。当前 `NeroPolicyContext` 的策略时间以上下文创建时间为原点，而原有物理执行器按仿真绝对时间、实际反馈及其目标几何重新检查后采样；两条路径没有完成时间、场景监控和载荷/执行所有权的连接。不得直接将策略时间轨迹交给物理执行器。需要完成并测试该连接及夹持确认后再验证实际执行。

## 仿真执行边界的实验尝试

`--grasp-source obb --execute-sim` 在 MuJoCo 中尝试把已认证的 `full_preplanned` 参考交给执行边界。边界以 `application_sim_s - policy_clock_origin` 转换时钟，并检查原策略的轨迹归属、观测及场景复核、关节与夹爪几何和控制器重定时；错误时撤销待发队列。此运行循环是诊断，不是物理实时性验证。当前版本在 MuJoCo 实际应用指令时重新采样轨迹并使用新反馈检查；首次闭合通过原策略的闭合判定和夹爪几何检查后才提交。夹持确认及载物阶段已经接线，但尚未经过成功抓取的仿真验证。

在 `obb_execution_attempt4.json` 的一次运行中，轨迹认证通过，提交 14 个保持阶段命令后，场景监控尚未完成新轨迹的复核，按原逻辑拒绝继续执行（`scene_review_expired_or_missing_for_plan`）。`obb_execution_latest.json` 的另一轮在发送前发现跟随闭合窗口预测超出轨迹容差（`predicted_close_window_left_certificate_tolerance`），未发任何仿真命令。这些均不是夹持成功，也没有触碰真机。下一步需要缩短或并行完成绑定新轨迹后的场景复核、解决该时序与预测偏差，并在实际应用边界接入首闭合判定和持物反馈。

## 关闭在线场景复核的仿真对照

可以在上述仿真命令中加 `--disable-scene-review`；该选项必须与 `--grasp-source obb --execute-sim` 一起使用，仅作用于此诊断实例，默认不关闭。规划时仍基于捕获场景认证轨迹；关闭后不再检查捕获之后出现或移动的新障碍物，因此只适合受控仿真对照。目标身份、时钟、预测、执行反馈和既有轨迹判断没有关闭。当前闭合和载物阶段已接入应用时刻检查，但尚未在物理仿真中通过完整夹持和抬升验证，不会因这一选项自动变成完整抓取。

2026-10-28 两次无窗口对照中，规划都因抓取窗口耗尽而被拒绝，发送 0 条仿真命令，记录分别为 `runs/visualization/obb_no_scene_review_headless.json` 和 `runs/visualization/obb_no_scene_review_headless2.json`。可视窗口的一轮在规划完成前关闭，不能用来判断对照效果。


## 2026-10-28 实际执行进展

对相机观测使用 Nero7 卡尔曼测速，抓姿来源仅为 OBB，保留 100 ms 仿真发送参考提前并重新认证。为了隔离时间预算，在 0.05 m/s 和 0.03 m/s 仿真带速下试运行；默认 0.1 m/s 没有更改。修正了精确落在抓取区边界时浮点舍入造成的误拒绝，仅增加 1 nm 数值比较余量。0.05 m/s 一次执行 138 个应用时刻指令、关节累计最大变化 0.349 rad 后因反馈与参考的距离超出原有门限停止；另一次 5.8 s 接近尝试执行 267 个指令后因预测闭合位置超出证书容差停止。0.03 m/s 尝试执行 475 个指令后也因预测接触位置偏差停止。均没有发出闭合动作；仿真接触评分未检测到夹持或抬升。

当前 100 ms 前馈轨迹包含 `command_curves`，原有未来重规划明确不支持这类轨迹，不能简单删除条件：需要同时认证父轨迹与新轨迹名义及命令 q/v/a 连接，并处理版本、未来接点和逾期结果。同步仿真主循环的物理时钟在耗时计算期间不推进，这些不是实时性能结论。详细记录分别为 `obb_apply_boundary_attempt.json`、`obb_apply_boundary_slow.json` 和 `obb_apply_boundary_003.json`。

原配置 0.1 m/s 的执行尝试 `runs/visualization/obb_apply_boundary_01.json`：视觉估计速度约 0.09875 m/s，完整规划通过并应用了 123 条仿真指令，最大关节变化约 0.424 rad；之后因预测接触位置偏离认证轨迹停止。该次没有发送闭合指令，没有双指接触或物体抬升。成功规划的墙钟计算约 3.96 s；该同步诊断的仿真时钟会在计算期间停顿，不能将它解读为实时交付性能。

## 冻结初始视觉观测的仿真对照

`--freeze-visual` 是显式的 MuJoCo 执行对照：在构建第一个 full_preplanned 上下文后冻结最近一次合格的视觉位置和 Nero7 卡尔曼速度，按原始曝光时戳作匀速预测，不再把后续图像喂给目标跟踪、预测或在线场景复核，也不启用未来轨迹重规划。相机继续运行以显示点云，后续显示帧不能更新目标预测；物体真实状态只用于独立物理评分。实验为了执行冻结预测，明确延长观测年龄和预测有效期，绝不将旧观测伪装为新观测。真实机械臂入口没有使用此模式。

```bash
PYTHONPATH=src/nero_core:src/nero_bringup MUJOCO_GL=egl \
python tools/probe_measured_network_scene.py --grasp-source obb --execute-sim \
  --freeze-visual --compute-budget-s 2.7 --seconds 40 --viewer --hold-viewer \
  --output runs/visualization/obb_frozen_observation_budget27.json
```

一次 0.1 m/s 对照中，冻结曝光时间为仿真 8.300 s、初始视觉速度约 0.098741 m/s，随后 150 帧仅用于显示；控制器接收 172 次实际应用指令，执行到夹爪闭合，独立评分检测到双指接触。抬升之前，实际状态未满足已认证抬升起点要求（`nominal_loaded_lift_start_changed`），因此停止；物体最大抬升约 0.124 mm，未满足原有 20 mm、0.5 s 双指保持判据，不是成功抓取。初始预算 2 s 的独立尝试在规划结果晚于启动时刻后停止、未运动。两次都是同步 MuJoCo 实验，不能报告实机或实时成功。

冻结观测可视化：当前窗口用紫色球标识用于抓姿输入的观测中心，红色球标识冻结预测使用的最后视觉观测中心；左上角分开显示抓姿输入曝光、冻结位置曝光、速度采样时刻、世界 X 速度与红点世界坐标。两球是历史位置，物体继续沿传送带运动。速度由该时刻之前的连续视觉观测和卡尔曼滤波得到，不是凭红点单帧直接测速。只有冻结位置曝光与结果绑定记录时间一致才显示红点；速度采样时刻单独显示，不假定两者必然相同。
