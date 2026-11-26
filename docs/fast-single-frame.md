# Fast 单帧连续抓取

`fixed_freeze_pick --fast-stereo` 使用现场配置中的 `fast_cloud_delivery_mode: direct_to_planner`。原生检测先选定工件像素。Fast 在同一曝光的这些像素上重建深度，所有有限正深度点共用一个点云编号，直接用于计算可见包围盒（OBB）和检查机械臂与运动工件的碰撞。生成抓姿需要至少 21 个有效点，且这些点在带面上的投影能围出面积。[现场配置](../src/nero_bringup/config/two_second.hardware.tracked_freeze.yaml)

现场单帧模式按曝光位置选择抓取区入口上游 45～75 cm 范围内的工件。结果发布时工件会继续随带移动。事件 `fast_cloud_delivered` 记录点云交付，`target_frozen` 记录正式冻结及当时的预计位置。轨迹通过关节、闭爪高度和碰撞检查后，执行端记录 `planning_delivery`。

## 从工作区根目录启动

使用 ROS 2 Humble 构建。Fast 后端会从 `~/.cache/nero-stereo/fast` 加载独立 Python 环境 `env`、Fast-FoundationStereo 源码 `repo` 和权重 `research_weights/23-36-37/model_best_bp2_serialize.pth`。

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-up-to nero_core nero_bringup
source install/setup.bash
```

在 can0 已启动且机械臂已使能后运行。将 `BELT_SPEED_CM_S` 设为现场测得的带速，单位 cm/s。命令显式引用仓库里的两段示教文件，可以从不同的工作区路径启动。

```bash
CFG_DIR="$PWD/install/nero_bringup/share/nero_bringup/config"
TAUGHT_DIR="$PWD/taught/session_20261104_223416"
BELT_SPEED_CM_S=30.5

OPENBLAS_NUM_THREADS=1 ros2 run nero_core fixed_freeze_pick \
  --config "$CFG_DIR/two_second.hardware.tracked_freeze.yaml" \
  --experimental-direct-drop-guard "$CFG_DIR/calibration/bin_rim_nominal_20261114.yaml" \
  --taught-transfer "$TAUGHT_DIR/transfer_trimmed_start025_20261105.json" \
  --taught-return "$TAUGHT_DIR/planned_return/return_generated.json" \
  --belt-speed-cm-s "$BELT_SPEED_CM_S" \
  --fast-stereo --object-relative-approach \
  --no-depth-separation --no-rgb-cloud-completion \
  --native-observation-shadow \
  --continuous --execute --preview --record-rgbd \
  --output "$PWD/runs/hardware_fast_single_frame_$(date +%Y%m%d_%H%M%S)"
```

将 `--execute` 换为 `--check-config` 可在连接设备前确认所选策略、标定及示教轨迹。点云和轨迹可分别按点云编号、曝光帧和工件 ID 在本次运行的 `events.jsonl` 和 `experiment/` 中核对。
