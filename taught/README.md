# 示教轨迹归档（2026-11-06）

当前配置使用以下两段，单位为弧度、秒，关节顺序为 `joint1`～`joint7`。轨迹 JSON 内包含 TCP、机器人模型、料框、CAN通道及固件来源，加载时逐项校验。

| 用途 | 仓库内文件 | 样本数 | 文件中的相对时长 |
| --- | --- | --- | --- |
| 当前投放 | [transfer_trimmed_start025_20261105.json](session_20261104_223416/transfer_trimmed_start025_20261105.json) | 42 | 2.049927473 s |
| 当前返回 | [return_generated.json](session_20261104_223416/planned_return/return_generated.json) | 61 | 1.2 s |

投放段来自 [transfer_retake_20261104_223900.json](session_20261104_223416/transfer_retake_20261104_223900.json) 的第25点起，后续关节采样和采样间隔保持。原始66点、3.244850874秒录制及另外两次录制保存在同一目录。返回段由离线规划生成，不能标为人工拖动录制；生成报告和曲线采样一并归档于 `planned_return/`。

当前投放使用 `path_preserving_variable_speed` 时间参数化，软件时长上限1.5秒；保留8 rad/s²载物加速度边界。文件中的录制相对时间不会被改写，执行时长由时间参数化和完整曲线检查决定。1.5秒约束仅针对示教投放段，抬升、接入、张爪与返回另行计时。

`prepared_taught_cycle_enabled` 使抬升、接入、示教投放及返回在执行搬运前准备，实际张爪后仍检查返回路径。返回轨迹记录的是生成时的等待位；当前等待位若有更新，执行器会连接到当前配置等待位，不能将历史返回终点当作当前配置。

`active_paths.json` 和 `session.json` 保留现场选择记录，其中绝对路径和诊断目录属于原机器。当前可迁移文件清单及哈希见 [trajectory_manifest_20261106.json](trajectory_manifest_20261106.json)，所有轨迹与返回曲线的字节校验见 [SHA256SUMS](SHA256SUMS)：

```bash
cd /path/to/nero_conveyor_ros2/taught
sha256sum -c SHA256SUMS
```

在现有现场目录启动时，YAML已经指定这对轨迹；换目录克隆后，用以下显式路径先检查，覆盖原机器的绝对路径：

```bash
cd /path/to/nero_conveyor_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
PICK_ROOT="$PWD"

OPENBLAS_NUM_THREADS=1 ros2 run nero_core fixed_freeze_pick \
  --config "$PICK_ROOT/install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml" \
  --taught-transfer "$PICK_ROOT/taught/session_20261104_223416/transfer_trimmed_start025_20261105.json" \
  --taught-return "$PICK_ROOT/taught/session_20261104_223416/planned_return/return_generated.json" \
  --belt-speed-cm-s 53 \
  --check-config
```

该检查不打开相机或CAN，不发送动作。`fixed_freeze_pick` 与原 `two_second_pick` 使用相同示教执行器；原入口继续保留。现场启动参数见[功能和启动说明](../docs/功能归档与固定冻结入口_20261106.md)。此处归档的轨迹、模型时间参数化和离线验证不能代替当前实测场景下的路径认证或实机抓放结果。
