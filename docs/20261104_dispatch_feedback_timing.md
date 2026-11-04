# 发送检查窗口与运动反馈时间诊断

本次不连接设备、不发送控制指令。最近一次事件来自
`runs/hardware_continuous_20261104_175019`：抓取阶段实际发送时刻超过已检查窗口
2.649 ms；随后恢复运动时，关节 FK 与法兰反馈的位置差为 5.93 mm、角度差为
0.90°。这两个事件发生在不同阶段，不能认定后一事件是前一事件的原因。

发送侧现在只对检查窗口过期进行有限重试：丢弃旧指令，重新读取设备反馈，重新执行
身份、时钟、夹持和几何检查，再按发送时刻采样。发送调用边界还会重新检查窗口。
连续三次过期仍停止；其他异常不自动重试。`dispatch_window_retry` 记录每次重试。
身份检查、5 mm / 1° 的 FK 检查与几何检查未因此放宽。离线测试覆盖首次窗口超过
2.649 ms、校验过程本身导致过期、连续过期不发送和正常完成。2.649 ms 为日志中的
窗口超出量，不等于已知的完整计算耗时。

关节与法兰的 CAN 帧没有设备采样时间戳；现有配对按主机接收时间将各轴实测值
插值到法兰包接收时刻，不外推，并检查时钟代次。对六段历史日志以相同的运动帧比较
不同接收时间偏移，位置误差中位数均在约 -2 ms 处较小，详见
`runs/fk_timing_diagnostic_20261104/report.json` 和
`tools/analyze_fk_receive_timing.py`。这些日志保存的是稀疏控制快照，不是完整
CAN 流；插值结果既不能证明采样时差确为 2 ms，也无法重现最后一个异常包。静止时
误差很小、运动时增大支持时序问题的可能性，但不能排除其他原因。

根据用户要求，真机配置已采用 `hardware.joint_receive_offset_s: -0.002`：关节
按法兰主机接收时间减去 2 ms 的位置进行实测插值；法兰原始时间不变，未找到实测
前后包则等待，不外推。用于规划和运动的最新关节反馈不变，FK 验证的 5 mm / 1°
阈值不变。旧配置不含此项时使用 0；改回 0 可取消补偿。

实时快照同时保存原始关节接收历史上 -6、-4、-2、0、+2 ms 的诊断配对，以及每轴
插值包围区间，写入执行 trace 的 `feedback_pose_joint_pairing_json`。主配对记录
原始法兰时刻、实际关节插值时刻和配置偏移。此 2 ms 是历史接收时刻的经验修正，不
是厂家提供的设备采样时间，仍应在静止及不同速度、方向的实机数据中验证异常帧和
跨运行稳定性。若厂家能提供设备采样时间，优先使用它；不能把离线拟合当作实机验证。

只读采集工具如下；工具自身不会使能、移动或发送停止指令，机械臂的运动须由现场
独立控制。它沿用设备适配器的状态读取条件；若控制器处于不支持的状态会退出并
保存已取得的数据。

```bash
cd /home/ubuntu22/nero_conveyor_ros2
source /opt/ros/humble/setup.bash
source install/setup.bash
OPENBLAS_NUM_THREADS=1 python3 tools/record_fk_receive_feedback.py \
  --config install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml \
  --seconds 20 --hz 50 \
  --output "runs/fk_feedback_$(date +%Y%m%d_%H%M%S).npz"
```
