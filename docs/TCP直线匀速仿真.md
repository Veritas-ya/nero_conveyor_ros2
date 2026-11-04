# TCP 直线匀速仿真

这是独立的 MuJoCo 末端运动实验。仿真重置时，使用 IK 将 TCP 放在抓取区中线上方约 20 cm 的等待姿态；夹爪保持向下。实验默认沿世界 X 正方向走 0.12 m，目标匀速 0.10 m/s，开始和结束各用 0.30 s 加减速。传送带停止，不生成工件，不使用抓取辅助或接入设备。

参考轨迹在笛卡尔空间生成，按仿真控制周期逐点求解保持夹爪方向的 IK，并检查采样关节状态、相邻连接的仿真静态碰撞及命令范围。之后逐周期向 MuJoCo 的关节位置执行器发送命令；测量值来自仿真 TCP 位置、雅可比和关节速度，不把参考位置或速度冒充物理反馈。它是单独的直线运动实验，未改变抓取控制器 `full_preplanned`；当前未验证从先前姿态到重置等待位的运动连接，也不是硬件或完整抓取验证。

2026-10-29 验证 `runs/visualization/linear_tcp_slow_viewer.json`：目标直线从约 `[0.0405, -0.00024, 0.7000] m` 到 `[0.1605, -0.00024, 0.7000] m`。参考轨迹持续 1.5 仿真秒。匀速参考区间的实际 X 速度均值为 9.627 cm/s，包含进入匀速时的执行器滞后；从仿真时刻 0.5 s 到匀速段结束的 35 个样本，平均 9.968 cm/s，标准差 0.042 cm/s。最大横向偏移约 1.41 mm，末端停止后的位置误差约 1.04 mm。速度数据记录于同名 `.csv`；参考匀速不意味着起步、停止或物理反馈每个采样都瞬时匀速。

可视窗口绿色是参考路径，蓝色是仿真反馈路径，右侧显示目标和实际 X 速度。`--playback-rate 0.25` 只将画面按墙钟慢放到四分之一速度；CSV 中的时间和速度仍按仿真时间计算。

运行（使用安装好 MuJoCo、SciPy、PyYAML 等依赖的 Python）：

```bash
cd /home/ubuntu22/nero_conveyor_ros2
PYTHONPATH=src/nero_core:src/nero_bringup MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
python tools/demo_linear_tcp_sim.py --output runs/visualization/linear_tcp_slow_viewer.json \
  --viewer --hold-viewer --playback-rate 0.25 --start-delay-s 3
```

`--distance-m`、`--speed-m-s`、`--ramp-s` 可调整实验的路径长度、标称速度及加减速时长。系统会拒绝不满足这些数值约束、IK 不收敛或碰撞/关节范围检查失败的设置。
