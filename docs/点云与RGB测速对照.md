# 点云与 RGB 测速仿真对照

`tools/compare_rgb_cloud_velocity.py` 使用同一轮 RGB-D 仿真帧对比：

- 当前算法：原 Nero7 深度分割区域中心和中值深度转三维，使用 XY 卡尔曼滤波。
- 点云实验：目标可见点云的相邻帧平移 ICP；中位数位移初始化、最近邻和稳健平移，非既有部署算法；当前仅估计平移、不估计旋转。未用真实速度或设定带速作为输入。
- RGB 实验：纯 RGB 饱和橙色分割初始化目标特征，之后用 RGB KLT 光流；射线投影到固定标定带面以换算米，未逐帧使用深度。橙色阈值仅适合当前仿真物体，不是通用 RGB 检测器。三者只对已有 RGB-D 跟踪器确认且测速有效的帧统计，因此覆盖率并非独立 RGB 检测器的总帧召回率。

真值为两次曝光时物体 MuJoCo 世界 X 位置之差除以时间差；只用于离线评分。点云/RGB 是区间平均速度，卡尔曼值是后一曝光时刻滤波估计，减速时两者物理含义有区别。两次曝光时间相同且目标单一，仿真带速恒定 0.1 m/s；结果、原始记录分别为 `runs/visualization/rgb_cloud_speed_final.json` 和同名 CSV，图为同名 PNG。

本次 258 个共同有效帧，平均绝对误差：当前卡尔曼 0.1515 cm/s，点云 ICP 0.2417 cm/s，RGB 光流 0.5622 cm/s。前两种实验方法没有卡尔曼平滑；不存在算法或参数公平最优的结论，尤其 RGB 固定平面会受物体高度和成像角度影响。点云 ICP、RGB 光流的平均核心方法耗时分别约 2.83、3.76 ms，不包含相机采集、分割、点云提取、模型初始化或完整管线时间；同步仿真不证明实时性能。只是一种仿真物体、视角和轨迹，未覆盖实机传感器噪声、遮挡和材质泛化，也没有抓取。

```bash
cd /home/ubuntu22/nero_conveyor_ros2
PYTHONPATH=src/nero_core:src/nero_bringup MUJOCO_GL=egl \
python tools/compare_rgb_cloud_velocity.py --seconds 18 --output runs/visualization/rgb_cloud_speed_final.json
```
