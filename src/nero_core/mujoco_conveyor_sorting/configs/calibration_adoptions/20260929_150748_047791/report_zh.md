# 外参采用记录

已按用户指示采用 20260929_134016_545973 的 18 组外参结果。

- 真机世界坐标系：NERO 机械臂基座；base_to_world = I。
- 运行矩阵：原生深度光学坐标 → 基座，平移单位米；不是 RGB 矩阵直接填入。
- 相机：CP28563000CT；彩色 1280×720 RGB@30，深度 1280×800 Y16@30。
- 平移一致性 RMS 2.872 mm、最大 5.120 mm；旋转一致性 RMS 0.622°、最大 1.009°。这些不是独立真值精度。
- 原求解器 accepted=false 原样保留：最大值略超 5 mm / 1° 门槛。此次为用户指定采用，未修改自动质量门槛。
- 原始内参、原始采集数据和原结果文件未改动；仅更新 runtime.yaml 中的外参和对应相机配置。
- 离线验证通过：工程配置加载、相机配置解析、旋转矩阵、深度/RGB 变换合成、设备与分辨率匹配、源文件哈希。
- 仍为 simulation 模式、production=false，安全配置不变；未连接硬件，未进行运动验证。

来源：/home/hzq/机械臂分拣/traditional_pipeline_linux_20260928/recordings/teach/20260929_134016_545973/evaluation_20260929_135049_957715/result.json

旧配置：runtime.before.yaml；采用配置快照：runtime.selected.yaml；完整记录：adoption.json。
