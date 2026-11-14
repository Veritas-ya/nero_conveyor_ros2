# 多帧 ID 匹配离线诊断

`tools/diagnose_track_matching.py` 逐帧回放已保存的 RGB-D，通过当前源码中的生产测速和跟踪器，记录每个目标的 ID、确认状态、观测位置、检测与跟踪耗时，以及关联前的距离和统计门限。程序不打开相机、不连接 CAN、不发送机械臂命令，也不执行点云冻结、轨迹规划或抓取。

在项目根目录运行，必须使用与录制画面相对应的相机标定。下面是本机 2026-11-04 连续过件的离线例子；配置副本保存在 `runs/diagnostics/track_matching_20261104_config.yaml`，仅用于这批历史画面。

```bash
cd /home/ubuntu22/nero_conveyor_ros2
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src/nero_core:tools \
  python3 tools/diagnose_track_matching.py \
  --config runs/diagnostics/track_matching_20261104_config.yaml \
  --run runs/camera_diagnosis_20261104_pass \
  --isolate-depth-gaps \
  --output runs/diagnostics/my_track_diagnosis
```

输出 `summary.json` 为统计和各 ID 第一次确认的时间；`frames.jsonl` 为逐帧数据。`one_qualified_track` 只代表跟踪和速度条件通过，不代表点云已经冻结或轨迹已交付。`association_pairs` 是进入跟踪器前测出的关联距离和门限；工具只增加诊断，不改变跟踪器的分配决定。

可加 `--fixed-belt-speed-m-s 0.39` 做固定速度假设对照；此值必须根据具体实验设定，示例不代表真实带速。可以用 `--no-isolate-depth-gaps` 与 `--isolate-depth-gaps` 对比候选隔离。两种对照仅用于离线诊断，不改变配置文件或实际设备。

检查报告里的序号和时间间隔：低频或跳帧存档无法复现原本 30 Hz 实时关联。统计含空带时段；一个 ID 不一定是一件工件。工具的测量代码会增加少量开销。不同时间的相机安装、标定、设备或源码会影响结果，历史结果不能直接视作当前真机表现。
