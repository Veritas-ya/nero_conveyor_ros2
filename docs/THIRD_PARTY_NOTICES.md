# 随包第三方源码与资产

- **Nero7_grasping 测速模块**：应用户要求复用同门仓库
  https://github.com/Veritas-ya/Nero7_grasping ，提交
  `57735899ab138ce293982059ec642b5dec669d78`。
  上游包声明 MIT；原检测、跟踪、XY 卡尔曼和标签源码保存在
  `vendor/nero7_speed_reference/`，来源与哈希见该目录 `source_manifest.json`。
  本机相机输入和独立测速入口适配在项目自身代码中，未复制其机器人控制。
- **AgileX pyAgxArm SDK**：保留在 `vendor/pyAgxArm/`，版本 1.0.0。
  项目此前固定的源码 revision 为 `e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b`。
  LGPL-3.0-only，完整许可证见 `vendor/pyAgxArm/LICENSE`。本包复制 SDK 源文件，
  排除缓存、演示、测试和不相关外围资产，未修改 SDK 源码。
- **AgileX NERO mesh/描述**：来自 agx_arm_urdf，revision
  `f6642ce0d7872c686f29c99e9e10cd23d1d49313`，MIT。
  许可证与项目模型适配说明见 `mujoco_conveyor_sorting/assets/robots/nero/`。
- **T-LESS 派生几何**：Hodaň 等作者的 T-LESS / BOP 模型，CC BY 4.0。
  既有米制 OBJ 和出处见 `mujoco_conveyor_sorting/assets/tless/SOURCE.md`。
  本包不包含原始数据集或下载归档。

MuJoCo、NumPy、OpenCV、PyYAML、ImageIO、python-can 和 Orbbec SDK 等依赖由目标机
通过各自发布包安装，遵循各自许可证。项目自己的算法代码按当前工作区复制，
本迁移包不为原项目额外声明新的许可证。

- **Orbbec SDK v1 for Dabai DC1**：由 `scripts/install_dabai_sdk.py` 从官方
  https://github.com/orbbec/OrbbecSDK/releases/tag/v1.10.37 下载固定的 Linux x64
  发布包，版本 1.10.37，SHA256 验证后安装到忽略提交的 `vendor/orbbec_sdk_v1/`。
  发布包自带 `OrbbecSDK_v1.10.37/LICENSE.txt`；使用者应遵循其许可证。
  本地 ctypes 接口只加载该版本的 C API，不替换现有 SDK v2 Python 包。

- **NVIDIA GraspGen-X**：固定版本源码保留在 `experiments/graspgenx_sim/vendor/GraspGenX/`。
  上游为 https://github.com/NVlabs/GraspGenX ，具体 revision 见目录中的
  `UPSTREAM_COMMIT.txt`，Apache-2.0 许可证保留在上游源码目录。
  网络权重不随 Git 仓库提交，由独立下载脚本取得并验证哈希。
- **gripper_descriptions**：保留在 `experiments/graspgenx_sim/vendor/gripper_descriptions/`。
  各夹爪资产的许可不统一，请参阅该目录中的 `LICENSE_ASSETS` 及原始许可文件；
  上游信息与锁定清单保留在供应商源码目录，不能将全部资产视为同一许可证。
