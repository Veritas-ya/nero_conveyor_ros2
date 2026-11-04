AGX 原装 100 mm 两指夹爪：独立 GraspGen-X 仿真资产

重新生成与自检（在 traditional_pipeline_linux_20260928 目录）：
  .venv/bin/python experiments/graspgenx_sim/gripper_agx/build_assets.py

这是现有 NERO MuJoCo 模型的夹爪几何适配，未连接相机、CAN 或机器人。
生成器仅依赖工程已有 numpy、scipy、mujoco；不会安装软件、下载文件。

来源
1. 几何和夹爪运动学：mujoco_conveyor_sorting/models/robot.xml。
2. STL：读取当前 robot.xml 实际引用的网格；掌座采用
   gripper_base_drawing_20261012.stl，手指沿用原 STL。
   上游 https://github.com/agilexrobotics/agx_arm_urdf
   固定版本 f6642ce0d7872c686f29c99e9e10cd23d1d49313；MIT 许可证保留在 meshes/LICENSE。
3. 同版本官方 nero/urdf/nero_with_gripper_description.xacro 已归档至 source/，
   用于记录与上游的差异。当前夹指原点及镜像轴已有修订，不要求与旧描述完全相同；
   生成几何以当前模型为准，并用编译后的 MuJoCo 独立核对。
4. 配置字段依据官方 NVlabs/GraspGenX 的 scripts/gripper_config_wizard.py、
   graspgenx/x_grippers.py 和 graspgenx/serving/types.py。
   本次源码版本 b9429097728cb1c430dd78b92edf17ba318aad03。

坐标与使用
统一记号 T_A_B：把 B 坐标下的点变换到 A；所有长度为米。
canonical 原点是已有模型 gripper_base 原点。
canonical +X = gripper_base +Y：两指张开/闭合轴。
canonical +Y = gripper_base -X。
canonical +Z = gripper_base +Z：从掌部指向指尖的接近轴。
T_gripper_base_canonical 是 Rz(+90°)，已写入 transforms.json。
URDF 根坐标已直接对齐 canonical，故 config.json 的 base_rotation 为单位阵。

GraspGen-X 输入优先使用 sweep_volume_params.json 的 6 个字段，直接传给
SweepVolumeParams.from_dict 和 GraspGenXSampler.from_sweep_volume。
此文件没有额外元数据。无需注册 gripper 名称，无需修改 vendor 目录。
open/half 的内扫掠盒分别为 100/50 mm 宽；另两维来自实际指网格。
掌部顶部以上保留 1 mm，扫掠盒 z 约 0.070501 至 0.1425 m。
这是模型条件使用的轴对齐近似，不是经过力闭合证明的接触区。
真实手指有斜面，末端和靠掌部的局部开口不同，不能把盒宽当作每处精确接触宽度。

当前 fingertip_depth = 0.1425 m，不能把扫掠盒中心 z≈0.1065 m 当指尖。
当前 ee_site 在 gripper_base z=0.1325 m，比指尖平面后退 10 mm。
网络输出 T_world_canonical 后，模型 TCP 目标应为：
  T_world_ee_site = T_world_canonical @ T_canonical_ee_site
其中 T_canonical_ee_site 在 transforms.json，不仅有 z=0.1325 平移，
还包含 Rz(-90°) 旋转。不要把网络原点直接作为 TCP。
文件同时给出 gripper_flange、link7 的映射；这些是模型坐标，未声称是真机测量坐标。

其他资产
gripper.urdf：保留实际网格、左右关节与主开口关节的 mimic 关系。
meshes/：原始 STL 副本，未缩放、简化或凭空替换手指。
vis_mesh.obj / coll_mesh.obj：100 mm 张开时的真实合并网格，canonical 坐标。
points.json：真实网格面积加权确定性采样，open/close 各 10500 点。
未生成 TSDF/VAE 描述，仅面向当前发布模型的 sweep_volume_v2 条件接口。
完整物理碰撞和开合接触应继续使用 MuJoCo 内原有真实夹爪几何。
张开 OBJ 不代表闭合全过程；它不能替代轨迹碰撞或夹持动力学验证。

实际验证
geometry_check.json 记录每项数值、网格 SHA256、官方源核对结果。
生成后的 URDF 被重新解析，以独立 FK 验证 0/50/100 mm 三档开口。
同三档开口的四个实体坐标与现有 MuJoCo FK 比较，均应小于 1e-10。
上述检查验证的是几何、轴、单位和转换；没有测量或证明抓取成功率。
本资产不能作为真机 TCP 标定结果，也不提供任何真实 CAN/控制入口。
