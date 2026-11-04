"""Write an honest, compact Chinese report from an executed benchmark."""
from pathlib import Path


def write_report(output, result):
    output = Path(output)
    config = result["config"]["experiment"]
    evaluation = result.get("evaluation", result.get("interim", {}))
    timing = result.get("timing", {})
    successes = evaluation.get("successful_sorted_objects", 0)
    count = evaluation.get("expected_objects", config["objects"])
    rate = evaluation.get("success_rate", 0.)
    lower = evaluation.get("one_sided_95_percent_lower_bound", 0.)
    lines = [
        "NERO + AGX 传送带分拣独立仿真实测报告",
        f"抓取来源：{config.get('grasp_source', 'graspgenx')}；工况：{config.get('test_mode', 'moving')}",
        f"可用于新方法运动验收：{result.get('acceptance_eligible', False)}（停带和旧模板仅作诊断）。",
        f"运行状态：{result['status']}",
        f"官方网络已实际执行：{result.get('network_executed', False)}",
        f"带速：{config['belt_speed_m_s']} m/s；来料间隔：{config['feed_interval_s']} s",
        f"类别范围：{config['active_category_ids']}；随机种子：{config['seed']}",
        "物理判据：按实际碰撞几何与实体箱壁/箱底内域判定完整落箱，接触正确箱、无手指接触，并低速稳定至少0.5秒。",
        f"几何评分版本：{evaluation.get('geometry_metric_version', '未记录')}；边界容差3mm、地面容差6mm；catalog包围盒仅作诊断。",
        f"正确落箱：{successes}/{count}；实测成功率：{rate:.2%}",
        f"未成功或未完成：{evaluation.get('failed_or_incomplete_objects', count - successes)}；未到料也计分母。",
        f"抓到但未正确落箱：{evaluation.get('grasped_but_not_sorted', 0)}",
        f"单侧95%二项置信下界：{lower:.2%}（采用独立同分布假设；连续周期可能相关）。",
        f"实测分拣{config['target_success_rate']:.0%}目标：{result.get('sorting_target_met', False)}",
        f"实时节拍目标：{result.get('timing_target_met', False)}",
        f"联合目标通过：{result.get('target_met', False)}",
        f"推理调用：{result.get('inference_calls', 0)}；不把网络评分作为抓取成功。",
        f"计时模式：{timing.get('mode', '尚未完成')}",
        f"仿真秒：{timing.get('sim_seconds')}；墙钟秒：{timing.get('wall_seconds')}",
        f"实时因子：{timing.get('real_time_factor')}；最大落后秒：{timing.get('maximum_wall_lag_s')}",
        f"正确落箱吞吐：{evaluation.get('throughput_sorted_per_minute')} 件/分钟（含首件到达等待）。",
        f"平均完成间隔：{evaluation.get('mean_completion_interval_s')} 秒。",
        "",
        "复现文件：config.yaml、result.json、objects.csv、inputs/、inference/、continuous_sorting.mp4。",
        "inputs保留实际控制帧深度和视觉mask；inference保留网络选中TCP坐标轴图。放大图只用于查看，不改变网络输入。",
        "GraspGen-X入口使用官方固定版本生成器和判别器，纯diffusion候选；无PCA/OBB抓取候选回退。旧模板入口独立标记。",
        "控制：视觉深度组件跟踪 + 带速补偿 + 既有NERO运动控制；GraspGen-X提供抓取姿态。",
        "夹爪：真实模型几何与开合参数，未启用抓取吸附、抓取焊接或物体跟随。",
        "相机：保存数据的固定外置相机相对基座姿态；当前传送带/箱体布局属于已有仿真。",
        "碰撞模型：T-LESS使用现有凸包近似，孔/凹陷不等同于真实物件接触。",
        f"终态对象{config['terminal_retention_s']}秒后回收；本轮不验证箱体装满后的堆积。物体固定方向传播仅适用于沿带面平移，滚动/旋转还需重新推理。",
        "正式控制：延迟视觉buffer、关节与夹爪编码器、执行器力和带速编码器；不读取对象真值位姿、接触对象ID或物理生命周期。",
        "夹爪确认来自电机堵转与非零开口持续反馈，不能识别夹住的对象；正确对象与完整入箱仅由独立评分器验证。",
        "仿真成功率只适用于记录的类别、来料、姿态和物理参数，不能保证真机或任意场景达到目标。",
    ]
    if result.get("error"):
        lines += ["", "运行错误：" + result["error"]]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
