"""Preview the selected pick plan without replaying or controlling hardware."""
from pathlib import Path
import json

import cv2
import numpy as np

from .grasp_pose_display import save_grasp_pose_overlay


def save_grasp_preview(directory, plan, observation, model, *, prefix='grasp_pose'):
    """Persist the actual selected TCP, camera overlay and interactive scene.

    Model updates below are private FK computations; restore the caller's
    state even if preview generation fails. No IK, replanning or device IO.
    """
    if 'flange_to_tcp_nominal' in plan:
        error_message = '保存计划的夹爪/TCP与当前模型不一致，请先离线重新规划'
        try:
            saved_offset = np.asarray(plan['flange_to_tcp_nominal'], dtype=float)
            model_offset = np.asarray(model.flange_to_tcp, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError(error_message) from error
        if (saved_offset.shape != (4, 4) or model_offset.shape != (4, 4)
                or not np.isfinite(saved_offset).all() or not np.isfinite(model_offset).all()
                or not np.allclose(saved_offset, model_offset, atol=1e-6, rtol=0)):
            raise ValueError(error_message)

    from tools.visualize_pick_scene import (
        display_points, image_data, mesh_layout, world_vertices)

    directory = Path(directory)
    frame = observation['frame']
    artifacts = save_grasp_pose_overlay(
        directory, plan, frame, observation['depth_to_color_cv'],
        flange_to_tcp=model.flange_to_tcp, prefix=prefix)
    report = artifacts['report']
    report['image_frame'] = Path(observation['record_stem']).name
    Path(artifacts['json']).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    descend = next((stage for stage in plan['stages'] if stage['name'] == 'descend'), None)
    if descend is None and plan.get('preview_kind') != 'full_policy_target':
        raise ValueError('selected plan has no descent trajectory')
    width = plan['settings']['open_width_m']
    qpos = model.data.qpos.copy()
    try:
        layout, triangles = mesh_layout(model)
        initial = world_vertices(model, layout, plan['start_joints_rad'],
                                 observation['scene_state']['width_m']).copy()
        grasp = (world_vertices(model, layout, descend['joints_rad'][-1], width).copy()
                 if descend is not None else None)
        trajectory = []
        model.set_state(plan['start_joints_rad'], width)
        trajectory.append(model.tcp()[:3, 3].tolist())
        for stage in plan['stages']:
            for q in stage['joints_rad']:
                model.set_state(q, width)
                trajectory.append(model.tcp()[:3, 3].tolist())
    finally:
        model.data.qpos[:] = qpos
        model.mj.mj_forward(model.model, model.data)
    layers = []
    for name, label, color, points, limit in [
        ('scene', '参与规划的障碍点', '#d78a40', model.scene, 40000),
        ('target', '目标工件', '#5de18a', observation['object_points'], 10000),
    ]:
        layers.append({'name': name, 'label': label, 'color': color,
                       'points': display_points(points, limit), 'point_size': 2,
                       'visible': True})
    overlay = cv2.imread(artifacts['overlay_png'])
    if overlay is None:
        raise OSError(f"cannot load grasp preview: {artifacts['overlay_png']}")
    result = observation.get('cycle_result', {})
    notice = ('完整规划已通过。此预览显示计划位置，不能证明实际夹持成功。'
              if not result else f"本次执行状态：{result.get('status', 'unknown')}。"
              f"{result.get('error', result.get('note', ''))} 此预览显示计划位置。")
    if plan.get('preview_kind') == 'full_policy_target' and not result:
        notice = ('Fast双目重算的可见目标，属于学习估计；场景障碍仍为原生深度。'
                  if observation.get('grasp_depth_refinement', {}).get('applied') else
                  '实测抓取目标；')
        notice += '完整策略将从当前关节规划。尚未生成整段轨迹。'
    if plan.get('moving_belt'):
        notice += ' 工件图层和抓取坐标是初始观测快照；预测拦截与移动工件请查看完整动态轨迹回放。'
    robot_states = [{'label': '采样时模型姿态', 'vertices': display_points(initial, len(initial))}]
    if grasp is not None:
        robot_states.append({'label': ('策略下降末端姿态（张开夹爪）'
                                      if plan.get('preview_kind') == 'full_policy_target' else '计划抓取姿态（张开夹爪）'),
                             'vertices': display_points(grasp, len(grasp))})
    payload = {
        'title': '本次计划抓取位姿', 'source': str(directory.resolve()),
        'frame': Path(observation['record_stem']).name,
        'pairing_note': '当前采样的场景与已选计划；图像为抓取前快照。',
        'error': notice,
        'mode': plan.get('perception_mode'), 'grasp_pose': report,
        'counts': {'scene': len(model.scene), 'target': len(observation['object_points'])},
        'detail': {'selected_plan': 'plan.json', 'pose': report, 'execution_result': result,
                   'grasp_depth_refinement': observation.get('grasp_depth_refinement')},
        'images': {'rgb': image_data(cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)),
                   'overlay': image_data(overlay)}, 'layers': layers,
        'robot': {'triangles': triangles, 'states': robot_states},
        'trajectory': trajectory, 'target_frame': plan['grasp_tcp'],
        'camera_position': frame.camera_to_world_cv[:3, 3].tolist(),
        'collision_links': [], 'preview_kind': plan.get('preview_kind', 'selected_plan'),
    }
    template = (Path(__file__).resolve().parents[2]/'tools/pick_scene_view.html').read_text(encoding='utf-8')
    data = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    path = directory/f'{prefix}.html'
    path.write_text(template.replace('__SCENE_DATA_JSON__', data), encoding='utf-8')
    artifacts['html'] = str(path)
    return artifacts
