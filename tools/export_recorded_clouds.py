"""Export saved RGB-D frames as colored PLY; no device connections."""
import argparse
import json
from pathlib import Path
import numpy as np

from nero_core.two_second.collision_view import restore_frame, project_base
from mujoco_conveyor_sorting.runtime.single_pick import frame_points
from mujoco_conveyor_sorting.runtime.depth_calibration import (
    apply_runtime_depth_calibration, load_runtime_depth_calibration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--calibration', type=Path, required=True,
                        help='运行使用的深度标定文件；将保存的原始米制深度校正后重建')
    parser.add_argument('--frame', help='仅导出指定帧文件名；省略时导出全部')
    args = parser.parse_args()
    calibration = load_runtime_depth_calibration(args.calibration)
    output = args.run / 'pointclouds'
    output.mkdir(exist_ok=True)
    files = sorted((args.run / 'rgbd_frames').glob('*.npz'))
    if args.frame:
        files = [path for path in files if path.name == args.frame]
        if not files:
            parser.error('指定帧不存在')
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            frame = restore_frame(meta, {'frame_rgb': data['frame_rgb'], 'frame_depth_m': data['frame_depth_m']})
        d2c = np.asarray(meta['depth_to_color_cv'])
        frame = apply_runtime_depth_calibration(frame, calibration, d2c)
        points = frame_points(frame).reshape(-1, 3)
        # Missing depth is represented as zero; it is not a physical point.
        points = points[frame.depth_m.reshape(-1) > 0]
        uv, visible = project_base(points, frame, d2c)
        color = np.zeros((len(points), 3), dtype=np.uint8)
        pixel = np.rint(uv[visible]).astype(int)
        pixel[:, 0] = np.clip(pixel[:, 0], 0, frame.rgb.shape[1]-1)
        pixel[:, 1] = np.clip(pixel[:, 1], 0, frame.rgb.shape[0]-1)
        color[visible] = frame.rgb[pixel[:, 1], pixel[:, 0]]
        out = output / (path.stem + '.ply')
        vertices = np.empty(len(points), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),
                                                ('red','u1'),('green','u1'),('blue','u1')])
        for index, key in enumerate(('x','y','z')): vertices[key] = points[:, index]
        for index, key in enumerate(('red','green','blue')): vertices[key] = color[:, index]
        with out.open('wb') as stream:
            stream.write(('ply\nformat binary_little_endian 1.0\n'
                          f'element vertex {len(vertices)}\n'
                          'property float x\nproperty float y\nproperty float z\n'
                          'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode())
            vertices.tofile(stream)
        print(f'{out}: {len(points)} points', flush=True)


if __name__ == '__main__':
    main()
