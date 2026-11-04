"""Record already-moving robot feedback without enabling or sending CAN commands."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from .config import CellConfig
from .taught import provenance


def collect(device, seconds, hz, *, clock=time.monotonic, sleep=time.sleep):
    from mujoco_conveyor_sorting.runtime.feedback_clock import mapped_stamp
    started = clock()
    deadline = started + seconds
    next_sample = started
    samples = []
    epoch = None
    maximum_packet_span = 0.
    interrupted = False
    try:
        while clock() < deadline:
            sleep(max(0., next_sample-clock()))
            state = device.snapshot_teaching_joints()
            stamp = mapped_stamp(state, 'joint')
            q = np.asarray(state['joints'], dtype=float)
            if stamp is None or not np.isfinite(stamp) or q.shape != (7,) or not np.isfinite(q).all():
                raise ValueError('关节反馈时间或角度无效')
            if samples and stamp < samples[-1][0]:
                raise ValueError('关节反馈时钟倒退，录制已取消')
            if epoch is None:
                epoch = state['clock_epoch']
            if state['clock_epoch'] != epoch:
                raise ValueError('关节反馈时钟重置，录制已取消')
            maximum_packet_span = max(maximum_packet_span, float(state['joint_packet_span_s']))
            if not samples or stamp > samples[-1][0]:
                samples.append((float(stamp), q.copy()))
            next_sample = max(next_sample+1./hz, clock())
    except KeyboardInterrupt:
        interrupted = True
    if len(samples) < 4:
        raise ValueError('有效样本不足，未保存示教轨迹')
    stamps = np.array([s for s, _ in samples])
    return {'elapsed_s': (stamps-stamps[0]).tolist(),
            'joints_rad': np.stack([q for _, q in samples]).tolist(),
            'capture': {'time_base': 'host_monotonic_mapped_from_can_receive',
                'sample_count': len(samples), 'requested_hz': hz,
                'clock_epoch': epoch, 'max_joint_packet_span_s': maximum_packet_span,
                'max_feedback_gap_s': float(np.max(np.diff(stamps))), 'interrupted': interrupted,
                'first_joint_stamp_monotonic_s': float(stamps[0]),
                'last_joint_stamp_monotonic_s': float(stamps[-1]),
                'limitation': 'CAN host receipt time is not simultaneous device sample time'}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--segment', choices=['transfer', 'return'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, required=True)
    p.add_argument('--hz', type=float, default=20.)
    args = p.parse_args(argv)
    if not np.isfinite([args.seconds, args.hz]).all() or args.seconds <= 0 or not 0 < args.hz <= 50:
        p.error('录制秒数须为正数，频率须在 (0, 50] Hz')
    if args.output.exists():
        p.error('输出文件已存在；请选择新的文件名')
    cfg = CellConfig.load(args.config)
    from mujoco_conveyor_sorting.runtime.nero_pick_hardware import Hardware
    device = Hardware(cfg.raw['hardware']['can_channel'], cfg.raw['hardware']['firmware'], False)
    try:
        device.start()
        print('[示教] 只读录制开始；保持起点和终点静止后再结束。Ctrl+C 可提前结束。', flush=True)
        data = collect(device, args.seconds, args.hz)
    finally:
        device.close()
    doc = {'schema': 'nero_taught_joint_path_v1', 'segment': args.segment,
           'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': provenance(cfg), **data}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(doc, file, ensure_ascii=False, indent=2)
        file.write('\n')
    print(f"[示教] 已保存 {len(data['elapsed_s'])} 个样本：{args.output}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
