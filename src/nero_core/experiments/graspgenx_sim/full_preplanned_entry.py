"""Fail-closed entry for the pending live visual full-preplanned simulator.

The current visual adapter cannot yet provide same-frame collision geometry,
scene monitoring and certified trajectory ownership to the physical loop.
Do not start the historical policy as a substitute.
"""
import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='视觉网络 + full_preplanned 物理仿真入口')
    parser.add_argument('--output', help='预留输出目录；本入口不会创建文件')
    parser.add_argument('--planning-mode', choices=('full_preplanned',), default='full_preplanned')
    parser.parse_args(argv)
    print('视觉 full_preplanned 物理执行尚未接通：缺少实时视觉场景认证与认证轨迹发送链。'
          '已停止，未启动历史控制器、相机、网络推理或物理执行。'
          '可运行 physics_sim 验证 full_preplanned，但该入口使用仿真历史观测和目录抓姿。',
          file=sys.stderr)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
