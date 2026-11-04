#!/usr/bin/env bash
# One real-hardware pick; belt speed is supplied by the operator, not measured.
set -eo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/humble/setup.bash
source "$project_dir/install/setup.bash"

config="$project_dir/install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml"
output="$project_dir/runs/two_second_hardware/fixed_580_$(date +%Y%m%d_%H%M%S_%N)"
printf '[配置] 固定带速 5.8 cm/s，单件抓取，记录 RGB-D\n'
printf '[日志] %s\n' "$output"
printf '[回位] 如机械臂不在等待位，程序会尝试以配置的 100%% 速度自动回位\n'

export OPENBLAS_NUM_THREADS=1
exec ros2 run nero_core two_second_pick \
  --config "$config" \
  --fixed-belt-speed-m-s 0.058 \
  --execute --count 1 --record-rgbd \
  --output "$output"
