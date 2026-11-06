#!/usr/bin/env bash
set -e
cd /home/ubuntu22/nero_conveyor_ros2
source /opt/ros/humble/setup.bash
source /home/ubuntu22/nero_conveyor_ros2/install/setup.bash
export OPENBLAS_NUM_THREADS=1
exec /home/ubuntu22/nero_conveyor_ros2/install/nero_core/lib/nero_core/two_second_pick --config /home/ubuntu22/nero_conveyor_ros2/install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml --taught-transfer /home/ubuntu22/nero_conveyor_ros2/taught/session_20261104_223416/transfer_trimmed_start025_20261105.json --taught-return /home/ubuntu22/nero_conveyor_ros2/taught/session_20261104_223416/planned_return/return_generated.json --output "/home/ubuntu22/nero_conveyor_ros2/runs/two_second_hardware/taught_new_$(date +%Y%m%d_%H%M%S)" "$@"
