#!/usr/bin/env bash
set -e
cd /home/ubuntu22/nero_conveyor_ros2
source /opt/ros/humble/setup.bash
source /home/ubuntu22/nero_conveyor_ros2/install/setup.bash
exec /home/ubuntu22/nero_conveyor_ros2/install/nero_core/lib/nero_core/record_taught_path --config /home/ubuntu22/nero_conveyor_ros2/install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml --segment transfer --seconds 60 --hz 20 --output /home/ubuntu22/nero_conveyor_ros2/taught/session_20261104_223416/transfer.json
