#!/usr/bin/env bash
# Continuous physical grasping with the current site configuration and preview.
# Forward optional arguments, e.g. --belt-speed-cm-s 4.53. Ctrl+C cancels.
set -eo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
source /opt/ros/humble/setup.bash
source "$project_dir/install/setup.bash"
run_stamp="$(date +%Y%m%d_%H%M%S)_$$"
exec env OPENBLAS_NUM_THREADS=1 ros2 run nero_core two_second_pick \
  --config "$project_dir/install/nero_bringup/share/nero_bringup/config/two_second.hardware.legacy_observed.yaml" \
  --continuous --execute --preview \
  --output "$project_dir/runs/continuous_$run_stamp" "$@"
