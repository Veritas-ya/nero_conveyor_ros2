#!/usr/bin/env python3
"""Offline direct return in saved scenes; no device connection or commands."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from nero_core.two_second.config import CellConfig
from nero_core.two_second.diagnostics import json_value
from nero_core.two_second.planning import MeasuredPlanner
from nero_core.two_second.prepared_cycle import nominal_state
from nero_core.two_second.prepared_cycle import direct_return
from nero_core.two_second.taught import TaughtSegment
from nero_core.two_second.trajectory import extrema
from mujoco_conveyor_sorting.runtime.single_pick import PickRejected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = CellConfig.load(args.config)
    transfer = TaughtSegment.load(config.raw['taught_transfer_file'], config, 'transfer')
    rows = []
    for source in args.inputs:
        planner = MeasuredPlanner(config)
        planner.identity = (0, 1, len(rows))
        with np.load(source, allow_pickle=False) as data:
            planner.model.scene = data['scene'].copy()
            planner.model.tree = cKDTree(planner.model.scene)
            planner.plane = data['plane'].copy()
            planner.belt_bounds = data['belt_bounds'].copy()
            planner.support_origin = data['support_origin'].copy()
            planner.support = planner.plane[:3, 2].copy()
            planner.payload_local = data['payload_local'].copy()
            planner.payload_tree = cKDTree(planner.payload_local)
            width = float(data['width_m'])
        planner.scene_revision = 1
        q_drop = transfer.joints[-1]
        row = {'input': str(source), 'status': None, 'scene_points': len(planner.model.scene),
               'payload_points': len(planner.payload_local)}
        started = time.monotonic()
        try:
            drop = nominal_state(planner.model, q_drop, width)
            planner.validate_release_opening(drop)
            planner.release_measured(drop)
            opening = config.settings.open_width_m
            return_started = time.monotonic()
            path = direct_return(planner, config, q_drop, opening, cancelled=lambda: False)
            first_seconds = time.monotonic()-return_started
            planner.check_path(path, opening, payload=False)
            reuse_started = time.monotonic()
            reused = direct_return(planner, config, q_drop, opening, cancelled=lambda: False)
            reuse_seconds = time.monotonic()-reuse_started
            if not reused.diagnostics['route_template_reused']:
                raise PickRejected('直接返回模板未按预期复用')
            row.update(status='path_certified_offline', duration_s=path.duration,
                       planning_wall_s=time.monotonic()-started,
                       first_return_planning_wall_s=first_seconds,
                       reuse_recheck_wall_s=reuse_seconds,
                       start_error_rad=float(np.max(abs(path.q0-q_drop))),
                       end_error_rad=float(np.max(abs(path.q1-config.standby))),
                       peak_velocity_rad_s=np.max(abs(np.asarray(extrema(path.curve, 1))), axis=0),
                       peak_acceleration_rad_s2=np.max(abs(np.asarray(extrema(path.curve, 2))), axis=0),
                       route_diagnostics=path.diagnostics.get('route_diagnostics'))
        except PickRejected as error:
            row.update(status='rejected', error=str(error), details=error.details,
                       elapsed_s=time.monotonic()-started)
        rows.append(row)
        print(json.dumps(json_value(row), ensure_ascii=False), flush=True)
    result = {'scope': 'saved scene and estimated payload; not current live scene',
              'configuration': str(args.config.resolve()), 'hardware_connections': 0,
              'commands_sent': 0, 'physical_motion_tested': False, 'results': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as file:
        json.dump(json_value(result), file, ensure_ascii=False, indent=2)
        file.write('\n')
    return 0 if all(row['status'] == 'path_certified_offline' for row in rows) else 1


if __name__ == '__main__':
    raise SystemExit(main())
