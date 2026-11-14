"""Read local planning wall times for a startup estimate, never a motion certificate."""
import json
import math
from pathlib import Path
import platform

from .phase_timing_experiment import digest


def _rounds(value):
    if isinstance(value, dict):
        # New errors contain both forms. Read only one to avoid double counting.
        if 'measured_planning' in value:
            yield from value['measured_planning'].get('rounds', [])
        elif 'measured_rounds' in value:
            yield from value['measured_rounds']
        for key, child in value.items():
            if key not in ('measured_planning', 'measured_rounds'):
                yield from _rounds(child)
    elif isinstance(value, list):
        for child in value:
            yield from _rounds(child)


def startup_estimate(history_root, *, base_sha256, phase_timing, maximum_target_length_m,
                     run=None):
    """Newest compatible local run; max observed round, including failed rounds.

    A run from a different recorded host/configuration is not used. Older logs
    have no hostname: their local origin is an explicit assumption in provenance.
    Read only event files, not RGB-D/feedback traces. Invalid/incomplete runs are
    skipped during discovery; an explicitly selected invalid run is an error.
    """
    candidates = ([Path(run)] if run is not None else
                  sorted(Path(history_root).glob('*/experiment.json'), reverse=True)[:32])
    for candidate in candidates:
        directory = candidate if run is not None else candidate.parent
        try:
            manifest = json.loads((directory/'experiment.json').read_text())
            if (manifest.get('schema') != 'nero_phase_timing_hardware_test_v1'
                    or manifest.get('base_sha256') != base_sha256
                    or manifest.get('phase_timing') != phase_timing
                    or manifest.get('maximum_target_length_m', .16) != maximum_target_length_m
                    or manifest.get('host', platform.node()) != platform.node()
                    or manifest.get('config_sha256') != digest(directory/'effective.yaml')):
                raise ValueError('历史测时日志与当前主机或测试配置不匹配')
            samples, sources = [], []
            for path in sorted((directory/'hardware').glob('cycle_*/events.json')):
                events = json.loads(path.read_text())
                # Delivery records repeat the same diagnostics in components
                # and trajectory metadata; count each round only once per file.
                rows = {json.dumps(row, sort_keys=True):row for row in _rounds(events)}.values()
                values = [row.get('cold_equivalent_wall_s', row.get('wall_s')) for row in rows
                          if row.get('result') in ('missed', 'delivered', 'rejected')
                          and row.get('estimate_eligible', True)]
                values = [float(v) for v in values if isinstance(v, (int, float))
                          and not isinstance(v, bool) and math.isfinite(v) and v > 0]
                if values:
                    samples.extend(values)
                    sources.append({'path':str(path.resolve()), 'sha256':digest(path)})
            if not samples:
                raise ValueError('历史日志没有有效的测时规划轮次')
            return max(samples), {'policy':'latest_compatible_local_run_max_observed_round',
                'run':str(directory.resolve()), 'host':manifest.get('host'),
                'legacy_local_host_assumed': 'host' not in manifest,
                'sample_count':len(samples), 'min_s':min(samples), 'max_s':max(samples),
                'sources':sources, 'guaranteed_time_bound':False}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            if run is not None:
                raise
    return 0., {'policy':'no_compatible_local_timing_history', 'guaranteed_time_bound':False}
