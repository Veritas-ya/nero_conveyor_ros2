"""Explain response-model admission without changing planning or actuation."""
import numpy as np

from mujoco_conveyor_sorting.controllers.full_pick_response import eligible_response


def response_model_status(model, *, path=None):
    if model is None:
        return dict(configured=False, eligible=False, reason='no_explicit_model_file',
                    command_reference='nominal', path=None, axes=[])
    rows = []
    for index, axis in enumerate(model.axes):
        checks = {'present': axis is not None}
        values = {}
        if axis is not None:
            values = dict(delay_s=axis.delay_s, time_constant_s=axis.time_constant_s,
                holdout_rmse_rad=axis.holdout_rmse_rad,
                unmodeled_holdout_rmse_rad=axis.unmodeled_holdout_rmse_rad,
                holdout_samples=axis.holdout_samples, status=axis.status,
                parameter_at_boundary=axis.fit_at_search_boundary)
            checks.update(identified=axis.status == 'identified',
                finite=bool(np.isfinite([axis.delay_s, axis.time_constant_s,
                    axis.holdout_rmse_rad, axis.unmodeled_holdout_rmse_rad]).all()),
                enough_holdout=axis.holdout_samples >= 2,
                positive_time_constant=axis.time_constant_s > 0,
                nonnegative_delay=axis.delay_s >= 0,
                holdout_not_worse=axis.holdout_rmse_rad <= axis.unmodeled_holdout_rmse_rad)
        rows.append(dict(joint=index + 1, values=values, checks=checks))
    eligible = eligible_response(model)
    return dict(configured=True, eligible=eligible,
        reason='eligible_requires_command_certificate' if eligible else 'model_unvalidated',
        command_reference='planner_decides_after_full_certificate',
        path=None if path is None else str(path),
        runtime_eligible=bool(model.metadata.get('runtime_eligible', False)),
        seven_axes=len(model.axes) == 7, axes=rows)
