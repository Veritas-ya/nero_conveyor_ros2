"""Simulation-only candidate for a fixed command-reference time advance.

This is not a measured servo model. It creates a smooth curve proposal; the
existing full planner must certify it before the physical harness may send it.
"""
import numpy as np
from scipy.interpolate import make_interp_spline
from mujoco_conveyor_sorting.controllers.full_pick_response import ResponseCurve, command_within_limits


def advance_curves(request, curves, advance_s):
    approach, descent, closing, lift = curves
    advance = float(advance_s)
    if not np.isfinite(advance) or advance <= 0.:
        raise ValueError('simulation advance must be finite and positive')
    if request.full_approach_mode != 'continuous' or descent is not None or closing is None or lift is None:
        raise ValueError('requires the complete continuous approach, closing and lift')
    contact, end = approach.duration, approach.duration + closing.duration
    if contact <= 0. or closing.duration <= 0.:
        raise ValueError('invalid trajectory duration')
    def nominal(t):
        return approach.sample_full(t) if t < contact else closing.sample_full(min(t-contact, closing.duration))
    times = np.unique(np.r_[np.linspace(0., contact, max(8, int(np.ceil(contact/.08)))+1),
                            np.linspace(contact, end, max(8, int(np.ceil(closing.duration/.08)))+1)])
    positions = []
    for t in times:
        # Smooth start and fade during closing; no discontinuous first command
        # and the nominal loaded lift starts with its original q/v/a.
        start_u = float(np.clip(t/(2*advance), 0., 1.))
        ramp = 10*start_u**3-15*start_u**4+6*start_u**5
        u = float(np.clip((t-contact)/closing.duration, 0., 1.))
        fade = 1.-(10*u**3-15*u**4+6*u**5)
        positions.append(nominal(min(end, t+advance*ramp*fade))[0])
    initial, final = nominal(0.), nominal(end)
    positions[0], positions[-1] = initial[0], final[0]
    spline = make_interp_spline(times, np.asarray(positions), k=5,
        bc_type=([(1, initial[1]), (2, initial[2])], [(1, final[1]), (2, final[2])]))
    proposal = (ResponseCurve(spline, 0., contact), None,
                ResponseCurve(spline, contact, end), lift)
    return proposal, float(command_within_limits(request, proposal))


def install_simulation_advance(engine, advance_s):
    """Require the proposed curve and original full certificate; never fallback."""
    advance = float(advance_s)
    original_once = engine._plan_full_pick_once
    def proposal(request, curves, report):
        diagnostic = {'method': 'simulation_fixed_reference_advance', 'advance_s': advance,
                      'measured_servo_model': False, 'applied': False}
        report['response'] = diagnostic
        try:
            commands, ratio = advance_curves(request, curves, advance)
            diagnostic['command_derivative_ratio'] = ratio
            if ratio > 1.+1e-9:
                diagnostic['status'] = 'command_limit_rejected'
                return None, 0.
            diagnostic['status'] = 'candidate_requires_full_geometry_certificate'
            return commands, 0.
        except (ValueError, TypeError) as error:
            diagnostic.update(status='proposal_unavailable', reason=str(error))
            return None, 0.
    def plan_once(request, cancelled, started):
        result = original_once(request, cancelled, started)
        if result.success and (result.full_pick_plan is None or result.full_pick_plan.command_curves is None):
            return engine._full_failure(request, 'simulation_advance_not_fully_certified', started)
        return result
    engine._full_response_proposal = proposal
    engine._plan_full_pick_once = plan_once
