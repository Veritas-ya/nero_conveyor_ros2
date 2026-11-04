"""Simulation-only causal constant-velocity fit to observed RGB-D positions.

Each identity has independent online least-squares statistics. No belt setpoint,
object truth, predicted positions or future observations enter the fit.
"""
import numpy as np


class ConstantVisualSpeed:
    def __init__(self, *, min_samples=8, min_span_s=.45):
        self.min_samples = int(min_samples)
        self.min_span_s = float(min_span_s)
        self.tracks = {}

    def update(self, identity, stamp, position):
        identity = tuple(identity)
        stamp = float(stamp)
        p = np.asarray(position, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(stamp):
            return False
        state = self.tracks.get(identity)
        if state is None:
            self.tracks[identity] = {'first': stamp, 'last': stamp, 'n': 1,
                                     'mean_t': 0., 'mean_p': p[:2].copy(),
                                     's_tt': 0., 's_tp': np.zeros(2), 's_pp': np.zeros(2)}
            return True
        if stamp <= state['last']:
            return False
        t = stamp - state['first']
        n = state['n'] + 1
        dt = t - state['mean_t']
        dp = p[:2] - state['mean_p']
        state['mean_t'] += dt / n
        state['mean_p'] += dp / n
        state['s_tt'] += dt * (t - state['mean_t'])
        state['s_tp'] += dt * (p[:2] - state['mean_p'])
        state['s_pp'] += dp * (p[:2] - state['mean_p'])
        state['last'], state['n'] = stamp, n
        return True

    def estimate(self, identity):
        state = self.tracks.get(tuple(identity))
        if state is None or state['n'] < self.min_samples or state['last'] - state['first'] < self.min_span_s or state['s_tt'] <= 0.:
            return None
        velocity = state['s_tp'] / state['s_tt']
        residual_squares = np.maximum(0., state['s_pp'] - state['s_tp'] ** 2 / state['s_tt'])
        return {'velocity_world_m_s': np.r_[velocity, 0.],
                'sample_count': state['n'], 'span_s': state['last'] - state['first'],
                'first_stamp_s': state['first'], 'last_stamp_s': state['last'],
                'residual_rms_xy_m': np.sqrt(residual_squares / state['n'])}
