"""Historical diagnostic name for the shared measured/simulation reference.

The experiment retains its original 0.6 s default; production configuration
and the simulation runner explicitly select a 0.8 s follow window.
"""
from dataclasses import dataclass
from nero_core.two_second.smooth_reference import SmoothIntercept


@dataclass(frozen=True)
class SmoothDiagnosticReference(SmoothIntercept):
    follow_s: float = .6
