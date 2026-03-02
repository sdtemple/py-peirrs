"""Multitype epidemic models for PEIRRS."""

from .estimators import (
    peirr_tau_multitype,
    peirr_bayes_multitype,
    peirr_bootstrap_multitype,
    bayes_complete_multitype,
)
from .simulate import (
    simulator_multitype,
)

__all__ = [
    "peirr_tau_multitype",
    "peirr_bayes_multitype",
    "peirr_bootstrap_multitype",
    "bayes_complete_multitype",
    "simulator_multitype",
]
