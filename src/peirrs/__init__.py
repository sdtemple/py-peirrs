"""PEIRRS: Pair-based Estimators of Infection and Removal Rates."""

__version__ = "1.0.0"

# Core modules
from . import simulate, estimators, utils

# Subpackages
from . import spatial, multitype

# Main estimation functions (core)
from .estimators import (
	peirr_tau,
	peirr_imputed,
	peirr_bootstrap,
	peirr_bayes,
	peirr_removal_rate,
)

# Main simulation functions (core)
from .simulate import (
	simulator,
)

from .multitype import (
    simulator_multitype,
    peirr_tau_multitype,
    peirr_bayes_multitype,
    peirr_bootstrap_multitype,
)

__all__ = [
	# Version
	"__version__",
    
	# Subpackages
	"simulate",
	"utils",
	"estimators",
	"spatial",
	"multitype",
    
	# Core estimation functions
	"peirr_tau",
	"peirr_imputed",
	"peirr_bootstrap",
	"peirr_bayes",
	"peirr_removal_rate",
    
	# Core simulation functions
	"simulator",
    
	# Multitype estimation functions
	"peirr_tau_multitype",
	"peirr_bootstrap_multitype",
	"peirr_bayes_multitype",
    "simulator_multitype",
    
]