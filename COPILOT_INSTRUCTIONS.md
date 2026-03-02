# Copilot Instructions: PEIRRS R to Python Translation

## Project Overview

**PEIRRS** (Pair-based Estimators of Infection and Removal Rates) is a statistical package for estimating epidemic parameters from partially observed infection and removal times. You are translating the R package into a pure Python package.

## Code Style & Documentation

### Docstrings: NumPy Style

All functions **must** use NumPy-style docstrings. Follow this template:

```python
def function_name(param1, param2, param3=default):
    """Brief one-line description.
    
    Extended description if needed. Explain what the function does,
    the algorithm used, and any important details.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.
    param3 : type, default default
        Description of param3.
    
    Returns
    -------
    result : type
        Description of return value.
    
    Notes
    -----
    Additional notes about implementation, algorithm, or references.
    
    Examples
    --------
    >>> import numpy as np
    >>> result = function_name(1, 2)
    >>> result
    3
    """
    pass
```

Key points:
- Use `numpy` docstring format (not Google or Sphinx)
- Document all parameters with type and description
- Document all return values
- Include Notes section for algorithms or references
- Include Examples section where practical

### Type Hints

Optionally include type hints in function signatures for clarity:

```python
def function_name(param1: float, param2: int) -> np.ndarray:
    """Brief description."""
    pass
```

## Dependencies

### Allowed Packages (in order of preference)

1. **numpy** — Array operations, linear algebra, random sampling
2. **scipy** — Statistical distributions, optimization, interpolation
3. **random** — Basic random operations (if simpler than numpy)
4. **pandas** — Only for data I/O (reading CSV/TSV), NOT for computation

### Preference Rules

1. **Always prefer NumPy >= SciPy > Pandas** for computation
2. Use `numpy.random.Generator` for random sampling (modern API)
3. Use `scipy.stats` for distributions (e.g., `scipy.stats.expon`, `scipy.stats.gamma`)
4. Use `scipy.optimize` for maximum likelihood and optimization
5. Never use pandas for numerical operations; convert to/from NumPy arrays instead

### Example: Don't use Pandas for computation

❌ **BAD:**
```python
import pandas as pd
df['tau'] = df['infection_times'] / df['removal_times']
```

✅ **GOOD:**
```python
import numpy as np
tau = infection_times / removal_times
```

## Package Structure

```
src/peirrs/
├── __init__.py                      # Main module exports
├── simulate.py                      # Stochastic epidemic simulation
├── estimators.py                    # Core estimators (tau, bayes, bootstrap)
├── utils.py                         # Helper functions (sort, filter, etc.)
├── multitype/
│   ├── __init__.py
│   ├── simulate_multitype.py        # Multitype simulation
│   └── estimators_multitype.py      # Multitype estimators
└── spatial/
    ├── __init__.py
    ├── simulate_spatial.py          # Spatial simulation
    └── estimators_spatial.py        # Spatial estimators
```

## Translation Guidelines

### General R-to-Python Conversions

| R Concept | Python Equivalent |
|-----------|------------------|
| `data.frame` | `np.ndarray` or dict-like structure |
| `list` | `dict` (for named lists) or `list` |
| `vector` | `np.ndarray` (1D) |
| `matrix` | `np.ndarray` (2D) |
| `rexp(n, rate)` | `np.random.exponential(scale=1/rate, size=n)` |
| `rgamma(n, shape, rate)` | `np.random.gamma(shape, scale=1/rate, size=n)` |
| `sample()` | `np.random.choice()` |
| `cumsum()` | `np.cumsum()` |
| `apply()` | NumPy broadcasting or `np.apply_along_axis()` |
| `sort()` | `np.sort()` |

### Data Structures

**For epidemic data**, use dictionaries with clear keys:

```python
epidemic_data = {
    'infection_times': np.array([...]),    # n_individuals
    'removal_times': np.array([...]),      # n_individuals
    'infection_source': np.array([...]),   # n_individuals (indices)
    'individual_id': np.arange(n),
}
```

**For contact/distance data**, use 2D arrays:

```python
distance_matrix = np.ndarray((n_individuals, n_individuals))    # Symmetric
contact_matrix = np.ndarray((n_individuals, n_individuals))     # Binary or weighted
```

### Function Implementation Strategy

1. **Input validation**: Check array shapes, data types, and ranges
2. **Vectorization**: Use NumPy broadcasting instead of loops
3. **Random sampling**: Use `np.random.default_rng()` for reproducibility
4. **Output**: Return NumPy arrays or dicts of arrays (not DataFrames)

Example:

```python
def peirr_tau(infection_times, removal_times, contact_matrix):
    """Tau-based estimator of transmission rate."""
    infection_times = np.asarray(infection_times, dtype=float)
    removal_times = np.asarray(removal_times, dtype=float)
    contact_matrix = np.asarray(contact_matrix, dtype=float)
    
    if infection_times.shape[0] != removal_times.shape[0]:
        raise ValueError("infection_times and removal_times must have same length")
    
    tau = np.zeros(contact_matrix.shape)
    # Vectorized computation
    ...
    return tau
```

## Algorithm References

When translating R functions:

1. **Look at R comments** for algorithm descriptions
2. **Check docstring** (`?function_name` in R) for parameter meanings
3. **Implement the same algorithm**, not a different one—maintain numerical consistency

## Testing

### Test File Organization

```
tests/
├── test_simulate.py         # Tests for simulation functions
├── test_estimators.py       # Tests for core estimators
└── test_multitype.py        # Tests for multitype variants
```

### Test Requirements

- Use `pytest` framework
- Test input validation (bad shapes, negative values, etc.)
- Test output shapes and types
- Test with both small synthetic data

Example:

```python
import pytest
import numpy as np
from peirrs.estimators import peirr_tau

def test_peirr_tau_output_shape():
    """Test that peirr_tau returns matrix of correct shape."""
    n = 10
    infection_times = np.sort(np.random.uniform(0, 10, n))
    removal_times = infection_times + np.random.exponential(2, n)
    contact_matrix = np.random.binomial(1, 0.3, (n, n))
    
    tau = peirr_tau(infection_times, removal_times, contact_matrix)
    assert tau.shape == (n, n)
    assert np.all(tau >= 0)
```

## Key Functions to Implement

### In `simulate.py`
- `simulator()` — Main stochastic epidemic simulator
- `simulate_sem()` — Simplified epidemic model data generation

### In `estimators.py`
- `peirr_tau()` — Transmission rate based on tau method
- `peirr_bayes()` — Bayesian inference with MCMC
- `peirr_bootstrap()` — Bootstrap confidence intervals
- `peirr_removal_rate()` — Removal rate estimator

### In `estimators.py` remove the following functions
- `peirr_pbla_infection_rate()` — PBLA method for infection rate
- `peirr_pbla_both_rates()` — PBLA method for both rates

### In `utils.py`
- `sort_sem()` — Sort epidemic data
- `filter_sem()` — Filter by criteria
- `decomplete_sem()` — Introduce missing data
- `simulate_distance_matrix()` — Generate spatial distance matrices

### In `multitype/` and `spatial/`
- Implement corresponding functions with type/spatial extensions
- Maintain same interface as base functions where possible
- Remove the pbla versions of multitype

## Common Patterns

### Vector Operations (Broadcasting)

```python
# R: vectorized operations
# R: x + y broadcasts if y is scalar or compatible shape

# Python:
result = x + y  # NumPy broadcasts automatically
result = np.add(x, y)  # Explicit function call
```

### Conditional Assignment

```python
# R: ifelse(condition, true_val, false_val)
# Python:
np.where(condition, true_val, false_val)
```

### Matrix Operations

```python
# R: t(x) for transpose
# Python: x.T or np.transpose(x)

# R: x %*% y for matrix multiply
# Python: x @ y or np.dot(x, y)

# R: diag(x) for diagonal matrix
# Python: np.diag(x)
```

### Loops (Avoid when possible)

```python
# R: for(i in 1:n) { ... }
# Python: Use NumPy vectorization, or if necessary:
for i in range(n):
    ...
```

## Consistency Checks

Before marking a function as complete:

1. ✓ Docstring is NumPy style with all parameters documented
2. ✓ Function signature matches R equivalent (same parameters, logical grouping)
3. ✓ No pandas used for computation (only I/O if needed)
4. ✓ Returns NumPy arrays or dicts of arrays
5. ✓ Input validation includes shape and type checks
6. ✓ Algorithm matches R implementation numerically
7. ✓ At least one basic test case exists
8. ✓ No hardcoded magic numbers without explanation

## References

- [NumPy documentation](https://numpy.org/doc/)
- [SciPy documentation](https://scipy.org/doc/)
- [NumPy Style Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- R package: See `R/` folder for original implementations
- Data examples: See `data/` folder

---

**Last Updated:** 2026-02-27
