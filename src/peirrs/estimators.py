"""Epidemic parameter estimation functions."""

from typing import Union, Dict, Any, Callable, Optional
import numpy as np
from . import utils
from . import simulate


def peirr_removal_rate(removals: Union[np.ndarray, list, tuple], 
                       infections: Union[np.ndarray, list, tuple]) -> float:
    """Maximum likelihood estimator of removal rate.
    
    Estimates the removal rate (gamma) as the inverse of the mean infectious period.
    Computes the MLE for exponentially distributed infectious periods.
    
    Parameters
    ----------
    removals : array-like
        Removal times for each individual.
    infections : array-like
        Infection times for each individual.
    
    Returns
    -------
    gamma_estimate : float
        Estimated removal rate. Returns NaN if there are no complete pairs.
    
    Notes
    -----
    The estimator only uses complete pairs where both infection and removal times
    are finite. The formula is:
    
    .. math::
        \hat{\gamma} = \frac{n}{\sum_{i=1}^{n} (r_i - i_i)}
    
    where n is the number of complete pairs and (r_i - i_i) is the infectious period.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import peirr_removal_rate
    >>> infections = np.array([0.0, 1.0, 2.0])
    >>> removals = np.array([2.0, 3.5, 5.0])
    >>> gamma = peirr_removal_rate(removals, infections)
    """
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    
    if removals.shape != infections.shape:
        raise ValueError("removals and infections must have the same shape")
    
    # Find complete pairs (both finite)
    complete_mask = np.isfinite(removals) & np.isfinite(infections)
    removals_complete = removals[complete_mask]
    infections_complete = infections[complete_mask]
    
    if len(removals_complete) == 0:
        return np.nan
    
    # Compute infectious periods
    infectious_periods = removals_complete - infections_complete
    
    # MLE of gamma
    n = len(infectious_periods)
    gamma_hat = n / np.sum(infectious_periods)
    
    return gamma_hat


def peirr_tau(removals: Union[np.ndarray, list, tuple],
              infections: Union[np.ndarray, list, tuple],
              population_size: int,
              lag: float = 0.0) -> Dict[str, float]:
    """Tau-based estimator of infection and removal rates.
    
    Estimates epidemic parameters using expectation-maximization with pairwise
    transmission indicators (tau), handling missing infection and removal times.
    
    Parameters
    ----------
    removals : array-like
        Removal times for each individual. Use np.nan for missing values.
    infections : array-like
        Infection times for each individual. Use np.nan for missing values.
    population_size : int
        Total population size (N).
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : float
            Estimated beta (infection rate per susceptible-infectious pair).
        - 'removal_rate' : float
            Estimated gamma (removal rate).
        - 'effective_number' : float
            Basic reproduction number R0 = beta / gamma.
    
    Raises
    ------
    ValueError
        If insufficient data to estimate removal rate (too many missing values).
    
    Notes
    -----
    The function:
    1. Estimates gamma using maximum likelihood on complete pairs
    2. Computes conditional expectations E[tau_{kj}] for all pairs
    3. Estimates beta using the pseudo-likelihood formula
    
    The estimation assumes exponential infectious periods and handles missing data
    through conditional expectation (tau_moment function).
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import peirr_tau
    >>> np.random.seed(42)
    >>> removals = np.array([2.0, 3.5, 5.0, 4.2])
    >>> infections = np.array([0.5, 1.0, 2.0, 1.5])
    >>> fit = peirr_tau(removals, infections, population_size=100, lag=0)
    >>> fit['removal_rate']
    0.397590...
    >>> fit['effective_number'] > 0
    True
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    
    if removals.shape != infections.shape:
        raise ValueError("removals and infections must have the same shape")
    
    # Filter to keep only individuals with at least one finite time
    or_finite = np.isfinite(removals) | np.isfinite(infections)
    infections = infections[or_finite]
    removals = removals[or_finite]
    
    # Estimate removal rate
    gamma_estim = peirr_removal_rate(removals, infections)
    
    if np.isnan(gamma_estim):
        raise ValueError("Cannot estimate removal rate: insufficient complete pairs")
    
    # Count number of infected
    epidemic_size = np.sum(~np.isnan(removals) | ~np.isnan(infections))
    
    # Check data sufficiency
    if np.sum(np.isnan(removals)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate the removal rate")
    if np.sum(np.isnan(infections)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate the removal rate")
    
    # Find first infected individual (by earliest time)
    removals_temp = removals.copy()
    removals_temp[np.isnan(removals_temp)] = np.inf
    removals_argmin = np.argmin(removals_temp)
    
    infections_temp = infections.copy()
    infections_temp[np.isnan(infections_temp)] = np.inf
    infections_argmin = np.argmin(infections_temp)
    
    if infections[infections_argmin] < removals[removals_argmin]:
        alpha = infections_argmin
    else:
        alpha = removals_argmin
    
    # Sum tau values for all pairs (k, j) where j != alpha
    tau_sum = 0.0
    for j in range(epidemic_size):
        if j == alpha:
            continue
        
        removals_j = removals[j]
        infections_j = infections[j]
        
        for k in range(epidemic_size):
            if k == j:
                continue
            
            removals_k = removals[k]
            infections_k = infections[k]
            
            tau_kj = utils.tau_moment(removals_k, removals_j, infections_k, 
                                     infections_j, gamma_estim, gamma_estim, lag)
            
            if np.isnan(tau_kj):
                print(f"Warning: tau_kj is NaN for pair ({k}, {j}): "
                      f"rk={removals_k}, rj={removals_j}, "
                      f"ik={infections_k}, ij={infections_j}")
            else:
                tau_sum += tau_kj
    
    # Get complete pairs
    complete_mask = ~np.isnan(removals) & ~np.isnan(infections)
    removals_complete = removals[complete_mask]
    infections_complete = infections[complete_mask]
    
    # Calculate sum of infectious periods
    num_not_complete = epidemic_size - len(removals_complete)
    complete_period_sum = (num_not_complete / gamma_estim + 
                          np.sum(removals_complete - infections_complete))
    
    # Estimate infection rate (beta)
    beta_estim = (epidemic_size - 1.0) / (tau_sum + 
                                         (population_size - epidemic_size) * complete_period_sum)
    
    # Scale beta to per-capita rate
    beta_per_capita = beta_estim * population_size
    
    # Effective number (R0)
    r0 = beta_per_capita / gamma_estim
    
    return {
        'infection_rate': beta_per_capita,
        'removal_rate': gamma_estim,
        'effective_number': r0
    }


def peirr_imputed(removals: Union[np.ndarray, list, tuple],
                  infections: Union[np.ndarray, list, tuple],
                  population_size: int,
                  lag: float = 0.0) -> Dict[str, Any]:
    """Infection and removal rate estimator with imputed missing times.
    
    Estimates epidemic parameters by imputing missing times using the mean
    infectious period, then computing tau-based estimates on the completed data.
    
    Parameters
    ----------
    removals : array-like
        Removal times for each individual. Use np.nan for missing values.
    infections : array-like
        Infection times for each individual. Use np.nan for missing values.
    population_size : int
        Total population size (N).
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : float
            Estimated beta (infection rate).
        - 'removal_rate' : float
            Estimated gamma (removal rate).
        - 'effective_number' : float
            Basic reproduction number R0 = beta / gamma.
    
    Raises
    ------
    ValueError
        If insufficient data to estimate removal rate.
    
    Notes
    -----
    Step 1: Estimate gamma using maximum likelihood on complete pairs.
    Step 2: Impute missing times using mean infectious period 1/gamma:
        - Missing infection: imputed = removal - 1/gamma
        - Missing removal: imputed = infection + 1/gamma
    Step 3: Compute tau and estimate beta from completed data.
    
    This is an alternative to peirr_tau() using imputation instead of 
    conditional expectations.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import peirr_imputed
    >>> np.random.seed(42)
    >>> removals = np.array([2.0, np.nan, 5.0, 4.2])
    >>> infections = np.array([0.5, 1.0, np.nan, 1.5])
    >>> fit = peirr_imputed(removals, infections, population_size=100, lag=0)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    
    if removals.shape != infections.shape:
        raise ValueError("removals and infections must have the same shape")
    
    # Filter to keep only individuals with at least one finite time
    or_finite = np.isfinite(removals) | np.isfinite(infections)
    infections = infections[or_finite].copy()
    removals = removals[or_finite].copy()
    
    # Estimate removal rate
    gamma_estim = peirr_removal_rate(removals, infections)
    
    if np.isnan(gamma_estim):
        raise ValueError("Cannot estimate removal rate: insufficient complete pairs")
    
    # Count number of infected
    epidemic_size = np.sum(~np.isnan(removals) | ~np.isnan(infections))
    
    # Check data sufficiency
    if np.sum(np.isnan(removals)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate the removal rate")
    if np.sum(np.isnan(infections)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate the removal rate")
    
    # Impute missing times using mean infectious period
    mean_period = 1.0 / gamma_estim
    
    # Missing infections: impute as removal - 1/gamma
    missing_infection_mask = np.isnan(infections) & np.isfinite(removals)
    infections[missing_infection_mask] = removals[missing_infection_mask] - mean_period
    
    # Missing removals: impute as infection + 1/gamma
    missing_removal_mask = np.isnan(removals) & np.isfinite(infections)
    removals[missing_removal_mask] = infections[missing_removal_mask] + mean_period
    
    # Find first infected individual
    alpha = np.argmin(infections)
    
    # Sum tau values for all pairs (k, j) where j != alpha
    tau_sum = 0.0
    for j in range(epidemic_size):
        if j == alpha:
            continue
        
        removals_j = removals[j]
        infections_j = infections[j]
        
        for k in range(epidemic_size):
            if k == j:
                continue
            
            removals_k = removals[k]
            infections_k = infections[k]
            
            tau_kj = utils.tau_moment(removals_k, removals_j, infections_k, 
                                     infections_j, gamma_estim, gamma_estim, lag)
            
            if np.isnan(tau_kj):
                print(f"Warning: tau_kj is NaN for pair ({k}, {j}): "
                      f"rk={removals_k}, rj={removals_j}, "
                      f"ik={infections_k}, ij={infections_j}")
            else:
                tau_sum += tau_kj
    
    # Calculate sum of infectious periods (all complete now after imputation)
    complete_period_sum = np.sum(removals - infections)
    
    # Estimate infection rate (beta)
    beta_estim = (epidemic_size - 1.0) / (tau_sum + 
                                         (population_size - epidemic_size) * complete_period_sum)
    
    # Scale beta to per-capita rate
    beta_per_capita = beta_estim * population_size
    
    # Effective number (R0)
    r0 = beta_per_capita / gamma_estim
    
    # Count originally complete pairs (for diagnostics)
    complete_mask_original = ~np.isnan(removals) & ~np.isnan(infections)
    num_complete = np.sum(complete_mask_original)
    
    # Non-infected sum and count
    not_infected_sum = complete_period_sum
    num_not_infected = population_size - epidemic_size
    
    return {
        'infection_rate': beta_per_capita,
        'removal_rate': gamma_estim,
        'effective_number': r0,
    }


def peirr_bootstrap(num_bootstrap: int,
                   beta: float,
                   gamma: float,
                   population_size: int,
                   epidemic_size: int,
                   prop_complete: float,
                   prop_infection_missing: float,
                   peirr: Callable = None,
                   num_renewals: int = 1,
                   lag: float = 0.0,
                   within: float = 0.1,
                   **kwargs) -> Dict[str, np.ndarray]:
    """Parametric bootstrap for infection and removal rate estimates.
    
    Performs a parametric bootstrap procedure to estimate variability in
    epidemic parameter estimates. Repeatedly simulates epidemic data under
    the specified model parameters and applies a user-specified estimator
    function to obtain bootstrap confidence intervals.
    
    Parameters
    ----------
    num_bootstrap : int
        Number of bootstrap replicates to perform. Must be positive.
    beta : float
        Infection rate parameter used for simulation.
    gamma : float
        Removal rate parameter used for simulation.
    population_size : int
        Total population size.
    epidemic_size : int
        Target number of infected individuals in each simulation.
    prop_complete : float
        Expected proportion of complete cases (both times observed).
        Must be in [0, 1].
    prop_infection_missing : float
        Probability that an infection time is missing (conditional on
        not being a complete case). Must be in [0, 1].
    peirr : callable, default peirr_tau
        Estimator function that takes removals, infections, population_size,
        lag, and additional keyword arguments via **kwargs.
        Should return a dict with 'infection_rate' and 'removal_rate' keys.
    num_renewals : int, default 1
        Number of renewals (reinfection stages) in simulation.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    within : float, default 0.1
        Fractional tolerance around epidemic_size. Bootstrap replicates
        will have epidemic sizes in the range:
        [epidemic_size * (1 - within), epidemic_size * (1 + within)].
    **kwargs
        Additional arguments passed to the estimator function peirr().
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            Array of length num_bootstrap + 1 containing infection rate estimates.
            First element is the true value used for simulation (beta).
            Remaining elements are estimates from each bootstrap replicate.
        - 'removal_rate' : ndarray
            Array of length num_bootstrap + 1 containing removal rate estimates.
            First element is the true value used for simulation (gamma).
            Remaining elements are estimates from each bootstrap replicate.
    
    Raises
    ------
    ValueError
        If num_bootstrap <= 0, epidemic_size <= 0, or invalid probability values.
    
    Notes
    -----
    For each bootstrap iteration:
    1. Simulates an epidemic using the specified parameters (beta, gamma)
    2. Introduces missing data according to prop_complete and prop_infection_missing
    3. Applies the estimator function to obtain parameter estimates
    4. Stores the resulting estimates
    
    The first row of results contains the true simulation parameters (beta, gamma),
    allowing direct comparison of estimator performance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import peirr_bootstrap, peirr_tau
    >>> np.random.seed(42)
    >>> results = peirr_bootstrap(
    ...     num_bootstrap=10,
    ...     beta=0.5,
    ...     gamma=0.3,
    ...     population_size=500,
    ...     epidemic_size=100,
    ...     prop_complete=0.8,
    ...     prop_infection_missing=0.2,
    ...     peirr=peirr_tau
    ... )
    >>> results['infection_rate'][0]  # True beta
    0.5
    >>> results['removal_rate'][0]  # True gamma
    0.3
    >>> len(results['infection_rate'])  # num_bootstrap + 1
    11
    """
    # Use peirr_tau as default estimator
    if peirr is None:
        peirr = peirr_tau
    
    # Input validation
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if epidemic_size <= 0:
        raise ValueError("epidemic_size must be positive")
    if not (0 <= prop_complete <= 1):
        raise ValueError("prop_complete must be in [0, 1]")
    if not (0 <= prop_infection_missing <= 1):
        raise ValueError("prop_infection_missing must be in [0, 1]")
    
    # Initialize storage for results
    infection_rates = np.zeros(num_bootstrap + 1)
    removal_rates = np.zeros(num_bootstrap + 1)
    
    # First row stores true values
    infection_rates[0] = beta
    removal_rates[0] = gamma
    
    # Calculate epidemic size bounds
    min_epidemic_size = int(np.ceil((1 - within) * epidemic_size))
    max_epidemic_size = int(np.floor((1 + within) * epidemic_size))
    
    # Bootstrap loop
    for b in range(1, num_bootstrap + 1):
        # Simulate epidemic
        epidemic_data = simulate.simulator(
            beta=beta,
            gamma=gamma,
            population_size=population_size,
            num_renewals=num_renewals,
            lag=lag,
            prop_complete=prop_complete,
            prop_infection_missing=prop_infection_missing,
            min_epidemic_size=min_epidemic_size,
            max_epidemic_size=max_epidemic_size
        )
        
        # Extract times from simulated data
        matrix_time = epidemic_data['matrix_time']
        infections = matrix_time[:, 0]
        removals = matrix_time[:, 1]
        
        # Apply estimator function
        try:
            estimate = peirr(
                removals=removals,
                infections=infections,
                population_size=population_size,
                lag=lag,
                **kwargs
            )
            
            # Store results
            infection_rates[b] = estimate['infection_rate']
            removal_rates[b] = estimate['removal_rate']
        
        except Exception as e:
            # Warn but continue with NaN if estimation fails
            print(f"Warning: Estimation failed at bootstrap iteration {b}: {e}")
            infection_rates[b] = np.nan
            removal_rates[b] = np.nan
    
    return {
        'infection_rate': infection_rates,
        'removal_rate': removal_rates
    }


def bayes_complete(removals: Union[np.ndarray, list, tuple],
                  infections: Union[np.ndarray, list, tuple],
                  population_size: int,
                  beta_init: float = 1.0,
                  gamma_init: float = 1.0,
                  beta_shape: float = 1.0,
                  gamma_shape: float = 1.0,
                  num_iter: int = 10000,
                  num_renewals: int = 1,
                  lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Posterior sampling for complete epidemic data using Gibbs sampling.
    
    Performs Gibbs sampling to generate posterior samples of infection and
    removal rates when infection times and removal times are fully observed.
    Assumes gamma-distributed priors on both rate parameters.
    
    Parameters
    ----------
    removals : array-like
        Removal times for each infected individual (observed, no NaNs).
    infections : array-like
        Infection times for each infected individual (observed, no NaNs).
    population_size : int
        Total population size N.
    beta_init : float, default 1.0
        Initial (prior mean) estimate for infection rate. Used to compute
        the rate parameter of the gamma prior.
    gamma_init : float, default 1.0
        Initial (prior mean) estimate for removal rate.
    beta_shape : float, default 1.0
        Shape parameter (alpha) for gamma prior on beta.
    gamma_shape : float, default 1.0
        Shape parameter (alpha) for gamma prior on gamma.
    num_iter : int, default 10000
        Number of MCMC iterations to generate.
    num_renewals : int, default 1
        Number of exponential stages in infectious period distribution.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious delay).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            MCMC samples of beta (infection rate), shape (num_iter,)
        - 'removal_rate' : ndarray
            MCMC samples of gamma (removal rate), shape (num_iter,)
    
    Raises
    ------
    ValueError
        If epidemic_size > population_size or data length mismatch.
    
    Notes
    -----
    Uses Gibbs sampling where:
    
    - Beta | data ~ Gamma(shape = beta_shape + n - 1,
                          rate = beta_rate + tau_sum)
    - Gamma | data ~ Gamma(shape = gamma_shape + n * m,
                           rate = gamma_rate + period_sum)
    
    where n is epidemic size, m is num_renewals, and tau_sum is the sum of
    pairwise transmission durations.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import bayes_complete
    >>> np.random.seed(42)
    >>> infections = np.array([0.0, 1.0, 1.5, 2.5])
    >>> removals = np.array([2.0, 2.5, 3.5, 4.0])
    >>> samples = bayes_complete(removals, infections, population_size=100,
    ...                          beta_init=1.0, gamma_init=0.8, num_iter=100)
    >>> samples['infection_rate'].shape
    (100,)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    
    if removals.shape != infections.shape:
        raise ValueError("removals and infections must have same length")
    
    epidemic_size = len(removals)
    
    if epidemic_size > population_size:
        raise ValueError("Epidemic size cannot be larger than population size")
    
    # Prior rate parameters from initial estimates
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_rate_scaled = beta_rate / population_size
    
    # Compute tau matrix: pairwise transmission durations
    # tau[j, k] = min(infection[k] - lag, removal[j]) - min(infection[k] - lag, infection[j])
    tau_matrix = np.zeros((epidemic_size, population_size))
    
    for j in range(epidemic_size):
        for k in range(epidemic_size):
            tau_matrix[j, k] = min(infections[k] - lag, removals[j]) - \
                              min(infections[k] - lag, infections[j])
    
    # For uninfected individuals, tau equals full infectious period
    if epidemic_size < population_size:
        tau_matrix[:, epidemic_size:] = removals[:, np.newaxis] - infections[:, np.newaxis]
    
    tau_sum = np.sum(tau_matrix)
    period_sum = np.sum(removals - infections)
    
    # Gibbs sampling
    beta_samples = np.random.gamma(
        shape=beta_shape + epidemic_size - 1,
        scale=1.0 / (beta_rate_scaled + tau_sum),
        size=num_iter
    )
    
    gamma_samples = np.random.gamma(
        shape=gamma_shape + epidemic_size * num_renewals,
        scale=1.0 / (gamma_rate + period_sum),
        size=num_iter
    )
    
    return {
        'infection_rate': beta_samples * population_size,
        'removal_rate': gamma_samples
    }


def _check_if_epidemic(removals: np.ndarray,
                       infections: np.ndarray,
                       lag: float) -> bool:
    """Check if infection/removal configuration is consistent with epidemic dynamics.
    
    Validates that the configuration could arise from a valid epidemic where:
    - Individual j can only be infected by individual k if k infected before j
    - And k must still be infectious when j gets infected
    
    Parameters
    ----------
    removals : ndarray
        Removal times for infected individuals.
    infections : ndarray
        Infection times for infected individuals.
    lag : float
        Exposure-to-infectious lag.
    
    Returns
    -------
    valid : bool
        True if configuration is consistent with epidemic dynamics.
        Allows up to one individual (e.g., index case) to have no potential infectors.
    """
    epidemic_size = len(removals)
    
    # Build chi matrix: for each individual j, count how many could have infected them
    chi_matrix = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit = infections[j] - lag
        
        # Count individuals k where: k infected before infection_limit AND
        # k was still infectious when j got infected (removals[k] > infection_limit)
        # These are individuals who COULD have infected j
        could_infect = np.sum((infections[:epidemic_size] < infection_limit) & 
                             (removals > infection_limit))
        
        chi_matrix[j] = could_infect
    
    # Filter to keep only individuals with zero potential infectors
    chi_zero = chi_matrix[chi_matrix == 0]
    
    # Return False only if more than one individual has no potential infectors
    # (allows for exactly one index case with no infector)
    if len(chi_zero) > 1:
        return False
    else:
        return True


def _update_infected_prob(removals: np.ndarray,
                         infections: np.ndarray,
                         infections_proposed: np.ndarray,
                         beta_shape: float,
                         beta_rate: float,
                         lag: float) -> float:
    """Compute log Metropolis-Hastings acceptance ratio for infection time proposal.
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    infections_proposed : ndarray
        Proposed infection times (population_size,).
    beta_shape : float
        Beta prior shape parameter.
    beta_rate : float
        Beta prior rate parameter (scaled by population).
    lag : float
        Exposure-to-infectious lag.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio = log(p(proposal)/p(current)).
    """
    epidemic_size = len(removals)
    population_size = len(infections)
    
    if len(infections_proposed) != population_size:
        raise ValueError("Proposed and current infection vectors must have same length")
    
    # Compute tau matrices for current and proposed configurations
    tau_matrix = np.zeros((epidemic_size, population_size))
    tau_proposed = np.zeros((epidemic_size, population_size))
    
    for j in range(epidemic_size):
        for k in range(population_size):
            tau_matrix[j, k] = min(infections[k] - lag, removals[j]) - \
                              min(infections[k] - lag, infections[j])
            tau_proposed[j, k] = min(infections_proposed[k] - lag, removals[j]) - \
                                min(infections_proposed[k] - lag, infections_proposed[j])
    
    # Compute chi: for each j, count how many k (k != j) could have infected j
    chi_current = np.zeros(epidemic_size)
    chi_proposed = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit_current = infections[j] - lag
        infection_limit_proposed = infections_proposed[j] - lag
        
        # Count infectious individuals at time of j's infection (current)
        chi_current[j] = np.sum((infections[:epidemic_size] < infection_limit_current) & 
                               (removals > infection_limit_current))
        
        # Count infectious individuals at time of j's infection (proposed)
        chi_proposed[j] = np.sum((infections_proposed[:epidemic_size] < infection_limit_proposed) & 
                                (removals > infection_limit_proposed))
    
    # Remove zeros for log calculation
    chi_current_nz = chi_current[chi_current > 0]
    chi_proposed_nz = chi_proposed[chi_proposed > 0]
    
    # Log acceptance ratio
    ell_ratio = np.sum(np.log(chi_proposed_nz)) - np.sum(np.log(chi_current_nz))
    ell_ratio += (beta_shape + epidemic_size - 1) * \
                (np.log(beta_rate + np.sum(tau_matrix)) - 
                 np.log(beta_rate + np.sum(tau_proposed)))
    
    return ell_ratio


def _update_removal_prob(removals: np.ndarray,
                        infections: np.ndarray,
                        removals_proposed: np.ndarray,
                        beta_shape: float,
                        beta_rate: float,
                        lag: float) -> float:
    """Compute log Metropolis-Hastings acceptance ratio for removal time proposal.
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    removals_proposed : ndarray
        Proposed removal times (epidemic_size,).
    beta_shape : float
        Beta prior shape parameter.
    beta_rate : float
        Beta prior rate parameter (scaled by population).
    lag : float
        Exposure-to-infectious lag.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio = log(p(proposal)/p(current)).
    """
    epidemic_size = len(removals)
    population_size = len(infections)
    
    if len(removals_proposed) != epidemic_size:
        raise ValueError("Proposed and current removal vectors must have same length")
    
    # Compute tau matrices for current and proposed configurations
    tau_matrix = np.zeros((epidemic_size, population_size))
    tau_proposed = np.zeros((epidemic_size, population_size))
    
    for j in range(epidemic_size):
        for k in range(population_size):
            tau_matrix[j, k] = min(infections[k] - lag, removals[j]) - \
                              min(infections[k] - lag, infections[j])
            tau_proposed[j, k] = min(infections[k] - lag, removals_proposed[j]) - \
                                min(infections[k] - lag, infections[j])
    
    # Compute chi: for each j, count how many k (k != j) could have infected j
    chi_current = np.zeros(epidemic_size)
    chi_proposed = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit = infections[j] - lag
        
        chi_current[j] = np.sum((infections[:epidemic_size] < infection_limit) & 
                               (removals > infection_limit))
        chi_proposed[j] = np.sum((infections[:epidemic_size] < infection_limit) & 
                                (removals_proposed > infection_limit))
    
    # Remove zeros for log calculation
    chi_current_nz = chi_current[chi_current > 0]
    chi_proposed_nz = chi_proposed[chi_proposed > 0]
    
    # Log acceptance ratio
    ell_ratio = np.sum(np.log(chi_proposed_nz)) - np.sum(np.log(chi_current_nz))
    ell_ratio += (beta_shape + epidemic_size - 1) * \
                (np.log(beta_rate + np.sum(tau_matrix)) - 
                 np.log(beta_rate + np.sum(tau_proposed)))
    
    return ell_ratio


def peirr_bayes(removals: Union[np.ndarray, list, tuple],
               infections: Union[np.ndarray, list, tuple],
               population_size: int,
               beta_init: float = 1.0,
               gamma_init: float = 1.0,
               beta_shape: float = 1.0,
               gamma_shape: float = 1.0,
               num_iter: int = 500,
               num_update: int = 10,
               num_tries: int = 5,
               num_print: int = 100,
               update_gamma: bool = False,
               num_renewals: int = 1,
               lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Bayesian inference for epidemic parameters with partial/missing data.
    
    Performs MCMC with data augmentation to estimate infection and removal rates
    when infection and removal times may be partially observed. Uses alternating
    Gibbs and Metropolis-Hastings steps.
    
    Parameters
    ----------
    removals : array-like
        Removal times, with np.nan indicating unobserved times.
    infections : array-like
        Infection times, with np.nan indicating unobserved times.
    population_size : int
        Total population size N.
    beta_init : float, default 1.0
        Initial estimate for infection rate (prior mean).
    gamma_init : float, default 1.0
        Initial estimate for removal rate (prior mean).
    beta_shape : float, default 1.0
        Shape parameter for gamma prior on beta.
    gamma_shape : float, default 1.0
        Shape parameter for gamma prior on gamma.
    num_iter : int, default 500
        Number of MCMC iterations.
    num_update : int, default 10
        Number of Metropolis-Hastings updates per iteration for each
        missing time type (infection and removal).
    num_tries : int, default 5
        Maximum attempts to generate valid epidemic configuration in MH step.
    num_print : int, default 100
        Iteration interval for printing progress messages.
    update_gamma : bool, default False
        If True, update gamma via Gibbs sampling each iteration.
        If False, keep gamma fixed at gamma_init.
    num_renewals : int, default 1
        Number of exponential stages in infectious period distribution.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            MCMC samples of beta (infection rate), shape (num_iter,)
        - 'removal_rate' : ndarray
            MCMC samples of gamma (removal rate), shape (num_iter,)
        - 'prop_infection_updated' : ndarray
            Acceptance rate for infection time proposals per iteration
        - 'prop_removal_updated' : ndarray
            Acceptance rate for removal time proposals per iteration
    
    Notes
    -----
    The algorithm alternates between:
    1. Gibbs sample beta from posterior given current augmented data
    2. Metropolis-Hastings updates for missing infection times
    3. Metropolis-Hastings updates for missing removal times
    4. Gibbs sample gamma (if update_gamma=True)
    
    If all data is complete, calls bayes_complete() for efficiency.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.estimators import peirr_bayes
    >>> np.random.seed(42)
    >>> infections = np.array([0.0, 1.0, np.nan, 2.5])
    >>> removals = np.array([2.0, np.nan, 3.5, 4.0])
    >>> fit = peirr_bayes(removals, infections, population_size=100,
    ...                   beta_init=1.0, gamma_init=0.8,
    ...                   num_iter=50, num_print=100)
    >>> fit['infection_rate'].shape
    (50,)
    >>> fit['prop_infection_updated'].shape
    (50,)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    
    if removals.shape != infections.shape:
        raise ValueError("removals and infections must have same length")
    
    # Setup priors
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_curr = beta_rate / population_size
    beta_rate_scaled = beta_rate / population_size
    
    if update_gamma:
        gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
    else:
        gamma_curr = gamma_init
    
    # Count truly infected (have at least one time)
    epidemic_size = np.sum(np.isfinite(infections) | np.isfinite(removals))
    
    if np.sum(~np.isnan(infections)) == epidemic_size and \
       np.sum(~np.isnan(removals)) == epidemic_size:
        # Complete data: use bayes_complete
        infections_complete = infections[np.isfinite(infections)]
        removals_complete = removals[np.isfinite(removals)]
        out = bayes_complete(removals_complete, infections_complete, population_size,
                            beta_init=beta_init, gamma_init=gamma_init,
                            beta_shape=beta_shape, gamma_shape=gamma_shape,
                            num_iter=num_iter, num_renewals=num_renewals, lag=lag)
        
        return {
            'infection_rate': out['infection_rate'],
            'removal_rate': out['removal_rate'],
            'prop_infection_updated': np.full(num_iter, np.nan),
            'prop_removal_updated': np.full(num_iter, np.nan)
        }
    
    # Data augmentation initialization
    num_nan_infections = np.sum(np.isnan(infections))
    num_nan_removals = np.sum(np.isnan(removals))
    num_update_infections = min(num_nan_infections, num_update) if num_nan_infections > 0 else 1
    num_update_removals = min(num_nan_removals, num_update) if num_nan_removals > 0 else 1
    
    infections_aug = infections.copy()
    removals_aug = removals.copy()
    
    # Initialize missing infections: removal - Erlang(num_renewals, gamma_curr)
    if num_nan_infections > 0:
        infections_aug[np.isnan(infections)] = removals[np.isnan(infections)] - \
            np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr, size=num_nan_infections)
    
    # Initialize missing removals: infection + Erlang(num_renewals, gamma_curr)
    if num_nan_removals > 0:
        removals_aug[np.isnan(removals)] = infections[np.isnan(removals)] + \
            np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr, size=num_nan_removals)
    
    while not _check_if_epidemic(removals_aug, infections_aug, lag):
        if update_gamma:
            gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
        else:
            gamma_curr = gamma_init
        
        if num_nan_infections > 0:
            infections_aug[np.isnan(infections)] = removals[np.isnan(infections)] - \
                np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr, size=num_nan_infections)
        if num_nan_removals > 0:
            removals_aug[np.isnan(removals)] = infections[np.isnan(removals)] + \
                np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr, size=num_nan_removals)
    
    # Pad with infinities for non-infected
    if len(infections_aug) < population_size:
        infections_aug = np.concatenate([infections_aug, np.full(population_size - len(infections_aug), np.inf)])
    
    # Storage for MCMC samples
    storage = np.full((4, num_iter), np.nan)
    
    # MCMC iterations
    for k in range(num_iter):

        # Get current epidemic-sized arrays
        removals_curr = removals_aug[:epidemic_size]
        infections_curr = infections_aug[:epidemic_size]
        
        # 1. Gibbs sample beta
        tau_matrix = np.zeros((epidemic_size, population_size))
        for j in range(epidemic_size):
            for i in range(population_size):
                tau_matrix[j, i] = min(infections_aug[i] - lag, removals_curr[j]) - \
                                  min(infections_aug[i] - lag, infections_curr[j])
        
        beta_curr = np.random.gamma(
            shape=beta_shape + epidemic_size - 1,
            scale=1.0 / (beta_rate_scaled + np.sum(tau_matrix))
        )
        storage[0, k] = beta_curr * population_size
        
        # 2. Metropolis-Hastings for infection times
        successes = 0
        if num_nan_infections > 0:
            nan_indices = np.where(np.isnan(infections))[0]
            
            for update_step in range(num_update_infections):
                if len(nan_indices) == 1:
                    l = nan_indices[0]
                else:
                    l = np.random.choice(nan_indices)
                
                ctr = 0
                new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr)
                infections_proposed = infections_aug.copy()
                infections_proposed[l] = new_infection
                
                while (not _check_if_epidemic(removals_aug[:epidemic_size], 
                                             infections_proposed[:epidemic_size], lag) and 
                       ctr < num_tries):
                    ctr += 1
                    new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals, 
                                                                      scale=1.0 / gamma_curr)
                    infections_proposed[l] = new_infection
                
                if _check_if_epidemic(removals_aug[:epidemic_size], 
                                     infections_proposed[:epidemic_size], lag):
                    log_accept = _update_infected_prob(removals_aug[:epidemic_size],
                                                      infections_aug,
                                                      infections_proposed,
                                                      beta_shape,
                                                      beta_rate_scaled,
                                                      lag)
                    accept_prob = min(1.0, np.exp(log_accept))
                    if np.random.uniform() < accept_prob:
                        infections_aug[l] = new_infection
                        successes += 1
        
        if num_nan_infections > 0:
            storage[2, k] = successes / num_update_infections
        
        # 3. Metropolis-Hastings for removal times
        successes = 0
        if num_nan_removals > 0:
            nan_indices = np.where(np.isnan(removals))[0]
            
            for update_step in range(num_update_removals):
                if len(nan_indices) == 1:
                    l = nan_indices[0]
                else:
                    l = np.random.choice(nan_indices)
                
                ctr = 0
                new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals, scale=1.0 / gamma_curr)
                removals_proposed = removals_aug.copy()
                removals_proposed[l] = new_removal
                
                while (not _check_if_epidemic(removals_proposed[:epidemic_size],
                                             infections_aug[:epidemic_size], lag) and 
                       ctr < num_tries):
                    ctr += 1
                    new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals,
                                                                      scale=1.0 / gamma_curr)
                    removals_proposed[l] = new_removal
                
                if _check_if_epidemic(removals_proposed[:epidemic_size],
                                     infections_aug[:epidemic_size], lag):
                    log_accept = _update_removal_prob(removals_aug[:epidemic_size],
                                                     infections_aug,
                                                     removals_proposed,
                                                     beta_shape,
                                                     beta_rate_scaled,
                                                     lag)
                    accept_prob = min(1.0, np.exp(log_accept))
                    if np.random.uniform() < accept_prob:
                        removals_aug[l] = new_removal
                        successes += 1
        
        if num_nan_removals > 0:
            storage[3, k] = successes / num_update_removals
        
        # 4. Gibbs sample gamma (optional)
        if update_gamma:
            period_sum = np.sum(removals_aug[:epidemic_size] - infections_aug[:epidemic_size])
            gamma_curr = np.random.gamma(
                shape=gamma_shape + epidemic_size * num_renewals,
                scale=1.0 / (gamma_rate + period_sum)
            )
        
        storage[1, k] = gamma_curr
        
        # Progress printing
        if (k + 1) % num_print == 0:
            print(f"Completed iteration {k + 1} out of {num_iter}")
    
    return {
        'infection_rate': storage[0, :],
        'removal_rate': storage[1, :],
        'prop_infection_updated': storage[2, :],
        'prop_removal_updated': storage[3, :]
    }