"""Estimation functions for spatial epidemic models."""

from typing import Callable, Dict, Union
import numpy as np
from .. import utils, estimators as core_est


def peirr_tau_spatial(removals: Union[np.ndarray, list, tuple],
                     infections: Union[np.ndarray, list, tuple],
                     population_size: int,
                     kernel_spatial: Callable,
                     matrix_distance: Union[np.ndarray, list],
                     lag: float = 0.0) -> Dict[str, float]:
    """Spatial tau-based estimator for infection and removal rates.
    
    Estimates infection and removal rates in spatial epidemic models where the
    transmission rate depends on spatial distance between individuals via
    a kernel function.
    
    Parameters
    ----------
    removals : array-like
        Removal times for infected individuals. Use np.nan for missing values.
    infections : array-like
        Infection times for infected individuals. Use np.nan for missing values.
    population_size : int
        Total population size (infected + susceptible).
    kernel_spatial : callable
        Symmetric, non-negative function of distance modulating transmission rate.
        E.g., ``lambda d: np.exp(-2*d)`` or ``lambda d: 1/(1+d**2)``.
    matrix_distance : array-like
        Distance matrix, shape (population_size, population_size). Should be
        symmetric with zeros on diagonal. First `epidemic_size` rows/columns
        correspond to infected individuals, remaining to susceptibles.
    lag : float, default 0.0
        Fixed incubation period.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : float
            Estimated beta (spatial infection rate)
        - 'removal_rate' : float
            Estimated gamma (removal rate)
    
    Raises
    ------
    ValueError
        If insufficient complete pairs to estimate removal rate.
    
    Notes
    -----
    Uses expectation-maximization (EM) approach:
    
    1. Estimate gamma via MLE on complete pairs
    2. Estimate beta via EM using spatially-weighted tau terms
    
    The spatial weighting is incorporated through the kernel function applied
    to pairwise distances.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import peirr_tau_spatial
    >>> np.random.seed(42)
    >>> from scipy.spatial.distance import pdist, squareform
    >>> pop = 100
    >>> infections = np.array([0.5, 1.0, 2.0])
    >>> removals = np.array([2.0, 3.5, 4.0])
    >>> coords = np.random.uniform(0, 1, (pop, 2))
    >>> D = squareform(pdist(coords))
    >>> kernel = lambda d: np.exp(-2*d)
    >>> est = peirr_tau_spatial(removals, infections, pop, kernel, D)
    >>> est['infection_rate'] > 0 and est['removal_rate'] > 0
    True
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    matrix_distance = np.asarray(matrix_distance, dtype=float)
    
    # Filter to finite values
    or_finite = np.isfinite(removals) | np.isfinite(infections)
    infections = infections[or_finite]
    removals = removals[or_finite]
    
    # Estimate removal rate
    gamma_estimate = core_est.peirr_removal_rate(removals, infections)
    
    # Check for complete pairs
    epidemic_size = np.sum(~np.isnan(removals) | ~np.isnan(infections))
    num_complete_removals = np.sum(~np.isnan(removals))
    num_complete_infections = np.sum(~np.isnan(infections))
    
    if num_complete_removals >= epidemic_size or num_complete_infections >= epidemic_size:
        raise ValueError("Insufficient complete pairs to estimate removal rate")
    
    # Find first infected individual
    alpha_r = np.nanargmin(removals)
    alpha_i = np.nanargmin(infections)
    
    if infections[alpha_i] < removals[alpha_r]:
        alpha = alpha_i
    else:
        alpha = alpha_r
    
    # Compute spatial tau terms
    tau_sum = 0.0
    for j in range(epidemic_size):
        if j == alpha:
            continue
        
        removal_j = removals[j]
        infection_j = infections[j]
        
        for k in range(epidemic_size):
            if k == j:
                continue
            
            removal_k = removals[k]
            infection_k = infections[k]
            
            # Compute tau weighted by spatial kernel
            tau_kj = utils.tau_moment(
                removal_k, removal_j, infection_k, infection_j,
                gamma_estimate, gamma_estimate, lag
            )
            
            # Apply spatial weighting
            spatial_weight = kernel_spatial(matrix_distance[k, j])
            tau_sum += tau_kj * spatial_weight
    
    # Compute susceptible contribution with spatial weighting
    if epidemic_size == population_size:
        not_infected_sum = 0.0
    else:
        not_infected_sum = 0.0
        for j in range(epidemic_size):
            period = removals[j] - infections[j]
            if np.isnan(period):
                period = 1.0 / gamma_estimate
            
            for k in range(epidemic_size, population_size):
                spatial_weight = kernel_spatial(matrix_distance[j, k])
                not_infected_sum += period * spatial_weight
    
    # Maximum likelihood estimate
    beta_estimate = (epidemic_size - 1) / (tau_sum + not_infected_sum)
    
    return {
        'infection_rate': beta_estimate * population_size,
        'removal_rate': gamma_estimate
    }


def bayes_complete_spatial(removals: Union[np.ndarray, list, tuple],
                          infections: Union[np.ndarray, list, tuple],
                          population_size: int,
                          kernel_spatial: Callable,
                          matrix_distance: Union[np.ndarray, list],
                          beta_init: float = 1.0,
                          gamma_init: float = 1.0,
                          beta_shape: float = 1.0,
                          gamma_shape: float = 1.0,
                          num_iter: int = 10000,
                          num_renewals: int = 1,
                          lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Bayesian Gibbs sampling for complete spatial epidemic data.
    
    Posterior sampling of infection and removal rates via Gibbs sampling
    when all infection and removal times are fully observed, with spatial
    distance-dependent transmission.
    
    Parameters
    ----------
    removals : array-like
        Removal times (no NaNs).
    infections : array-like
        Infection times (no NaNs).
    population_size : int
        Total population size.
    kernel_spatial : callable
        Spatial kernel function.
    matrix_distance : array-like
        Distance matrix.
    beta_init : float, default 1.0
        Initial infection rate estimate.
    gamma_init : float, default 1.0
        Initial removal rate estimate.
    beta_shape : float, default 1.0
        Gamma prior shape for beta.
    gamma_shape : float, default 1.0
        Gamma prior shape for gamma.
    num_iter : int, default 10000
        Number of MCMC iterations.
    num_renewals : int, default 1
        Erlang shape parameter.
    lag : float, default 0.0
        Incubation lag.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            Posterior samples of infection rate (length num_iter)
        - 'removal_rate' : ndarray
            Posterior samples of removal rate (length num_iter)
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import bayes_complete_spatial
    >>> np.random.seed(42)
    >>> from scipy.spatial.distance import pdist, squareform
    >>> removals = np.array([2.0, 3.5, 4.0])
    >>> infections = np.array([0.5, 1.0, 1.5])
    >>> pop = 100
    >>> coords = np.random.uniform(0, 1, (pop, 2))
    >>> D = squareform(pdist(coords))
    >>> kernel = lambda d: np.exp(-2*d)
    >>> samples = bayes_complete_spatial(removals, infections, pop, kernel, D,
    ...                                 num_iter=100)
    >>> samples['infection_rate'].shape
    (100,)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    matrix_distance = np.asarray(matrix_distance, dtype=float)
    
    # Setup
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_rate_scaled = beta_rate / population_size
    epidemic_size = len(removals)
    
    # Extend infections to population size
    infections_augmented = np.concatenate([
        infections,
        np.full(population_size - epidemic_size, np.inf)
    ])
    
    # Compute spatial tau matrix
    tau_matrix = np.zeros((epidemic_size, population_size))
    for j in range(epidemic_size):
        for i in range(population_size):
            tau_val = (np.minimum(infections_augmented[i] - lag, removals[j]) -
                      np.minimum(infections_augmented[i] - lag, infections[j]))
            spatial_weight = kernel_spatial(matrix_distance[j, i])
            tau_matrix[j, i] = tau_val * spatial_weight
    
    tau_sum = np.sum(tau_matrix)
    period_sum = np.sum(removals - infections)
    
    # Storage for samples
    beta_samples = np.zeros(num_iter)
    gamma_samples = np.zeros(num_iter)
    
    # Gibbs sampling
    beta_samples = np.random.gamma(
        shape=beta_shape + epidemic_size - 1,
        scale=1.0 / (beta_rate_scaled + tau_sum),
        size=num_iter
    ) * population_size
    
    gamma_samples = np.random.gamma(
        shape=gamma_shape + epidemic_size * num_renewals,
        scale=1.0 / (gamma_rate + period_sum),
        size=num_iter
    )
    
    return {
        'infection_rate': beta_samples,
        'removal_rate': gamma_samples
    }


def _check_if_epidemic_spatial(removals: np.ndarray,
                              infections: np.ndarray,
                              lag: float) -> bool:
    """Check if spatial epidemic configuration is valid.
    
    Parameters
    ----------
    removals : ndarray
        Removal times.
    infections : ndarray
        Infection times.
    lag : float
        Incubation lag.
    
    Returns
    -------
    valid : bool
        True if configuration is epidemic-valid.
        Allows up to one individual (e.g., index case) to have no potential infectors.
    """
    epidemic_size = len(removals)
    
    # Build chi vector: for each individual j, count how many could have infected them
    chi_vector = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        chi_count = np.sum(
            (infections[:epidemic_size] < (infections[j] - lag)) &
            (removals > (infections[j] - lag))
        )
        chi_vector[j] = chi_count
    
    # Count individuals with zero potential infectors
    chi_zero = chi_vector[chi_vector == 0]
    
    # Return False only if more than one individual has no potential infectors
    # (allows for exactly one index case with no infector)
    return len(chi_zero) <= 1


def _update_infected_prob_spatial(removals: np.ndarray,
                                 infections: np.ndarray,
                                 infections_proposed: np.ndarray,
                                 beta_shape: float,
                                 beta_rate: float,
                                 lag: float,
                                 kernel_spatial: Callable,
                                 matrix_distance: np.ndarray) -> float:
    """Log MH acceptance ratio for infection time proposal (spatial).
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    infections_proposed : ndarray
        Proposed infection times (population_size,).
    beta_shape : float
        Beta prior shape.
    beta_rate : float
        Beta prior rate (scaled).
    lag : float
        Incubation lag.
    kernel_spatial : callable
        Spatial kernel.
    matrix_distance : ndarray
        Distance matrix.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio.
    """
    epidemic_size = len(removals)
    population_size = len(infections)
    
    # Compute tau matrices
    tau_matrix = np.zeros((epidemic_size, population_size))
    for j in range(epidemic_size):
        for i in range(population_size):
            tau_val = (np.minimum(infections[i] - lag, removals[j]) -
                      np.minimum(infections[i] - lag, infections[j]))
            spatial_weight = kernel_spatial(matrix_distance[j, i])
            tau_matrix[j, i] = tau_val * spatial_weight
    
    tau_matrix_proposed = np.zeros((epidemic_size, population_size))
    for j in range(epidemic_size):
        for i in range(population_size):
            tau_val = (np.minimum(infections_proposed[i] - lag, removals[j]) -
                      np.minimum(infections_proposed[i] - lag, infections_proposed[j]))
            spatial_weight = kernel_spatial(matrix_distance[j, i])
            tau_matrix_proposed[j, i] = tau_val * spatial_weight
    
    # Compute chi probabilities
    chi_prob = np.zeros(epidemic_size)
    for j in range(epidemic_size):
        chi_val = np.sum(
            (infections[:epidemic_size] < (infections[j] - lag)) &
            (removals > (infections[j] - lag))
        )
        for k in range(epidemic_size):
            spatial_weight = kernel_spatial(matrix_distance[j, k])
            chi_prob[j] += (
                (infections[k] < (infections[j] - lag)).astype(float) *
                (removals[j] > (infections[j] - lag)).astype(float) *
                spatial_weight
            )
    chi_prob = chi_prob[chi_prob > 0]
    
    chi_prob_proposed = np.zeros(epidemic_size)
    for j in range(epidemic_size):
        for k in range(epidemic_size):
            spatial_weight = kernel_spatial(matrix_distance[j, k])
            chi_prob_proposed[j] += (
                (infections_proposed[k] < (infections_proposed[j] - lag)).astype(float) *
                (removals[j] > (infections_proposed[j] - lag)).astype(float) *
                spatial_weight
            )
    chi_prob_proposed = chi_prob_proposed[chi_prob_proposed > 0]
    
    ell_ratio = (np.sum(np.log(chi_prob_proposed)) - np.sum(np.log(chi_prob)) +
                (beta_shape + epidemic_size - 1) *
                (np.log(beta_rate + np.sum(tau_matrix)) - 
                 np.log(beta_rate + np.sum(tau_matrix_proposed))))
    
    return ell_ratio


def _update_removal_prob_spatial(removals: np.ndarray,
                                infections: np.ndarray,
                                removals_proposed: np.ndarray,
                                beta_shape: float,
                                beta_rate: float,
                                lag: float,
                                kernel_spatial: Callable,
                                matrix_distance: np.ndarray) -> float:
    """Log MH acceptance ratio for removal time proposal (spatial).
    
    Similar to _update_infected_prob_spatial but for removal times.
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    removals_proposed : ndarray
        Proposed removal times (epidemic_size,).
    beta_shape : float
        Beta prior shape.
    beta_rate : float
        Beta prior rate (scaled).
    lag : float
        Incubation lag.
    kernel_spatial : callable
        Spatial kernel.
    matrix_distance : ndarray
        Distance matrix.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio.
    """
    epidemic_size = len(removals)
    population_size = len(infections)
    
    # Compute tau matrices
    tau_matrix = np.zeros((epidemic_size, population_size))
    for j in range(epidemic_size):
        for i in range(population_size):
            tau_val = (np.minimum(infections[i] - lag, removals[j]) -
                      np.minimum(infections[i] - lag, infections[j]))
            spatial_weight = kernel_spatial(matrix_distance[j, i])
            tau_matrix[j, i] = tau_val * spatial_weight
    
    tau_matrix_proposed = np.zeros((epidemic_size, population_size))
    for j in range(epidemic_size):
        for i in range(population_size):
            tau_val = (np.minimum(infections[i] - lag, removals_proposed[j]) -
                      np.minimum(infections[i] - lag, infections[j]))
            spatial_weight = kernel_spatial(matrix_distance[j, i])
            tau_matrix_proposed[j, i] = tau_val * spatial_weight
    
    # Compute chi probabilities
    chi_prob = np.zeros(epidemic_size)
    for j in range(epidemic_size):
        for k in range(epidemic_size):
            spatial_weight = kernel_spatial(matrix_distance[j, k])
            chi_prob[j] += (
                (infections[k] < (infections[j] - lag)).astype(float) *
                (removals[j] > (infections[j] - lag)).astype(float) *
                spatial_weight
            )
    chi_prob = chi_prob[chi_prob > 0]
    
    chi_prob_proposed = np.zeros(epidemic_size)
    for j in range(epidemic_size):
        for k in range(epidemic_size):
            spatial_weight = kernel_spatial(matrix_distance[j, k])
            chi_prob_proposed[j] += (
                (infections[k] < (infections[j] - lag)).astype(float) *
                (removals_proposed[j] > (infections[j] - lag)).astype(float) *
                spatial_weight
            )
    chi_prob_proposed = chi_prob_proposed[chi_prob_proposed > 0]
    
    ell_ratio = (np.sum(np.log(chi_prob_proposed)) - np.sum(np.log(chi_prob)) +
                (beta_shape + epidemic_size - 1) *
                (np.log(beta_rate + np.sum(tau_matrix)) - 
                 np.log(beta_rate + np.sum(tau_matrix_proposed))))
    
    return ell_ratio


def peirr_bayes_spatial(removals: Union[np.ndarray, list, tuple],
                       infections: Union[np.ndarray, list, tuple],
                       population_size: int,
                       kernel_spatial: Callable,
                       matrix_distance: Union[np.ndarray, list],
                       beta_init: float = 1.0,
                       gamma_init: float = 1.0,
                       beta_shape: float = 1.0,
                       gamma_shape: float = 1.0,
                       num_update: int = 10,
                       num_iter: int = 500,
                       num_print: int = 100,
                       num_tries: int = 5,
                       update_gamma: bool = False,
                       num_renewals: int = 1,
                       lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Bayesian MCMC for spatial epidemic parameters with partial data.
    
    Data augmentation MCMC for infection and removal rates in spatial epidemics
    with missing infection and removal times. Transmission rates depend on
    distance between individuals via a kernel function.
    
    Parameters
    ----------
    removals : array-like
        Removal times with np.nan for missing values.
    infections : array-like
        Infection times with np.nan for missing values.
    population_size : int
        Total population size.
    kernel_spatial : callable
        Spatial kernel function of distance.
    matrix_distance : array-like
        Distance matrix, shape (population_size, population_size).
    beta_init : float, default 1.0
        Initial infection rate estimate.
    gamma_init : float, default 1.0
        Initial removal rate estimate.
    beta_shape : float, default 1.0
        Gamma prior shape for beta.
    gamma_shape : float, default 1.0
        Gamma prior shape for gamma.
    num_update : int, default 10
        MH update attempts per iteration.
    num_iter : int, default 500
        Number of MCMC iterations.
    num_print : int, default 100
        Iteration print frequency.
    num_tries : int, default 5
        Max attempts for valid epidemic proposal.
    update_gamma : bool, default False
        If True, update gamma via Gibbs each iteration.
    num_renewals : int, default 1
        Erlang shape parameter.
    lag : float, default 0.0
        Incubation lag.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            MCMC samples of infection rate (length num_iter)
        - 'removal_rate' : ndarray
            MCMC samples of removal rate (length num_iter)
        - 'prop_infection_updated' : ndarray
            Acceptance rates for infection time proposals
        - 'prop_removal_updated' : ndarray
            Acceptance rates for removal time proposals
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import peirr_bayes_spatial
    >>> np.random.seed(42)
    >>> from scipy.spatial.distance import pdist, squareform
    >>> removals = np.array([2.0, np.nan, 4.0])
    >>> infections = np.array([0.5, 1.0, np.nan])
    >>> pop = 100
    >>> coords = np.random.uniform(0, 1, (pop, 2))
    >>> D = squareform(pdist(coords))
    >>> kernel = lambda d: np.exp(-2*d)
    >>> fit = peirr_bayes_spatial(removals, infections, pop, kernel, D,
    ...                          num_iter=50, num_print=100)
    >>> fit['infection_rate'].shape
    (50,)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    matrix_distance = np.asarray(matrix_distance, dtype=float)
    
    # Setup
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_rate_scaled = beta_rate / population_size
    
    epidemic_size = np.sum(np.isfinite(infections) | np.isfinite(removals))
    
    if update_gamma:
        gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
    else:
        gamma_curr = gamma_init
    
    # Check if data is complete
    if (np.sum(np.isfinite(infections)) == epidemic_size and
        np.sum(np.isfinite(removals)) == epidemic_size):
        # Complete data: use bayes_complete_spatial
        out = bayes_complete_spatial(
            removals, infections, population_size,
            kernel_spatial, matrix_distance,
            beta_init=beta_init, gamma_init=gamma_init,
            beta_shape=beta_shape, gamma_shape=gamma_shape,
            num_iter=num_iter, num_renewals=num_renewals, lag=lag
        )
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
    
    # Initialize missing infections
    if num_nan_infections > 0:
        nan_mask = np.isnan(infections)
        infections_aug[nan_mask] = (removals[nan_mask] -
                                   np.random.gamma(shape=num_renewals, 
                                                 scale=1.0 / gamma_curr,
                                                 size=num_nan_infections))
    
    # Initialize missing removals
    if num_nan_removals > 0:
        nan_mask = np.isnan(removals)
        removals_aug[nan_mask] = (infections[nan_mask] +
                                 np.random.gamma(shape=num_renewals,
                                               scale=1.0 / gamma_curr,
                                               size=num_nan_removals))
    
    # Ensure valid epidemic
    while not _check_if_epidemic_spatial(removals_aug[:epidemic_size],
                                        infections_aug[:epidemic_size], lag):
        if update_gamma:
            gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
        else:
            gamma_curr = gamma_init
        
        if num_nan_infections > 0:
            nan_mask = np.isnan(infections)
            infections_aug[nan_mask] = (removals[nan_mask] -
                                       np.random.gamma(shape=num_renewals,
                                                     scale=1.0 / gamma_curr,
                                                     size=num_nan_infections))
        
        if num_nan_removals > 0:
            nan_mask = np.isnan(removals)
            removals_aug[nan_mask] = (infections[nan_mask] +
                                     np.random.gamma(shape=num_renewals,
                                                   scale=1.0 / gamma_curr,
                                                   size=num_nan_removals))
    
    # Pad infections to population size
    infections_aug = np.concatenate([infections_aug, np.full(population_size - len(infections_aug), np.inf)])
    
    # Storage
    beta_samples = np.zeros(num_iter)
    gamma_samples = np.zeros(num_iter)
    updated_infections = np.zeros(num_iter)
    updated_removals = np.zeros(num_iter)
    
    # MCMC loop
    for k in range(num_iter):
        # Compute spatial tau matrix
        tau_matrix = np.zeros((epidemic_size, population_size))
        for j in range(epidemic_size):
            for i in range(population_size):
                tau_val = (np.minimum(infections_aug[i] - lag, removals_aug[j]) -
                          np.minimum(infections_aug[i] - lag, infections_aug[j]))
                spatial_weight = kernel_spatial(matrix_distance[j, i])
                tau_matrix[j, i] = tau_val * spatial_weight
        
        # Gibbs sample beta
        tau_sum = np.sum(tau_matrix)
        beta_samples[k] = np.random.gamma(
            shape=beta_shape + epidemic_size - 1,
            scale=1.0 / (beta_rate_scaled + tau_sum)
        ) * population_size
        
        # MH for infection times
        successes = 0
        if num_nan_infections > 0:
            nan_indices = np.where(np.isnan(infections))[0]
            
            for update_step in range(num_update_infections):
                if len(nan_indices) == 1:
                    l = nan_indices[0]
                else:
                    l = np.random.choice(nan_indices)
                
                ctr = 0
                new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals,
                                                                 scale=1.0 / gamma_curr)
                infections_proposed = infections_aug.copy()
                infections_proposed[l] = new_infection
                
                while (not _check_if_epidemic_spatial(removals_aug[:epidemic_size],
                                                     infections_proposed[:epidemic_size], lag) and
                       ctr < num_tries):
                    ctr += 1
                    new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals,
                                                                     scale=1.0 / gamma_curr)
                    infections_proposed[l] = new_infection
                
                if _check_if_epidemic_spatial(removals_aug[:epidemic_size],
                                             infections_proposed[:epidemic_size], lag):
                    log_accept = _update_infected_prob_spatial(
                        removals_aug[:epidemic_size], infections_aug, infections_proposed,
                        beta_shape, beta_rate_scaled, lag, kernel_spatial, matrix_distance
                    )
                    accept_prob = min(1.0, np.exp(log_accept))
                    if np.random.uniform() < accept_prob:
                        infections_aug[l] = new_infection
                        successes += 1
        
        updated_infections[k] = successes / num_update_infections if num_nan_infections > 0 else np.nan
        
        # MH for removal times
        successes = 0
        if num_nan_removals > 0:
            nan_indices = np.where(np.isnan(removals))[0]
            
            for update_step in range(num_update_removals):
                if len(nan_indices) == 1:
                    l = nan_indices[0]
                else:
                    l = np.random.choice(nan_indices)
                
                ctr = 0
                new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals,
                                                                 scale=1.0 / gamma_curr)
                removals_proposed = removals_aug.copy()
                removals_proposed[l] = new_removal
                
                while (not _check_if_epidemic_spatial(removals_proposed[:epidemic_size],
                                                     infections_aug[:epidemic_size], lag) and
                       ctr < num_tries):
                    ctr += 1
                    new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals,
                                                                     scale=1.0 / gamma_curr)
                    removals_proposed[l] = new_removal
                
                if _check_if_epidemic_spatial(removals_proposed[:epidemic_size],
                                             infections_aug[:epidemic_size], lag):
                    log_accept = _update_removal_prob_spatial(
                        removals_aug[:epidemic_size], infections_aug, removals_proposed,
                        beta_shape, beta_rate_scaled, lag, kernel_spatial, matrix_distance
                    )
                    accept_prob = min(1.0, np.exp(log_accept))
                    if np.random.uniform() < accept_prob:
                        removals_aug[l] = new_removal
                        successes += 1
        
        updated_removals[k] = successes / num_update_removals if num_nan_removals > 0 else np.nan
        
        # Gibbs sample gamma
        period_sum = np.sum(removals_aug[:epidemic_size] - infections_aug[:epidemic_size])
        gamma_samples[k] = np.random.gamma(
            shape=gamma_shape + epidemic_size * num_renewals,
            scale=1.0 / (gamma_rate + period_sum)
        )
        gamma_curr = gamma_samples[k]
        
        # Progress printing
        if (k + 1) % num_print == 0:
            print(f"Completed iteration {k + 1} out of {num_iter}")
    
    return {
        'infection_rate': beta_samples,
        'removal_rate': gamma_samples,
        'prop_infection_updated': updated_infections,
        'prop_removal_updated': updated_removals
    }
