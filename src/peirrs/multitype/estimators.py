"""Estimation functions for multitype epidemic models."""

from typing import Union, Dict, Tuple, List
import numpy as np
from .. import utils
from . import simulate


def peirr_tau_multitype(removals: Union[np.ndarray, list, tuple],
                       infections: Union[np.ndarray, list, tuple],
                       removal_classes: Union[np.ndarray, list, tuple],
                       infection_classes: Union[np.ndarray, list, tuple],
                       infection_class_sizes: Union[np.ndarray, list, tuple],
                       lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Class-specific tau-based estimator for multitype epidemic parameters.
    
    Estimates class-specific infection and removal rates in multitype epidemic
    models using tau-based expectation-maximization. Each individual has two
    independent class assignments: infection class (infectiousness) and removal
    class (recovery rate).
    
    Parameters
    ----------
    removals : array-like
        Removal times for each individual. Use np.nan for missing values.
    infections : array-like
        Infection times for each individual. Use np.nan for missing values.
    removal_classes : array-like
        Removal class assignment for each individual (0-indexed).
        Must have same length as removals/infections.
    infection_classes : array-like
        Infection class assignment for each individual (0-indexed).
        Must have same length as removals/infections.
    infection_class_sizes : array-like
        Total population size for each infection class (including both
        infected and susceptible individuals).
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            Class-specific infection rates (beta_1, ..., beta_m)
        - 'removal_rate' : ndarray
            Class-specific removal rates (gamma_1, ..., gamma_l)
    
    Raises
    ------
    ValueError
        If insufficient complete data to estimate removal rates.
    
    Notes
    -----
    Algorithm:
    
    1. Estimate class-specific removal rates (gamma_l) using MLE on complete
       pairs within each removal class:
       
       .. math::
           \\hat{\\gamma}_l = \\frac{n_l}{\\sum_{i \\in class\\ l} (r_i - i_i)}
    
    2. Estimate class-specific infection rates (beta_m) using EM with
       pairwise transmission indicators (tau).
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import peirr_tau_multitype
    >>> np.random.seed(42)
    >>> removals = np.array([2.0, 3.5, 2.5, 4.0])
    >>> infections = np.array([0.5, 1.0, 1.0, 2.0])
    >>> removal_classes = np.array([0, 1, 0, 1])
    >>> infection_classes = np.array([0, 0, 1, 1])
    >>> fit = peirr_tau_multitype(
    ...     removals, infections, removal_classes, infection_classes,
    ...     infection_class_sizes=[2, 2], lag=0.0
    ... )
    >>> fit['infection_rate'].shape
    (2,)
    >>> fit['removal_rate'].shape
    (2,)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    removal_classes = np.asarray(removal_classes, dtype=int)
    infection_classes = np.asarray(infection_classes, dtype=int)
    infection_class_sizes = np.asarray(infection_class_sizes, dtype=int)
    
    # Keep only individuals with at least one finite time
    or_finite = np.isfinite(removals) | np.isfinite(infections)
    removals = removals[or_finite]
    infections = infections[or_finite]
    removal_classes = removal_classes[or_finite]
    infection_classes = infection_classes[or_finite]
    
    # Epidemic size
    epidemic_size = len(removals)
    
    if np.sum(np.isnan(removals)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate removal rates")
    if np.sum(np.isnan(infections)) >= epidemic_size:
        raise ValueError("There are no complete case periods to estimate removal rates")
    
    # Find first infected individual
    removals_temp = removals.copy()
    removals_temp[np.isnan(removals_temp)] = np.inf
    alpha_r = np.argmin(removals_temp)
    
    infections_temp = infections.copy()
    infections_temp[np.isnan(infections_temp)] = np.inf
    alpha_i = np.argmin(infections_temp)
    
    if infections[alpha_i] < removals[alpha_r]:
        alpha = alpha_i
    else:
        alpha = alpha_r
    
    # Get unique classes
    removal_classes_unique = np.unique(removal_classes)
    infection_classes_unique = np.unique(infection_classes)
    
    # Initialize rate storage
    removal_rates = []
    removal_full_sizes = []
    removal_partial_sums = []
    
    # Step 1: Estimate class-specific removal rates
    for removal_class in removal_classes_unique:
        # Find individuals in this removal class
        class_mask = removal_classes == removal_class
        removals_class = removals[class_mask]
        infections_class = infections[class_mask]
        
        # Find complete pairs in this class
        complete_mask = np.isfinite(removals_class) & np.isfinite(infections_class)
        removals_complete = removals_class[complete_mask]
        infections_complete = infections_class[complete_mask]
        
        # Estimate removal rate
        removal_full_sizes.append(len(removals_complete))
        rate_estim = len(removals_complete) / np.sum(removals_complete - infections_complete)
        removal_rates.append(rate_estim)
        
        # Compute expected infectious period sum (complete + imputed)
        num_incomplete = len(removals_class) - len(removals_complete)
        partial_sum = np.sum(removals_complete - infections_complete) + num_incomplete / rate_estim
        removal_partial_sums.append(partial_sum)
    
    removal_rates = np.array(removal_rates)
    removal_partial_sums = np.array(removal_partial_sums)
    complete_period_sum = np.sum(removal_partial_sums)
    
    # Step 2: Estimate class-specific infection rates
    infection_rates = []
    
    for infection_class_idx, infection_class in enumerate(infection_classes_unique):
        tau_sum = 0.0
        
        # Count infected in this infection class
        class_mask = infection_classes == infection_class
        num_infected = np.sum(class_mask)
        num_not_infected = infection_class_sizes[infection_class_idx] - num_infected
        
        # If alpha is in this class, exclude it
        if infection_classes[alpha] == infection_class:
            num_infected -= 1
        
        # Sum tau over all relevant pairs
        # j is the recipient, k is the infector
        for j in range(epidemic_size):
            if j == alpha:
                continue
            
            infection_class_j = infection_classes[j]
            if infection_class_j != infection_class:
                continue
            
            removal_class_j = removal_classes[j]
            removal_j = removals[j]
            infection_j = infections[j]
            
            # Index of removal class j in unique removal classes
            removal_class_j_idx = np.where(removal_classes_unique == removal_class_j)[0][0]
            rate_j = removal_rates[removal_class_j_idx]
            
            # Sum tau over all potential infectors k
            for k in range(epidemic_size):
                if k == j:
                    continue
                
                removal_class_k = removal_classes[k]
                removal_k = removals[k]
                infection_k = infections[k]
                
                # Index of removal class k
                removal_class_k_idx = np.where(removal_classes_unique == removal_class_k)[0][0]
                rate_k = removal_rates[removal_class_k_idx]
                
                # Compute tau for this pair
                tau_kj = utils.tau_moment(removal_k, removal_j, infection_k, 
                                         infection_j, rate_k, rate_j, lag)
                
                if not np.isnan(tau_kj):
                    tau_sum += tau_kj
        
        # Estimate infection rate for this class
        rate_estim = num_infected / (tau_sum + num_not_infected * complete_period_sum)
        infection_rates.append(rate_estim)
    
    infection_rates = np.array(infection_rates)
    
    # Scale infection rates by total population size
    total_pop = np.sum(infection_class_sizes)
    
    return {
        'infection_rate': infection_rates * total_pop,
        'removal_rate': removal_rates
    }


def peirr_bootstrap_multitype(num_bootstrap: int,
                             beta: Union[np.ndarray, list, tuple],
                             gamma: Union[np.ndarray, list, tuple],
                             infection_class_sizes: Union[np.ndarray, list, tuple],
                             removal_class_sizes: Union[np.ndarray, list, tuple],
                             epidemic_size: int,
                             prop_complete: float,
                             prop_infection_missing: float,
                             num_renewals: int = 1,
                             lag: float = 0.0,
                             within: float = 0.1,
                             **kwargs) -> Dict[str, np.ndarray]:
    """Parametric bootstrap for multitype epidemic parameter estimation.
    
    Performs parametric bootstrap to estimate sampling variability of class-specific
    infection and removal rates in multitype epidemics.
    
    Parameters
    ----------
    num_bootstrap : int
        Number of bootstrap replicates. Must be positive.
    beta : array-like
        Infection rates for each infection class.
    gamma : array-like
        Removal rates for each removal class.
    infection_class_sizes : array-like
        Population sizes for each infection class.
    removal_class_sizes : array-like
        Population sizes for each removal class.
    epidemic_size : int
        Target number of infected individuals in each bootstrap replicate.
    prop_complete : float
        Proportion of complete infection-removal pairs observed.
        Must be in (0, 1].
    prop_infection_missing : float
        Probability infection time is missing (if not complete).
        Must be in [0, 1].
    num_renewals : int, default 1
        Number of exponential stages in infectious period distribution.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag).
    within : float, default 0.1
        Acceptable fractional deviation from target epidemic size.
        Bootstrap replicates will have size in
        [epidemic_size * (1 - within), epidemic_size * (1 + within)].
    **kwargs
        Additional arguments passed to peirr_tau_multitype().
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            Bootstrap samples of infection rates, shape (num_beta, num_bootstrap + 1).
            First column is true values, remaining are bootstrap estimates.
        - 'removal_rate' : ndarray
            Bootstrap samples of removal rates, shape (num_gamma, num_bootstrap + 1).
            First column is true values, remaining are bootstrap estimates.
    
    Raises
    ------
    ValueError
        If num_bootstrap <= 0 or epidemic_size <= 0.
    
    Notes
    -----
    Each bootstrap replicate:
    1. Simulates epidemic using simulator_multitype()
    2. Extracts infection/removal times and class assignments
    3. Estimates rates using peirr_tau_multitype()
    4. Stores results
    
    First column of output contains true simulation parameters, allowing direct
    comparison of estimator performance.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import peirr_bootstrap_multitype
    >>> np.random.seed(42)
    >>> results = peirr_bootstrap_multitype(
    ...     num_bootstrap=10,
    ...     beta=[1.5, 2.0],
    ...     gamma=[0.8, 1.2],
    ...     infection_class_sizes=[50, 50],
    ...     removal_class_sizes=[50, 50],
    ...     epidemic_size=50,
    ...     prop_complete=0.8,
    ...     prop_infection_missing=0.5
    ... )
    >>> results['infection_rate'].shape
    (2, 11)
    >>> results['removal_rate'].shape
    (2, 11)
    """
    # Convert inputs
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    infection_class_sizes = np.asarray(infection_class_sizes, dtype=int)
    removal_class_sizes = np.asarray(removal_class_sizes, dtype=int)
    
    num_beta = len(beta)
    num_gamma = len(gamma)
    
    # Input validation
    if num_bootstrap <= 0:
        raise ValueError("num_bootstrap must be positive")
    if epidemic_size <= 0:
        raise ValueError("epidemic_size must be positive")
    
    # Initialize storage
    infection_storage = np.zeros((num_beta, num_bootstrap + 1))
    removal_storage = np.zeros((num_gamma, num_bootstrap + 1))
    
    # First column: true values
    infection_storage[:, 0] = beta
    removal_storage[:, 0] = gamma
    
    # Epidemic size bounds
    min_epidemic_size = int(np.ceil((1 - within) * epidemic_size))
    max_epidemic_size = int(np.floor((1 + within) * epidemic_size))
    
    # Bootstrap loop
    for b in range(1, num_bootstrap + 1):
        # Simulate epidemic
        epidemic_data = simulate.simulator_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes,
            num_renewals=num_renewals,
            lag=lag,
            prop_complete=prop_complete,
            prop_infection_missing=prop_infection_missing,
            min_epidemic_size=min_epidemic_size,
            max_epidemic_size=max_epidemic_size
        )
        
        # Extract data
        matrix_time = epidemic_data['matrix_time']
        infections = matrix_time[:, 0]
        removals = matrix_time[:, 1]
        infection_classes = matrix_time[:, 2]
        removal_classes = matrix_time[:, 4]
        
        # Estimate rates
        try:
            estimate = peirr_tau_multitype(
                removals=removals,
                infections=infections,
                removal_classes=removal_classes,
                infection_classes=infection_classes,
                infection_class_sizes=infection_class_sizes,
                lag=lag,
                **kwargs
            )
            
            infection_storage[:, b] = estimate['infection_rate']
            removal_storage[:, b] = estimate['removal_rate']
        
        except Exception as e:
            print(f"Warning: Estimation failed at bootstrap iteration {b}: {e}")
            infection_storage[:, b] = np.nan
            removal_storage[:, b] = np.nan
    
    return {
        'infection_rate': infection_storage,
        'removal_rate': removal_storage
    }


def bayes_complete_multitype(removals: Union[np.ndarray, list, tuple],
                             infections: Union[np.ndarray, list, tuple],
                             removal_classes: Union[np.ndarray, list, tuple],
                             infection_classes: Union[np.ndarray, list, tuple],
                             infection_class_sizes: Union[np.ndarray, list, tuple],
                             beta_init: Union[np.ndarray, list, tuple],
                             beta_shape: Union[np.ndarray, list, tuple],
                             gamma_init: Union[np.ndarray, list, tuple],
                             gamma_shape: Union[np.ndarray, list, tuple],
                             num_iter: int = 10000,
                             num_renewals: int = 1,
                             lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Gibbs sampling for complete multitype epidemic data.
    
    Posterior sampling of class-specific infection and removal rates when
    all infection and removal times are fully observed in a multitype SIR model.
    
    Parameters
    ----------
    removals : array-like
        Removal times for each infected individual (no NaNs).
    infections : array-like
        Infection times for each infected individual (no NaNs).
        If longer than removals, padded with NaN for non-infected.
    removal_classes : array-like
        Removal class assignment for each person (0 or 1-indexed for classes).
    infection_classes : array-like
        Infection class assignment for each person.
    infection_class_sizes : array-like
        Total population size for each infection class.
    beta_init : array-like
        Initial (prior mean) estimates for infection rates, one per infection class.
    beta_shape : array-like
        Gamma prior shape parameters for infection rates.
    gamma_init : array-like
        Initial estimates for removal rates, one per removal class.
    gamma_shape : array-like
        Gamma prior shape parameters for removal rates.
    num_iter : int, default 10000
        Number of MCMC iterations.
    num_renewals : int, default 1
        Number of exponential stages in infectious period.
    lag : float, default 0.0
        Fixed incubation period.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            Posterior samples of class-specific infection rates, 
            shape (num_infection_classes, num_iter)
        - 'removal_rate' : ndarray
            Posterior samples of class-specific removal rates,
            shape (num_removal_classes, num_iter)
    
    Notes
    -----
    Uses conjugate Gibbs updates for both parameters conditioned on class-specific
    data subsets. Extends bayes_complete to multitype setting.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import bayes_complete_multitype
    >>> np.random.seed(42)
    >>> removals = np.array([2.0, 3.5, 2.5, 4.0])
    >>> infections = np.array([0.5, 1.0, 1.0, 2.0])
    >>> removal_classes = np.array([0, 1, 0, 1])
    >>> infection_classes = np.array([0, 0, 1, 1])
    >>> samples = bayes_complete_multitype(
    ...     removals, infections, removal_classes, infection_classes,
    ...     infection_class_sizes=[2, 2],
    ...     beta_init=[1.0, 1.0], beta_shape=[1.0, 1.0],
    ...     gamma_init=[0.8, 0.8], gamma_shape=[1.0, 1.0],
    ...     num_iter=100
    ... )
    >>> samples['infection_rate'].shape
    (2, 100)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    removal_classes = np.asarray(removal_classes, dtype=int)
    infection_classes = np.asarray(infection_classes, dtype=int)
    infection_class_sizes = np.asarray(infection_class_sizes, dtype=int)
    beta_init = np.asarray(beta_init, dtype=float)
    beta_shape = np.asarray(beta_shape, dtype=float)
    gamma_init = np.asarray(gamma_init, dtype=float)
    gamma_shape = np.asarray(gamma_shape, dtype=float)
    
    # Validation
    if len(removals) != len(removal_classes):
        raise ValueError("Removal times and classes must have same length")
    if len(infections) != len(infection_classes):
        raise ValueError("Infection times and classes must have same length")
    if len(infections) != len(removals):
        raise ValueError("Infection and removal vectors must have same length")
    if len(np.unique(removal_classes)) != len(gamma_shape):
        raise ValueError("Incorrect removal rate shape parameter size")
    if len(np.unique(infection_classes)) != len(beta_shape):
        raise ValueError("Incorrect infection rate shape parameter size")
    if len(beta_init) != len(beta_shape):
        raise ValueError("Initial infection rates and shapes must have same length")
    if len(gamma_init) != len(gamma_shape):
        raise ValueError("Initial removal rates and shapes must have same length")
    
    # Setup
    epidemic_size = len(removals)
    population_size = np.sum(infection_class_sizes)
    num_gamma = len(np.unique(removal_classes))
    num_beta = len(np.unique(infection_classes))
    
    # Extend infections to population size if needed
    if len(infections) < population_size:
        infections_full = np.concatenate([infections, np.full(population_size - epidemic_size, np.nan)])
        infection_classes_full = infection_classes.copy()
        
        # Extend infection classes
        for class_id in np.unique(infection_classes):
            class_size = infection_class_sizes[class_id] if class_id < len(infection_class_sizes) else 0
            current_count = np.sum(infection_classes == class_id)
            if current_count < class_size:
                new_count = class_size - current_count
                infection_classes_full = np.concatenate([
                    infection_classes_full,
                    np.full(new_count, class_id, dtype=int)
                ])
    else:
        infections_full = infections
        infection_classes_full = infection_classes
    
    # Prior rate parameters
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_rate_scaled = beta_rate / population_size
    
    # Initialize storage
    beta_samples = np.zeros((num_beta, num_iter))
    gamma_samples = np.zeros((num_gamma, num_iter))
    
    # Get unique classes sorted
    unique_removal_classes = np.unique(removal_classes)
    unique_infection_classes = np.unique(infection_classes)
    
    # Sample removal rates (class-specific)
    for class_idx, removal_class in enumerate(unique_removal_classes):
        class_mask = removal_classes == removal_class
        removals_class = removals[class_mask]
        infections_class = infections[:epidemic_size][class_mask]
        
        # Compute periods for complete observations
        periods = removals_class - infections_class
        period_sum = np.sum(periods[np.isfinite(periods)])
        num_periods = np.sum(np.isfinite(periods))
        
        gamma_samples[class_idx, :] = np.random.gamma(
            shape=gamma_shape[class_idx] + num_periods * num_renewals,
            scale=1.0 / (gamma_rate[class_idx] + period_sum),
            size=num_iter
        )
    
    # Compute tau matrix for all pairs
    tau_matrix = np.zeros((epidemic_size, epidemic_size))
    for j in range(epidemic_size):
        for i in range(epidemic_size):
            tau_matrix[j, i] = min(infections[i] - lag, removals[j]) - \
                              min(infections[i] - lag, infections[j])
    
    period_sum = np.sum(removals - infections)
    
    # Sample infection rates (class-specific)
    for class_idx, infection_class in enumerate(unique_infection_classes):
        class_mask = infection_classes[:epidemic_size] == infection_class
        tau_class = tau_matrix[:, class_mask]
        tau_sum = np.sum(tau_class)
        
        num_infected = np.sum(class_mask)
        num_not_infected = infection_class_sizes[class_idx] - num_infected
        
        beta_samples[class_idx, :] = np.random.gamma(
            shape=beta_shape[class_idx] + num_infected,
            scale=1.0 / (beta_rate_scaled[class_idx] + tau_sum + 
                        num_not_infected * period_sum),
            size=num_iter
        )
    
    return {
        'infection_rate': beta_samples * population_size,
        'removal_rate': gamma_samples
    }


def _check_if_epidemic_multitype(removals: np.ndarray,
                                 infections: np.ndarray,
                                 lag: float) -> bool:
    """Check if infection/removal configuration is valid epidemic (multitype version).
    
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
    
    # Build chi vector: for each individual j, count how many could have infected them
    chi_vector = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit = infections[j] - lag
        chi = np.sum((infections[:epidemic_size] < infection_limit) & 
                    (removals > infection_limit))
        chi_vector[j] = chi
    
    # Count individuals with zero potential infectors
    chi_zero = chi_vector[chi_vector == 0]
    
    # Return False only if more than one individual has no potential infectors
    # (allows for exactly one index case with no infector)
    return len(chi_zero) <= 1


def _update_infected_prob_multitype(removals: np.ndarray,
                                   infections: np.ndarray,
                                   infections_proposed: np.ndarray,
                                   infection_classes: np.ndarray,
                                   beta_shape: np.ndarray,
                                   beta_rate: np.ndarray,
                                   lag: float) -> float:
    """Log MH acceptance ratio for infection time proposal (multitype).
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    infections_proposed : ndarray
        Proposed infection times (population_size,).
    infection_classes : ndarray
        Infection class assignment.
    beta_shape : ndarray
        Beta prior shapes.
    beta_rate : ndarray
        Beta prior rates.
    lag : float
        Exposure lag.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio.
    """
    epidemic_size = len(removals)
    unique_classes = np.unique(infection_classes[:epidemic_size])
    num_beta = len(unique_classes)
    
    # Compute log likelihood ratio for current configuration
    tau_rolling = 0.0
    for class_num, infection_class in enumerate(unique_classes):
        class_mask = infection_classes[:epidemic_size] == infection_class
        tau_sum = 0.0
        
        for j in range(epidemic_size):
            for i in np.where(class_mask)[0]:
                tau_sum += min(infections[i] - lag, removals[j]) - \
                          min(infections[i] - lag, infections[j])
        
        num_infected = np.sum(class_mask)
        period_sum = np.sum(removals - infections[:epidemic_size])
        num_not_infected = np.sum(infection_classes == infection_class) - num_infected
        
        tau_rolling += (beta_shape[class_num] + num_infected) * \
                      np.log(beta_rate[class_num] + tau_sum + 
                            num_not_infected * period_sum)
    
    # Compute log likelihood ratio for proposed configuration
    tau_rolling_proposed = 0.0
    for class_num, infection_class in enumerate(unique_classes):
        class_mask = infection_classes[:epidemic_size] == infection_class
        tau_sum = 0.0
        
        for j in range(epidemic_size):
            for i in np.where(class_mask)[0]:
                tau_sum += min(infections_proposed[i] - lag, removals[j]) - \
                          min(infections_proposed[i] - lag, infections_proposed[j])
        
        num_infected = np.sum(class_mask)
        period_sum = np.sum(removals - infections_proposed[:epidemic_size])
        num_not_infected = np.sum(infection_classes == infection_class) - num_infected
        
        tau_rolling_proposed += (beta_shape[class_num] + num_infected) * \
                               np.log(beta_rate[class_num] + tau_sum + 
                                     num_not_infected * period_sum)
    
    # Compute chi for current and proposed
    chi_current = np.zeros(epidemic_size)
    chi_proposed = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit_curr = infections[j] - lag
        infection_limit_prop = infections_proposed[j] - lag
        
        chi_current[j] = np.sum((infections[:epidemic_size] < infection_limit_curr) & 
                               (removals > infection_limit_curr))
        chi_proposed[j] = np.sum((infections_proposed[:epidemic_size] < infection_limit_prop) & 
                                (removals > infection_limit_prop))
    
    chi_current_nz = chi_current[chi_current > 0]
    chi_proposed_nz = chi_proposed[chi_proposed > 0]
    
    ell_ratio = np.sum(np.log(chi_proposed_nz)) - np.sum(np.log(chi_current_nz)) + \
               tau_rolling - tau_rolling_proposed
    
    return ell_ratio


def _update_removal_prob_multitype(removals: np.ndarray,
                                  infections: np.ndarray,
                                  removals_proposed: np.ndarray,
                                  infection_classes: np.ndarray,
                                  beta_shape: np.ndarray,
                                  beta_rate: np.ndarray,
                                  lag: float) -> float:
    """Log MH acceptance ratio for removal time proposal (multitype).
    
    Similar to _update_infected_prob_multitype but for removal times.
    
    Parameters
    ----------
    removals : ndarray
        Current removal times (epidemic_size,).
    infections : ndarray
        Current infection times (population_size,).
    removals_proposed : ndarray
        Proposed removal times (epidemic_size,).
    infection_classes : ndarray
        Infection class assignment.
    beta_shape : ndarray
        Beta prior shapes.
    beta_rate : ndarray
        Beta prior rates.
    lag : float
        Exposure lag.
    
    Returns
    -------
    log_ratio : float
        Log acceptance ratio.
    """
    epidemic_size = len(removals)
    unique_classes = np.unique(infection_classes[:epidemic_size])
    num_beta = len(unique_classes)
    
    # Compute log likelihood for current
    tau_rolling = 0.0
    for class_num, infection_class in enumerate(unique_classes):
        class_mask = infection_classes[:epidemic_size] == infection_class
        tau_sum = 0.0
        
        for j in range(epidemic_size):
            for i in np.where(class_mask)[0]:
                tau_sum += min(infections[i] - lag, removals[j]) - \
                          min(infections[i] - lag, infections[j])
        
        num_infected = np.sum(class_mask)
        period_sum = np.sum(removals - infections[:epidemic_size])
        num_not_infected = np.sum(infection_classes == infection_class) - num_infected
        
        tau_rolling += (beta_shape[class_num] + num_infected) * \
                      np.log(beta_rate[class_num] + tau_sum + 
                            num_not_infected * period_sum)
    
    # Compute log likelihood for proposed
    tau_rolling_proposed = 0.0
    for class_num, infection_class in enumerate(unique_classes):
        class_mask = infection_classes[:epidemic_size] == infection_class
        tau_sum = 0.0
        
        for j in range(epidemic_size):
            for i in np.where(class_mask)[0]:
                tau_sum += min(infections[i] - lag, removals_proposed[j]) - \
                          min(infections[i] - lag, infections[j])
        
        num_infected = np.sum(class_mask)
        period_sum = np.sum(removals_proposed - infections[:epidemic_size])
        num_not_infected = np.sum(infection_classes == infection_class) - num_infected
        
        tau_rolling_proposed += (beta_shape[class_num] + num_infected) * \
                               np.log(beta_rate[class_num] + tau_sum + 
                                     num_not_infected * period_sum)
    
    # Compute chi for current and proposed
    chi_current = np.zeros(epidemic_size)
    chi_proposed = np.zeros(epidemic_size)
    
    for j in range(epidemic_size):
        infection_limit = infections[j] - lag
        chi_current[j] = np.sum((infections[:epidemic_size] < infection_limit) & 
                               (removals > infection_limit))
        chi_proposed[j] = np.sum((infections[:epidemic_size] < infection_limit) & 
                                (removals_proposed > infection_limit))
    
    chi_current_nz = chi_current[chi_current > 0]
    chi_proposed_nz = chi_proposed[chi_proposed > 0]
    
    ell_ratio = np.sum(np.log(chi_proposed_nz)) - np.sum(np.log(chi_current_nz)) + \
               tau_rolling - tau_rolling_proposed
    
    return ell_ratio


def peirr_bayes_multitype(removals: Union[np.ndarray, list, tuple],
                         infections: Union[np.ndarray, list, tuple],
                         removal_classes: Union[np.ndarray, list, tuple],
                         infection_classes: Union[np.ndarray, list, tuple],
                         infection_class_sizes: Union[np.ndarray, list, tuple],
                         beta_init: Union[np.ndarray, list, tuple],
                         gamma_init: Union[np.ndarray, list, tuple],
                         beta_shape: Union[np.ndarray, list, tuple],
                         gamma_shape: Union[np.ndarray, list, tuple],
                         num_iter: int = 500,
                         num_update: int = 10,
                         num_tries: int = 5,
                         num_print: int = 100,
                         update_gamma: bool = False,
                         num_renewals: int = 1,
                         lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Bayesian MCMC for multitype epidemic parameters with partial data.
    
    Data augmentation MCMC algorithm for class-specific infection and removal
    rates in multitype epidemics with missing infection and removal times.
    
    Parameters
    ----------
    removals : array-like
        Removal times with np.nan for missing values.
    infections : array-like
        Infection times with np.nan for missing values.
    removal_classes : array-like
        Removal class assignment for each individual.
    infection_classes : array-like
        Infection class assignment for each individual.
    infection_class_sizes : array-like
        Population size for each infection class.
    beta_init : array-like
        Initial estimates for infection rates (one per infection class).
    gamma_init : array-like
        Initial estimates for removal rates (one per removal class).
    beta_shape : array-like
        Gamma prior shapes for infection rates.
    gamma_shape : array-like
        Gamma prior shapes for removal rates.
    num_iter : int, default 500
        Number of MCMC iterations.
    num_update : int, default 10
        MH update attempts per iteration for each time type.
    num_tries : int, default 5
        Max attempts to generate valid epidemic configuration.
    num_print : int, default 100
        Iteration print frequency.
    update_gamma : bool, default False
        If True, update gamma via Gibbs each iteration.
    num_renewals : int, default 1
        Erlang shape parameter for infectious period.
    lag : float, default 0.0
        Fixed incubation period.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'infection_rate' : ndarray
            MCMC samples of infection rates, shape (num_beta, num_iter)
        - 'removal_rate' : ndarray
            MCMC samples of removal rates, shape (num_gamma, num_iter)
        - 'prop_infection_updated' : ndarray
            Acceptance rates for infection time proposals
        - 'prop_removal_updated' : ndarray
            Acceptance rates for removal time proposals
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import peirr_bayes_multitype
    >>> np.random.seed(42)
    >>> removals = np.array([2.0, np.nan, 2.5, 4.0])
    >>> infections = np.array([0.5, 1.0, np.nan, 2.0])
    >>> removal_classes = np.array([0, 1, 0, 1])
    >>> infection_classes = np.array([0, 0, 1, 1])
    >>> fit = peirr_bayes_multitype(
    ...     removals, infections, removal_classes, infection_classes,
    ...     infection_class_sizes=[2, 2],
    ...     beta_init=[1.0, 1.0], beta_shape=[0.1, 0.1],
    ...     gamma_init=[0.8, 0.8], gamma_shape=[0.1, 0.1],
    ...     num_iter=50, num_print=100
    ... )
    >>> fit['infection_rate'].shape
    (2, 50)
    """
    # Convert to numpy arrays
    removals = np.asarray(removals, dtype=float)
    infections = np.asarray(infections, dtype=float)
    removal_classes = np.asarray(removal_classes, dtype=int)
    infection_classes = np.asarray(infection_classes, dtype=int)
    infection_class_sizes = np.asarray(infection_class_sizes, dtype=int)
    beta_init = np.asarray(beta_init, dtype=float)
    gamma_init = np.asarray(gamma_init, dtype=float)
    beta_shape = np.asarray(beta_shape, dtype=float)
    gamma_shape = np.asarray(gamma_shape, dtype=float)
    
    # Setup
    population_size = np.sum(infection_class_sizes)
    epidemic_size = np.sum(np.isfinite(infections) | np.isfinite(removals))
    
    beta_rate = beta_shape / beta_init
    gamma_rate = gamma_shape / gamma_init
    beta_rate_scaled = beta_rate / population_size
    
    if update_gamma:
        gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
    else:
        gamma_curr = gamma_init
    
    # Get unique classes
    unique_removal_classes = np.unique(removal_classes)
    unique_infection_classes = np.unique(infection_classes)
    num_beta = len(unique_infection_classes)
    num_gamma = len(unique_removal_classes)
    
    # Check if data is complete
    if np.sum(~np.isnan(infections)) == epidemic_size and \
       np.sum(~np.isnan(removals)) == epidemic_size:
        # Complete data: use bayes_complete_multitype
        infections_complete = infections[np.isfinite(infections)]
        removals_complete = removals[np.isfinite(removals)]
        infection_classes_complete = infection_classes[np.isfinite(infections) | np.isfinite(removals)]
        removal_classes_complete = removal_classes[np.isfinite(infections) | np.isfinite(removals)]
        
        out = bayes_complete_multitype(
            removals_complete, infections_complete,
            removal_classes_complete, infection_classes_complete,
            infection_class_sizes,
            beta_init=beta_init, beta_shape=beta_shape,
            gamma_init=gamma_init, gamma_shape=gamma_shape,
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
        removal_class_indices = removal_classes[nan_mask]
        gamma_vals = gamma_curr[removal_class_indices] if len(gamma_curr) > 1 else \
                     np.full_like(removal_class_indices, gamma_curr[0], dtype=float)
        infections_aug[nan_mask] = removals[nan_mask] - \
            np.random.gamma(shape=num_renewals, scale=1.0 / gamma_vals, size=num_nan_infections)
    
    # Initialize missing removals
    if num_nan_removals > 0:
        nan_mask = np.isnan(removals)
        removal_class_indices = removal_classes[nan_mask]
        gamma_vals = gamma_curr[removal_class_indices] if len(gamma_curr) > 1 else \
                     np.full_like(removal_class_indices, gamma_curr[0], dtype=float)
        removals_aug[nan_mask] = infections[nan_mask] + \
            np.random.gamma(shape=num_renewals, scale=1.0 / gamma_vals, size=num_nan_removals)
    
    # Ensure valid epidemic
    while not _check_if_epidemic_multitype(removals_aug[:epidemic_size], 
                                          infections_aug[:epidemic_size], lag):
        if update_gamma:
            gamma_curr = np.random.gamma(shape=gamma_shape, scale=1.0 / gamma_rate)
        else:
            gamma_curr = gamma_init
        
        if num_nan_infections > 0:
            nan_mask = np.isnan(infections)
            removal_class_indices = removal_classes[nan_mask]
            gamma_vals = gamma_curr[removal_class_indices]
            infections_aug[nan_mask] = removals[nan_mask] - \
                np.random.gamma(shape=num_renewals, scale=1.0 / gamma_vals, size=num_nan_infections)
        
        if num_nan_removals > 0:
            nan_mask = np.isnan(removals)
            removal_class_indices = removal_classes[nan_mask]
            gamma_vals = gamma_curr[removal_class_indices]
            removals_aug[nan_mask] = infections[nan_mask] + \
                np.random.gamma(shape=num_renewals, scale=1.0 / gamma_vals, size=num_nan_removals)
    
    # Pad infections to population size
    if len(infections_aug) < population_size:
        infections_aug = np.concatenate([infections_aug, np.full(population_size - len(infections_aug), np.inf)])
    
    # Storage
    beta_samples = np.zeros((num_beta, num_iter))
    gamma_samples = np.zeros((num_gamma, num_iter))
    updated_infections = np.zeros(num_iter)
    updated_removals = np.zeros(num_iter)
    
    # MCMC loop
    for k in range(num_iter):
        # Sample infection rates
        tau_matrix = np.zeros((epidemic_size, epidemic_size))
        for j in range(epidemic_size):
            for i in range(epidemic_size):
                tau_matrix[j, i] = min(infections_aug[i] - lag, removals_aug[j]) - \
                                  min(infections_aug[i] - lag, infections_aug[j])
        
        period_sum = np.sum(removals_aug[:epidemic_size] - infections_aug[:epidemic_size])
        
        for class_idx, infection_class in enumerate(unique_infection_classes):
            class_mask = infection_classes[:epidemic_size] == infection_class
            tau_sum = np.sum(tau_matrix[:, class_mask])
            num_infected = np.sum(class_mask)
            num_not_infected = infection_class_sizes[class_idx] - num_infected
            
            beta_samples[class_idx, k] = np.random.gamma(
                shape=beta_shape[class_idx] + num_infected,
                scale=1.0 / (beta_rate_scaled[class_idx] + tau_sum + 
                            num_not_infected * period_sum)
            )
        
        # MH for infection times
        successes = 0
        if num_nan_infections > 0:
            nan_indices = np.where(np.isnan(infections))[0]
            
            for update_step in range(num_update_infections):
                if len(nan_indices) == 1:
                    l = nan_indices[0]
                else:
                    l = np.random.choice(nan_indices)
                
                removal_class_l = removal_classes[l]
                gamma_l = gamma_curr[removal_class_l] if len(gamma_curr) > 1 else gamma_curr[0]
                
                ctr = 0
                new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals, 
                                                                   scale=1.0 / gamma_l)
                infections_proposed = infections_aug.copy()
                infections_proposed[l] = new_infection
                
                while (not _check_if_epidemic_multitype(removals_aug[:epidemic_size],
                                                       infections_proposed[:epidemic_size], lag) and
                       ctr < num_tries):
                    ctr += 1
                    new_infection = removals_aug[l] - np.random.gamma(shape=num_renewals,
                                                                      scale=1.0 / gamma_l)
                    infections_proposed[l] = new_infection
                
                if _check_if_epidemic_multitype(removals_aug[:epidemic_size],
                                               infections_proposed[:epidemic_size], lag):
                    log_accept = _update_infected_prob_multitype(
                        removals_aug[:epidemic_size], infections_aug, infections_proposed,
                        infection_classes, beta_shape, beta_rate_scaled, lag
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
                
                removal_class_l = removal_classes[l]
                gamma_l = gamma_curr[removal_class_l] if len(gamma_curr) > 1 else gamma_curr[0]
                
                ctr = 0
                new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals,
                                                                  scale=1.0 / gamma_l)
                removals_proposed = removals_aug.copy()
                removals_proposed[l] = new_removal
                
                while (not _check_if_epidemic_multitype(removals_proposed[:epidemic_size],
                                                       infections_aug[:epidemic_size], lag) and
                       ctr < num_tries):
                    ctr += 1
                    new_removal = infections_aug[l] + np.random.gamma(shape=num_renewals,
                                                                      scale=1.0 / gamma_l)
                    removals_proposed[l] = new_removal
                
                if _check_if_epidemic_multitype(removals_proposed[:epidemic_size],
                                               infections_aug[:epidemic_size], lag):
                    log_accept = _update_removal_prob_multitype(
                        removals_aug[:epidemic_size], infections_aug, removals_proposed,
                        infection_classes, beta_shape, beta_rate_scaled, lag
                    )
                    accept_prob = min(1.0, np.exp(log_accept))
                    if np.random.uniform() < accept_prob:
                        removals_aug[l] = new_removal
                        successes += 1
        
        updated_removals[k] = successes / num_update_removals if num_nan_removals > 0 else np.nan
        
        # Sample removal rates
        for class_idx, removal_class in enumerate(unique_removal_classes):
            class_mask = removal_classes[:epidemic_size] == removal_class
            removals_class = removals_aug[:epidemic_size][class_mask]
            infections_class = infections_aug[:epidemic_size][class_mask]
            
            periods = removals_class - infections_class
            period_sum = np.sum(periods[np.isfinite(periods)])
            num_periods = np.sum(np.isfinite(periods))
            
            gamma_samples[class_idx, k] = np.random.gamma(
                shape=gamma_shape[class_idx] + num_periods * num_renewals,
                scale=1.0 / (gamma_rate[class_idx] + period_sum)
            )
        
        gamma_curr = gamma_samples[:, k]
        
        # Progress printing
        if (k + 1) % num_print == 0:
            print(f"Completed iteration {k + 1} out of {num_iter}")
    
    return {
        'infection_rate': beta_samples * population_size,
        'removal_rate': gamma_samples,
        'prop_infection_updated': updated_infections,
        'prop_removal_updated': updated_removals
    }
