"""Simulation functions for multitype epidemic models."""

from typing import Union, Dict, Tuple, List
import numpy as np
from .. import utils


def simulate_sem_multitype(beta: Union[np.ndarray, list, tuple],
                           gamma: Union[np.ndarray, list, tuple],
                           infection_class_sizes: Union[np.ndarray, list, tuple],
                           removal_class_sizes: Union[np.ndarray, list, tuple],
                           num_renewals: int = 1,
                           lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Stochastic epidemic model simulation with multitype classes.
    
    Simulates an SIR epidemic using a Gillespie algorithm with separate
    infection-rate and removal-rate classes. Individuals are assigned to
    classes based on stratified population structure.
    
    Parameters
    ----------
    beta : array-like
        Infection rate vector with length equal to number of infection classes.
        Each element is the infection rate for that class.
    gamma : array-like
        Removal rate vector with length equal to number of removal classes.
        Each element is the removal rate for that class.
    infection_class_sizes : array-like
        Population sizes for each infection-rate class. Sum must equal
        removal_class_sizes sum.
    removal_class_sizes : array-like
        Population sizes for each removal-rate class. Sum must equal
        infection_class_sizes sum.
    num_renewals : int, default 1
        Number of exponential stages in infectious period (Erlang shape).
        num_renewals=1 gives exponential, >1 gives Erlang distribution.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious delay).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'matrix_time' : ndarray
            Array of shape (N, 6) with columns:
            (infection, removal, infection_class, infection_rate, 
             removal_class, removal_rate).
            Uses np.inf for times that never occur.
        - 'matrix_record' : ndarray
            Array of shape (T, 5) recording evolution over time with columns:
            (St, Et, It, Rt, Time) where S/E/I/R are counts of 
            susceptible/exposed/infectious/removed individuals.
    
    Raises
    ------
    ValueError
        If infection_class_sizes and removal_class_sizes don't sum to same total.
    
    Notes
    -----
    This function implements a multitype SIR/SEIR model where each individual
    has two independent class assignments:
    
    - Infection class: determines infectiousness when infectious (controls beta)
    - Removal class: determines recovery speed when infected (controls gamma)
    
    Class assignment for new infections:
    - Infection class: sampled proportional to remaining class sizes and beta_k
    - Removal class: sampled uniformly from remaining class sizes
    
    The algorithm uses Gillespie's direct method with event rates:
    - Infection: I(t) * sum_k(S_k * beta_k / N)
    - Removal: sum_k(I_k * gamma_k) where I_k is infectious count in removal class k
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import simulate_sem_multitype
    >>> np.random.seed(42)
    >>> result = simulate_sem_multitype(
    ...     beta=[1.5, 2.0],
    ...     gamma=[0.8, 1.2],
    ...     infection_class_sizes=[50, 50],
    ...     removal_class_sizes=[50, 50],
    ...     num_renewals=1,
    ...     lag=0.0
    ... )
    >>> result['matrix_time'].shape
    (100, 6)
    >>> result['matrix_record'].shape[1]
    5
    """
    # Convert inputs to numpy arrays
    beta = np.asarray(beta, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    infection_class_sizes = np.asarray(infection_class_sizes, dtype=int).copy()
    removal_class_sizes = np.asarray(removal_class_sizes, dtype=int).copy()
    
    # Validation
    if np.sum(infection_class_sizes) != np.sum(removal_class_sizes):
        raise ValueError(
            "Infection and removal class sizes must sum to the same total. "
            f"infection sum={np.sum(infection_class_sizes)}, "
            f"removal sum={np.sum(removal_class_sizes)}"
        )
    
    population_size = int(np.sum(infection_class_sizes))
    beta_normalized = beta / population_size
    
    # Initialize individual tracking arrays
    t = 0.0
    infections = np.full(population_size, np.inf, dtype=float)
    removals = np.full(population_size, np.inf, dtype=float)
    infection_classes = np.full(population_size, -1, dtype=int)
    removal_classes = np.full(population_size, -1, dtype=int)
    infection_rates = np.full(population_size, np.nan, dtype=float)
    removal_rates = np.full(population_size, np.nan, dtype=float)
    renewals = np.zeros(population_size, dtype=int)
    
    # Count of each removal class with active infectious
    removals_current = np.zeros(len(removal_class_sizes), dtype=int)
    
    # Initialize patient zero
    infection_weights = infection_class_sizes / population_size
    zero_infection_class = np.random.choice(len(infection_class_sizes), p=infection_weights)
    infections[0] = t
    infection_classes[0] = zero_infection_class
    infection_class_sizes[zero_infection_class] -= 1
    infection_rates[0] = beta[zero_infection_class]
    
    removal_weights = removal_class_sizes / population_size
    zero_removal_class = np.random.choice(len(removal_class_sizes), p=removal_weights)
    removal_classes[0] = zero_removal_class
    removal_class_sizes[zero_removal_class] -= 1
    removal_rates[0] = gamma[zero_removal_class]
    removals_current[zero_removal_class] += 1
    
    itr = 1  # Index of next individual to be infected
    
    # Recording evolution
    susceptible_recording = [population_size - 1]
    exposed_recording = [0]
    infection_recording = [0]
    removed_recording = [0]
    time_recording = [0.0]
    
    # Gillespie algorithm loop
    while True:
        # Count compartments
        St = np.sum(np.isinf(infections))
        It = np.sum(np.isfinite(infections) & (infections <= t)) - np.sum(np.isfinite(removals))
        Rt = np.sum(np.isfinite(removals))
        Et = np.sum(np.isfinite(infections) & (infections > t))
        
        if St + It + Et + Rt != population_size:
            raise RuntimeError("Compartment counts don't sum to population size")
        
        if It == 0 and Et == 0:
            break  # Epidemic ended
        
        # Find earliest exposed individual who becomes infectious
        exposed_mask = np.isfinite(infections) & np.isinf(removals) & (infections > t)
        if np.any(exposed_mask):
            min_time = np.min(infections[exposed_mask])
            arg_min = np.argmin(np.where(exposed_mask)[0])
            arg_min_time = np.where(exposed_mask)[0][arg_min]
        else:
            min_time = np.inf
            arg_min_time = -1
        
        if It == 0:
            # No infecteds, just exposeds transitioning to infectious
            t = min_time + np.finfo(float).eps
            sampled_removal_class = removal_classes[arg_min_time]
            removals_current[sampled_removal_class] += 1
        else:
            # Compute event rates
            infection_rate = It * np.sum(infection_class_sizes * beta_normalized)
            removal_rate = np.sum(removals_current * gamma)
            total_rate = infection_rate + removal_rate
            
            # Sample time to next event
            t = t + np.random.exponential(1.0 / total_rate)
            
            if t > min_time:
                # Exposed individual becomes infectious before next event
                t = min_time + np.finfo(float).eps
                sampled_removal_class = removal_classes[arg_min_time]
                removals_current[sampled_removal_class] += 1
            else:
                # Determine event type: removal vs infection
                if np.random.uniform() < removal_rate / total_rate:
                    # Removal event
                    infectious_mask = np.isfinite(infections) & np.isinf(removals) & (infections <= t)
                    infectious_indices = np.where(infectious_mask)[0]
                    
                    if len(infectious_indices) > 1:
                        # Sample weighted by removal rates
                        removal_weights_local = removal_rates[infectious_indices]
                        removal_weights_local = removal_weights_local / np.sum(removal_weights_local)
                        argx = np.random.choice(infectious_indices, p=removal_weights_local)
                    else:
                        argx = infectious_indices[0]
                    
                    renewals[argx] += 1
                    if renewals[argx] == num_renewals:
                        removals[argx] = t
                        removal_class = removal_classes[argx]
                        removals_current[removal_class] -= 1
                else:
                    # Infection event
                    if itr < population_size:
                        # Sample infection class weighted by rate and available class size
                        infection_weights_local = infection_class_sizes * beta_normalized
                        infection_weights_local = infection_weights_local / np.sum(infection_weights_local)
                        sampled_infection_class = np.random.choice(len(infection_class_sizes),
                                                                   p=infection_weights_local)
                        infection_class_sizes[sampled_infection_class] -= 1
                        
                        infections[itr] = t + lag
                        infection_classes[itr] = sampled_infection_class
                        infection_rates[itr] = beta[sampled_infection_class]
                        
                        # Sample removal class uniformly from available
                        removal_weights_uniform = removal_class_sizes / np.sum(removal_class_sizes)
                        sampled_removal_class = np.random.choice(len(removal_class_sizes),
                                                                p=removal_weights_uniform)
                        removal_class_sizes[sampled_removal_class] -= 1
                        removal_classes[itr] = sampled_removal_class
                        removal_rates[itr] = gamma[sampled_removal_class]
                        
                        if lag == 0:
                            removals_current[sampled_removal_class] += 1
                        
                        itr += 1
        
        # Record state
        susceptible_recording.append(St)
        exposed_recording.append(Et)
        infection_recording.append(It)
        removed_recording.append(Rt)
        time_recording.append(t)
    
    # Assign remaining non-infected individuals to classes
    if itr < population_size:
        # Infection classes
        remain_infection_classes = []
        remain_infection_rates = []
        for class_idx in range(len(infection_class_sizes)):
            if infection_class_sizes[class_idx] > 0:
                remain_infection_classes.extend([class_idx] * infection_class_sizes[class_idx])
                remain_infection_rates.extend([beta[class_idx]] * infection_class_sizes[class_idx])
        
        shuffled_idx = np.random.permutation(len(remain_infection_classes))
        infection_classes[itr:population_size] = np.array(remain_infection_classes)[shuffled_idx]
        infection_rates[itr:population_size] = np.array(remain_infection_rates)[shuffled_idx]
        
        # Removal classes
        remain_removal_classes = []
        remain_removal_rates = []
        for class_idx in range(len(removal_class_sizes)):
            if removal_class_sizes[class_idx] > 0:
                remain_removal_classes.extend([class_idx] * removal_class_sizes[class_idx])
                remain_removal_rates.extend([gamma[class_idx]] * removal_class_sizes[class_idx])
        
        shuffled_idx = np.random.permutation(len(remain_removal_classes))
        removal_classes[itr:population_size] = np.array(remain_removal_classes)[shuffled_idx]
        removal_rates[itr:population_size] = np.array(remain_removal_rates)[shuffled_idx]
    
    # Validation: check non-negative infectious periods
    complete_mask = np.isfinite(removals) & np.isfinite(infections)
    infectious_periods = removals[complete_mask] - infections[complete_mask]
    if np.any(infectious_periods < 0):
        raise RuntimeError("Negative infectious period detected")
    
    # Format output
    matrix_time = np.column_stack([
        infections, removals, infection_classes, infection_rates,
        removal_classes, removal_rates
    ])
    
    matrix_record = np.column_stack([
        susceptible_recording, exposed_recording, infection_recording,
        removed_recording, time_recording
    ])
    
    return {
        'matrix_time': matrix_time,
        'matrix_record': matrix_record
    }


def simulator_multitype(beta: Union[np.ndarray, list, tuple],
                       gamma: Union[np.ndarray, list, tuple],
                       infection_class_sizes: Union[np.ndarray, list, tuple],
                       removal_class_sizes: Union[np.ndarray, list, tuple],
                       num_renewals: int = 1,
                       lag: float = 0.0,
                       prop_complete: float = 0.5,
                       prop_infection_missing: float = 1.0,
                       min_epidemic_size: int = 10,
                       max_epidemic_size: int = np.inf) -> Dict[str, np.ndarray]:
    """Multitype epidemic simulator with validation and post-processing.
    
    Repeatedly simulates multitype epidemics until epidemic size is within
    acceptable bounds and there are sufficient complete infectious periods
    to estimate removal rates for each removal class.
    
    Parameters
    ----------
    beta : array-like
        Infection rate vector for each infection class.
    gamma : array-like
        Removal rate vector for each removal class.
    infection_class_sizes : array-like
        Population sizes for infection classes.
    removal_class_sizes : array-like
        Population sizes for removal classes.
    num_renewals : int, default 1
        Number of exponential stages in infectious period.
    lag : float, default 0.0
        Fixed incubation period.
    prop_complete : float, default 0.5
        Proportion of individuals with both times observed.
        Must be > 0.
    prop_infection_missing : float, default 1.0
        Probability infection time is missing (if not complete).
        Must be in [0, 1].
    min_epidemic_size : int, default 10
        Minimum acceptable epidemic size.
    max_epidemic_size : int, default inf
        Maximum acceptable epidemic size.
    
    Returns
    -------
    epidemic : dict
        Dictionary with 'matrix_time' and 'matrix_record' keys containing
        post-processed simulation output. Data has been:
        - Filtered to remove non-infected individuals
        - Missing values introduced according to prop_complete
        - Sorted by removal time
    
    Raises
    ------
    ValueError
        If prop_complete <= 0 or other parameters invalid.
    
    Notes
    -----
    Post-processing steps:
    1. Remove non-infected individuals (both times infinite)
    2. Introduce missingness: complete with probability prop_complete;
       if incomplete, infection time missing with probability
       prop_infection_missing.
    3. Sort by removal time
    
    For multitype epidemics, validation ensures that each removal class has
    at least one individual with a complete infectious period, allowing
    independent estimation of each removal rate.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.multitype import simulator_multitype
    >>> np.random.seed(42)
    >>> epi = simulator_multitype(
    ...     beta=[1.5, 2.0],
    ...     gamma=[0.8, 1.2],
    ...     infection_class_sizes=[50, 50],
    ...     removal_class_sizes=[50, 50],
    ...     prop_complete=0.8,
    ...     prop_infection_missing=0.5
    ... )
    >>> epi['matrix_time'].shape[0] > 10  # At least min_epidemic_size
    True
    """
    # Input validation
    if prop_complete <= 0:
        raise ValueError("prop_complete must be > 0")
    if not (0 <= prop_infection_missing <= 1):
        raise ValueError("prop_infection_missing must be in [0, 1]")
    if min_epidemic_size < 1:
        raise ValueError("min_epidemic_size must be >= 1")
    
    gamma_array = np.asarray(gamma, dtype=float)
    removal_class_sizes_orig = np.asarray(removal_class_sizes, dtype=int)
    num_removal_classes = len(gamma_array)
    
    sample_size = 0
    valid_gamma = False
    
    while sample_size < min_epidemic_size or sample_size > max_epidemic_size or not valid_gamma:
        # Simulate
        epidemic = simulate_sem_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes_orig.copy(),
            num_renewals=num_renewals,
            lag=lag
        )
        
        # Post-process
        epidemic['matrix_time'] = utils.filter_sem(epidemic['matrix_time'])
        epidemic['matrix_time'] = utils.decomplete_sem(
            epidemic['matrix_time'],
            prop_complete=prop_complete,
            prop_infection_missing=prop_infection_missing
        )
        epidemic['matrix_time'] = utils.sort_sem(epidemic['matrix_time'])
        
        # Check sample size
        sample_size = epidemic['matrix_time'].shape[0]
        
        # Check if we can estimate all removal rates
        valid_gamma = True
        for removal_class_idx in range(num_removal_classes):
            removal_col = 4  # removal_class is column 4
            infections_col = 0
            removals_col = 1
            
            class_mask = epidemic['matrix_time'][:, removal_col] == removal_class_idx
            if np.sum(class_mask) == 0:
                valid_gamma = False
                break
            
            removals_class = epidemic['matrix_time'][class_mask, removals_col]
            infections_class = epidemic['matrix_time'][class_mask, infections_col]
            
            # Check for complete periods
            complete_mask = np.isfinite(removals_class) & np.isfinite(infections_class)
            if np.sum(complete_mask) == 0:
                valid_gamma = False
                break
    
    return epidemic
