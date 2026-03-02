"""Stochastic epidemic model simulation functions."""

import numpy as np
from . import utils
from .estimators import peirr_removal_rate


def simulate_sem(beta, gamma, population_size, num_renewals=1, lag=0.0):
    """Simulate a stochastic SIR epidemic model.
    
    Implements an event-driven (Gillespie) algorithm for simulating a stochastic
    SIR epidemic with optional exposure lag and multiple infection stages.
    
    Parameters
    ----------
    beta : float
        Infection rate parameter. The force of infection is beta * S * I / N.
    gamma : float
        Removal rate per infectious individual.
    population_size : int
        Population size (N).
    num_renewals : int, default 1
        Number of infection stages until removal. Implements a renewal process
        where an individual must go through num_renewals events before removal.
        Corresponds to Negative Binomial infectious period distribution with
        shape num_renewals.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag). When exposed at
        time t, individual becomes infectious at time t + lag.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'matrix_time' : ndarray, shape (population_size, 2)
            Columns: [infection_time, removal_time] for each individual.
            Non-infected individuals have infection_time = Inf, removal_time = Inf.
        
        - 'matrix_record' : ndarray, shape (num_events, 5)
            Columns: [St, Et, It, Rt, Time] recording susceptible, exposed,
            infectious, removed counts and elapsed time after each event.
    
    Notes
    -----
    The simulation initializes with one random individual exposed at time 0.
    The epidemic continues until all individuals are either susceptible or removed.
    
    References
    ----------
    Gillespie, D. T. (1976). A general method for numerically simulating the stochastic
    time evolution of coupled chemical reactions. Journal of Computational Physics, 22, 403-434.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.simulate import simulate_sem
    >>> np.random.seed(42)
    >>> result = simulate_sem(beta=1.5, gamma=1.0, population_size=100)
    >>> result['matrix_time'].shape
    (100, 2)
    >>> result['matrix_record'].shape[1]
    5
    """
    # Input validation
    if beta <= 0:
        raise ValueError("beta must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if num_renewals < 1:
        raise ValueError("num_renewals must be >= 1")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    
    # Initialize arrays
    t = 0.0
    betaN = beta / population_size
    infections = np.full(population_size, np.inf)
    removals = np.full(population_size, np.inf)
    renewals = np.zeros(population_size, dtype=int)
    
    # Initial infection: one random individual exposed at t=0
    alpha = np.random.randint(0, population_size)
    infections[alpha] = t
    
    # Initialize compartment counts
    St = np.sum(np.isinf(infections))
    It = np.sum(np.isfinite(infections)) - np.sum(np.isfinite(removals))
    Et = 0
    Rt = 0
    
    # Recording vectors
    susceptible_recording = [St]
    exposed_recording = [Et]
    infectious_recording = [It]
    removal_recording = [Rt]
    time_recording = [t]
    
    # Main simulation loop
    while (It > 0) or (Et > 0):
        # Find the closest time when an exposed person becomes infectious
        mask_infectious = np.isinf(removals) & np.isfinite(infections) & (infections > t)
        min_time = np.min(infections[mask_infectious]) if np.any(mask_infectious) else np.inf
        
        if It == 0:
            # No infectious individuals, advance to next exposure becoming infectious
            t = min_time + np.finfo(float).eps
        else:
            # Calculate event rates
            irate = betaN * It * St  # infection rate
            rrate = gamma * It       # removal rate
            total_rate = irate + rrate
            
            # Sample next event time
            t += np.random.exponential(1.0 / total_rate)
            
            if t > min_time:
                # Next exposed individual becomes infectious before next event
                t = min_time + np.finfo(float).eps
            else:
                # Event occurs: infection or removal
                # Probability of removal given event
                prob_removal = rrate / total_rate
                x = np.random.binomial(1, prob_removal)
                
                if x == 0:
                    # Infection event: expose a susceptible
                    susceptible_indices = np.where(np.isinf(infections))[0]
                    if len(susceptible_indices) > 0:
                        argx = np.random.choice(susceptible_indices)
                        infections[argx] = t + lag
                else:
                    # Removal event: remove an infectious individual
                    infectious_indices = np.where(np.isinf(removals) & 
                                                 np.isfinite(infections) & 
                                                 (infections <= t))[0]
                    if len(infectious_indices) == 0:
                        break
                    argx = np.random.choice(infectious_indices)
                    renewals[argx] += 1
                    if renewals[argx] == num_renewals:
                        removals[argx] = t
        
        # Update compartment counts
        St = np.sum(np.isinf(infections))
        It = np.sum(np.isfinite(infections) & (infections <= t)) - np.sum(np.isfinite(removals))
        Rt = np.sum(np.isfinite(infections) & np.isfinite(removals))
        Et = np.sum(np.isfinite(infections) & (infections > t))
        
        # Validate counts
        if (St + It + Et + Rt) != population_size:
            raise ValueError("S(t) + I(t) + E(t) + R(t) do not equal population_size")
        
        # Record values
        susceptible_recording.append(St)
        exposed_recording.append(Et)
        infectious_recording.append(It)
        removal_recording.append(Rt)
        time_recording.append(t)
    
    # Validate that removal times >= infection times
    complete_mask = np.isfinite(removals) & np.isfinite(infections)
    infectious_periods_valid = removals[complete_mask] - infections[complete_mask]
    if np.any(infectious_periods_valid < 0):
        raise ValueError("At least one removal time is before infection time")
    
    # Format output as 2D arrays
    matrix_time = np.column_stack((infections, removals))
    matrix_record = np.column_stack((susceptible_recording,
                                     exposed_recording,
                                     infectious_recording,
                                     removal_recording,
                                     time_recording))
    
    return {
        'matrix_time': matrix_time,
        'matrix_record': matrix_record
    }



def simulator(beta, gamma, population_size, num_renewals=1, lag=0.0,
              prop_complete=0.5, prop_infection_missing=1.0,
              min_epidemic_size=10, max_epidemic_size=np.inf):
    """Simulate complete epidemic with post-processing.
    
    Repeatedly simulates epidemics until the observed size falls within specified
    bounds and there are enough complete observations to estimate the removal rate.
    
    The output is post-processed by:
    1. Filtering to keep only infected individuals
    2. Introducing missingness (NaN values)
    3. Sorting by removal time
    
    Parameters
    ----------
    beta : float
        Infection rate.
    gamma : float
        Removal rate.
    population_size : int
        Population size.
    num_renewals : int, default 1
        Number of infection stages before removal.
    lag : float, default 0.0
        Fixed incubation period.
    prop_complete : float, default 0.5
        Expected proportion of complete infection-removal pairs. Must be > 0.
    prop_infection_missing : float, default 1.0
        Probability that missing time is infection time (vs removal time).
    min_epidemic_size : int, default 10
        Minimum number of infected individuals required.
    max_epidemic_size : float, default np.inf
        Maximum number of infected individuals allowed.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'matrix_time' : ndarray, shape (n_infected, 2)
            [infection_times, removal_times] sorted by removal time,
            with some values set to NaN (missing).
        
        - 'matrix_record' : ndarray, shape (num_events, 5)
            Time-indexed recording of [St, Et, It, Rt, Time].
    
    Raises
    ------
    ValueError
        If prop_complete <= 0 (no complete pairs possible).
    
    Notes
    -----
    This function is useful for generating realistic epidemic data with
    incomplete observation. It ensures sufficient data to estimate parameters.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.simulate import simulator
    >>> np.random.seed(42)
    >>> epi = simulator(beta=2.0, gamma=1.5, population_size=100,
    ...                 prop_complete=0.8, min_epidemic_size=20)
    >>> epi['matrix_time'].shape
    (20, 2)
    >>> np.sum(np.isnan(epi['matrix_time'])) > 0  # Has missing values
    True
    """
    if prop_complete <= 0:
        raise ValueError("prop_complete must be > 0 (need complete pairs to estimate gamma)")
    if not (0.0 <= prop_infection_missing <= 1.0):
        raise ValueError("prop_infection_missing must be in [0, 1]")
    if min_epidemic_size < 1:
        raise ValueError("min_epidemic_size must be >= 1")
    if max_epidemic_size < min_epidemic_size:
        raise ValueError("max_epidemic_size must be >= min_epidemic_size")
    
    sample_size = 0
    gamma_estimate = np.nan
    
    while (sample_size < min_epidemic_size or 
           sample_size > max_epidemic_size or 
           np.isnan(gamma_estimate)):
        
        # Simulate epidemic
        epidemic = simulate_sem(beta, gamma, population_size, num_renewals, lag)
        matrix_time = epidemic['matrix_time']
        
        # Filter to keep infected only
        matrix_time = utils.filter_sem(matrix_time)
        
        # Introduce missingness
        matrix_time = utils.decomplete_sem(matrix_time, 
                                          prop_complete=prop_complete,
                                          prop_infection_missing=prop_infection_missing)
        
        # Sort by removal time
        matrix_time = utils.sort_sem(matrix_time)
        
        # Check sample size
        sample_size = matrix_time.shape[0]
        
        # Try to estimate gamma to ensure there are complete pairs
        removals = matrix_time[:, 1]
        infections = matrix_time[:, 0]
        gamma_estimate = peirr_removal_rate(removals, infections)
    
    return {
        'matrix_time': matrix_time,
        'matrix_record': epidemic['matrix_record']
    }

