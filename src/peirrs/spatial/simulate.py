"""Simulation functions for spatial epidemic models."""

from typing import Callable, Dict, Union
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .. import utils
from .. import estimators


def simulate_sem_spatial(beta: float,
                        gamma: float,
                        population_size: int,
                        kernel_spatial: Callable,
                        matrix_distance: Union[np.ndarray, list],
                        num_renewals: int = 1,
                        lag: float = 0.0) -> Dict[str, np.ndarray]:
    """Simulate stochastic epidemic model with spatial distance effect.
    
    Draws infectious periods and infection/removal times for a spatial SIR model
    using the Gillespie algorithm. The infection rate between individuals depends
    on their spatial distance via the kernel_spatial function.
    
    Parameters
    ----------
    beta : float
        Baseline infection rate.
    gamma : float
        Removal rate (> 0).
    population_size : int
        Number of individuals in population.
    kernel_spatial : callable
        Function of distance (d) giving relative infection rate.
        Should be symmetric and non-negative, e.g., ``lambda d: np.exp(-2*d)``
        or ``lambda d: 1/(1+d**2)``.
    matrix_distance : array-like
        Pairwise distance matrix, shape (population_size, population_size).
        Should be symmetric with zeros on diagonal.
    num_renewals : int, default 1
        Number of exponential stages in infectious period (Erlang shape).
    lag : float, default 0.0
        Fixed incubation period (exposure-to-infectious lag).
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'matrix_time' : ndarray
            Shape (population_size, 2) with columns [infection, removal].
            Inf indicates never infected or currently infectious.
        - 'matrix_record' : ndarray
            Shape (T, 5) with columns [St, Et, It, Rt, Time] recording
            susceptible, exposed, infectious, removed counts and time.
        - 'matrix_distance' : ndarray
            The input distance matrix (returned for reference).
    
    Raises
    ------
    ValueError
        If final epidemic state is invalid (S+E+I+R ≠ N or removal < infection).
    
    Notes
    -----
    The infection rate at time t is:
    
    .. math::
        \\lambda_I(t) = \\frac{\\beta}{N} \\sum_{i \\text{ infectious}, j \\text{ susceptible}} 
                       \\text{kernel\\_spatial}(d_{ij})
    
    The Erlang renewal structure allows multiple stages of infectiousness.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import simulate_sem_spatial
    >>> np.random.seed(1)
    >>> n = 50
    >>> coords = np.random.uniform(0, 1, (n, 2))
    >>> from scipy.spatial.distance import pdist, squareform
    >>> D = squareform(pdist(coords))
    >>> kernel = lambda d: np.exp(-2*d)
    >>> epi = simulate_sem_spatial(beta=1.5, gamma=1.0,
    ...                           population_size=n,
    ...                           kernel_spatial=kernel,
    ...                           matrix_distance=D)
    >>> epi['matrix_time'].shape
    (50, 2)
    >>> epi['matrix_record'].shape[1]
    5
    """
    # Input validation
    matrix_distance = np.asarray(matrix_distance, dtype=float)
    
    if not callable(kernel_spatial):
        raise ValueError("kernel_spatial must be callable")
    if population_size <= 0:
        raise ValueError("population_size must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if matrix_distance.shape != (population_size, population_size):
        raise ValueError(
            f"matrix_distance shape {matrix_distance.shape} does not match "
            f"population_size {population_size}"
        )
    
    # Initialize
    t = 0.0
    beta_N = beta / population_size
    infections = np.full(population_size, np.inf)
    removals = np.full(population_size, np.inf)
    renewals = np.zeros(population_size, dtype=int)
    
    # Patient zero: randomly selected
    alpha = np.random.choice(population_size)
    infections[alpha] = t
    
    # Initialize recording
    St = np.sum(np.isinf(infections))
    It = np.sum(np.isfinite(infections)) - np.sum(np.isfinite(removals))
    Et = 0
    Rt = 0
    
    susceptible_recording = [St]
    exposed_recording = [Et]
    infectious_recording = [It]
    removed_recording = [Rt]
    time_recording = [0.0]
    
    # Gillespie loop
    while (It > 0) or (Et > 0):
        # Compute next exposure time (when an exposed becomes infectious)
        infectious_before_t = (np.isfinite(infections) & (infections <= t) & 
                               np.isinf(removals))
        exposed = np.isfinite(infections) & (infections > t)
        
        min_time = np.inf
        if np.any(exposed):
            min_time = np.min(infections[exposed])
        
        if It == 0:
            # No infecteds, but there are exposeds -> jump to next exposure
            t = min_time + np.finfo(float).eps
        else:
            # Compute rates
            # Infection rate depends on spatial kernel
            infectious_indices = np.where(infectious_before_t)[0]
            susceptible_indices = np.where(np.isinf(infections) & np.isinf(removals))[0]
            
            infection_rate = 0.0
            if len(infectious_indices) > 0 and len(susceptible_indices) > 0:
                # Sum kernel over all (infectious, susceptible) pairs
                for i in infectious_indices:
                    infection_rate += np.sum(kernel_spatial(matrix_distance[i, susceptible_indices]))
                infection_rate *= beta_N
            
            removal_rate = gamma * It
            total_rate = infection_rate + removal_rate
            
            # Draw time to next event
            if total_rate > 0:
                t = t + np.random.exponential(1.0 / total_rate)
            else:
                t = min_time + np.finfo(float).eps
            
            # Check if event happens before next exposure becomes infectious
            if t > min_time:
                t = min_time + np.finfo(float).eps
            else:
                # Decide if infection or removal event
                if total_rate > 0:
                    removal_prob = removal_rate / total_rate
                else:
                    removal_prob = 0.0
                
                event = np.random.binomial(1, removal_prob)
                
                if event == 0:
                    # Infection event
                    if len(susceptible_indices) > 0:
                        if len(susceptible_indices) == 1:
                            argx = susceptible_indices[0]
                        else:
                            argx = np.random.choice(susceptible_indices)
                        infections[argx] = t + lag
                else:
                    # Removal event
                    infectious_can_remove = np.where((np.isfinite(infections) & 
                                                     (infections <= t) & 
                                                     np.isinf(removals)))[0]
                    
                    if len(infectious_can_remove) > 0:
                        if len(infectious_can_remove) == 1:
                            argx = infectious_can_remove[0]
                        else:
                            argx = np.random.choice(infectious_can_remove)
                        
                        renewals[argx] += 1
                        if renewals[argx] == num_renewals:
                            removals[argx] = t
        
        # Update counts
        St = np.sum(np.isinf(infections))
        It = np.sum((np.isfinite(infections) & (infections <= t)) & 
                   np.isinf(removals))
        Rt = np.sum(np.isfinite(removals))
        Et = np.sum(np.isfinite(infections) & (infections > t))
        
        # Validation
        if St + Rt + Et + It != population_size:
            raise ValueError(
                f"S(t) + E(t) + I(t) + R(t) = {St + Et + It + Rt} != {population_size}"
            )
        
        # Record
        susceptible_recording.append(St)
        exposed_recording.append(Et)
        infectious_recording.append(It)
        removed_recording.append(Rt)
        time_recording.append(t)
    
    # Final validation
    valid_periods = removals[np.isfinite(removals)] - infections[np.isfinite(removals)]
    if np.any(valid_periods < 0):
        raise ValueError("At least one removal time is before infection time")
    
    # Format output
    matrix_time = np.column_stack([infections, removals])
    matrix_record = np.column_stack([
        susceptible_recording,
        exposed_recording,
        infectious_recording,
        removed_recording,
        time_recording
    ])
    
    return {
        'matrix_time': matrix_time,
        'matrix_record': matrix_record,
        'matrix_distance': matrix_distance
    }


def simulator_spatial(beta: float,
                     gamma: float,
                     population_size: int,
                     kernel_spatial: Callable,
                     matrix_distance: Union[np.ndarray, list],
                     num_renewals: int = 1,
                     lag: float = 0.0,
                     prop_complete: float = 0.5,
                     prop_infection_missing: float = 1.0,
                     min_epidemic_size: int = 10,
                     max_epidemic_size: int = np.inf) -> Dict[str, np.ndarray]:
    """Simulate spatial epidemic with post-processing.
    
    Repeatedly simulates spatial epidemics until achieving target epidemic size
    and data completeness, then applies post-processing: filtering, missingness,
    and sorting.
    
    Parameters
    ----------
    beta : float
        Infection rate.
    gamma : float
        Removal rate.
    population_size : int
        Population size.
    kernel_spatial : callable
        Spatial kernel function of distance.
    matrix_distance : array-like
        Distance matrix, shape (population_size, population_size).
    num_renewals : int, default 1
        Erlang shape parameter.
    lag : float, default 0.0
        Incubation period.
    prop_complete : float, default 0.5
        Expected fraction of complete infection-removal pairs (must be > 0).
    prop_infection_missing : float, default 1.0
        If pair is incomplete, probability of missing infection time (vs removal).
    min_epidemic_size : int, default 10
        Minimum acceptable epidemic size.
    max_epidemic_size : int, default np.inf
        Maximum acceptable epidemic size.
    
    Returns
    -------
    result : dict
        Dictionary with keys:
        
        - 'matrix_time' : ndarray
            Filtered, incomplete, sorted infection/removal times.
        - 'matrix_record' : ndarray
            Compartment counts over time.
        - 'matrix_distance' : ndarray
            Distance matrix subset to infected individuals.
    
    Raises
    ------
    ValueError
        If prop_complete <= 0 or cannot fit epidemic size/completeness criteria.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import simulator_spatial
    >>> np.random.seed(1)
    >>> n = 80
    >>> coords = np.random.uniform(0, 1, (n, 2))
    >>> from scipy.spatial.distance import pdist, squareform
    >>> D = squareform(pdist(coords))
    >>> kernel = lambda d: np.exp(-2*d)
    >>> epi = simulator_spatial(beta=2, gamma=1, population_size=n,
    ...                        kernel_spatial=kernel, matrix_distance=D,
    ...                        prop_complete=0.7)
    >>> epi['matrix_time'].shape[0] >= 10
    True
    """
    # Validation
    matrix_distance = np.asarray(matrix_distance, dtype=float)
    
    if prop_complete <= 0:
        raise ValueError("prop_complete must be > 0")
    if not callable(kernel_spatial):
        raise ValueError("kernel_spatial must be callable")
    
    sample_size = 0
    gamma_estimate = np.nan
    max_tries = 100
    tries = 0
    
    while (sample_size <= min_epidemic_size or 
           sample_size >= max_epidemic_size or 
           np.isnan(gamma_estimate)):
        
        tries += 1
        if tries > max_tries:
            raise ValueError(
                f"Could not generate valid spatial epidemic after {max_tries} tries. "
                f"Try relaxing epidemic size constraints or prop_complete."
            )
        
        # Simulate spatial epidemic
        epidemic = simulate_sem_spatial(
            beta, gamma, population_size, kernel_spatial, matrix_distance,
            num_renewals=num_renewals, lag=lag
        )
        
        # Filter out non-infected individuals
        matrix_time = epidemic['matrix_time']
        filter_indices = np.isfinite(matrix_time[:, 0]) & np.isfinite(matrix_time[:, 1])
        matrix_time = utils.filter_sem(matrix_time)
        
        # Subset distance matrix to infected individuals
        matrix_distance_subset = matrix_distance[filter_indices, :][:, filter_indices]
        
        # Introduce missingness
        matrix_time = utils.decomplete_sem(
            matrix_time,
            prop_complete=prop_complete,
            prop_infection_missing=prop_infection_missing
        )
        
        # Sort and reorder
        sort_indices = np.argsort(matrix_time[:, 1])
        matrix_time = utils.sort_sem(matrix_time)
        
        sample_size = matrix_time.shape[0]
        
        if sample_size > 1:
            # Reorder distance matrix according to sort
            matrix_distance_subset = matrix_distance_subset[sort_indices, :][:, sort_indices]
        
        # Ensure can estimate gamma from complete pairs
        try:
            removals = matrix_time[:, 1]
            infections = matrix_time[:, 0]
            gamma_estimate = estimators.peirr_removal_rate(removals, infections)
        except Exception:
            gamma_estimate = np.nan
    
    return {
        'matrix_time': matrix_time,
        'matrix_record': epidemic['matrix_record'],
        'matrix_distance': matrix_distance_subset
    }


def simulate_distance_matrix(population_size: int,
                            kernel: Callable = np.exp,
                            inverse_kernel: Callable = np.log,
                            mu: float = 0.9,
                            sigma: float = 0.01,
                            method: str = 'euclidean',
                            runif_max: float = 100.0,
                            scalar: float = -0.05,
                            num_tries: int = 1000) -> np.ndarray:
    """Simulate a distance matrix with specified statistical properties.
    
    Generates a random distance matrix by creating coordinates in 2D space,
    computing pairwise distances, applying a transformation function, and
    scaling the resulting values to match target mean and standard deviation.
    
    Parameters
    ----------
    population_size : int
        Number of points for which to generate distance matrix. Must be positive.
    kernel : callable, default np.exp
        Transformation function to apply to scaled distances.
        Must have a corresponding inverse function.
    inverse_kernel : callable, default np.log
        Inverse of the transformation function. Must satisfy
        `inverse_kernel(kernel(x)) ≈ x` for the inverse.
    mu : float, default 0.9
        Target mean for the transformed distance values.
    sigma : float, default 0.01
        Target standard deviation for the transformed distance values.
        Must be positive.
    method : str, default 'euclidean'
        Distance metric to use. Supported: 'euclidean', 'manhattan', 'chebyshev'.
    runif_max : float, default 100.0
        Upper bound for random coordinate generation. Coordinates are sampled
        uniformly from [0, runif_max].
    scalar : float, default -0.05
        Scaling factor applied to distances before transformation.
        Typically negative.
    num_tries : int, default 1000
        Maximum number of iterations to attempt finding a valid distance matrix.
    
    Returns
    -------
    distance_matrix : ndarray
        Pairwise distance matrix of shape (population_size, population_size)
        with diagonal set to 0. The transformed values have mean `mu` and
        standard deviation `sigma`.
    
    Raises
    ------
    TypeError
        If kernel or inverse_kernel are not callable.
    ValueError
        If kernel and inverse_kernel are not true inverses of each other.
        If population_size <= 0.
        If sigma <= 0.
        If maximum number of tries reached without finding valid matrix.
    
    Notes
    -----
    The function iteratively:
    1. Generates population_size random 2D coordinates
    2. Computes pairwise distances using the specified metric
    3. Applies transformation h = kernel(scalar * D)
    4. Scales to target mean/standard deviation
    5. Stops when all scaled values are non-negative
    6. Applies inverse transformation to obtain final distance matrix
    
    The kernel and inverse_kernel should satisfy:
    inverse_kernel(kernel(x)) ≈ x for valid x values.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.spatial import simulate_distance_matrix
    >>> np.random.seed(42)
    >>> D = simulate_distance_matrix(population_size=10,
    ...                              kernel=np.exp,
    ...                              inverse_kernel=np.log,
    ...                              mu=0.9, sigma=0.01)
    >>> D.shape
    (10, 10)
    >>> np.allclose(D, D.T)  # Check symmetry
    True
    >>> np.allclose(np.diag(D), 0)  # Check diagonal is zero
    True
    """
    # Input validation
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    if not callable(inverse_kernel):
        raise TypeError("inverse_kernel must be callable")
    
    # Test that kernel and inverse_kernel are true inverses
    test_values = [1.0, 2.0, 3.0]
    tolerance = 1e-10
    
    for test_val in test_values:
        try:
            result = inverse_kernel(kernel(test_val))
        except Exception as e:
            raise ValueError(
                f"kernel and inverse_kernel failed on test value {test_val}: {e}"
            )
        
        if abs(result - test_val) > tolerance:
            raise ValueError(
                f"kernel and inverse_kernel are not true inverses. "
                f"inverse_kernel(kernel({test_val})) = {result}, expected {test_val}"
            )
    
    if population_size <= 0 or not isinstance(population_size, (int, np.integer)):
        raise ValueError("population_size must be a positive integer")
    
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    
    if method not in ('euclidean', 'manhattan', 'chebyshev'):
        raise ValueError(
            f"method '{method}' not supported. Use 'euclidean', 'manhattan', or 'chebyshev'"
        )
    
    # Iteratively generate distance matrix with valid properties
    condition = True
    ctr = 0
    
    while condition:
        ctr += 1
        
        # Generate random 2D coordinates
        coords = np.random.uniform(0, runif_max, size=(population_size, 2))
        
        # Compute pairwise distances
        if method == 'euclidean':
            distances = pdist(coords, metric='euclidean')
        elif method == 'manhattan':
            distances = pdist(coords, metric='cityblock')
        elif method == 'chebyshev':
            distances = pdist(coords, metric='chebyshev')
        
        # Convert to square matrix
        D = squareform(distances)
        
        # Apply kernel transformation
        W_h = kernel(scalar * D)
        
        # Get lower triangular values for statistics
        lower_indices = np.tril_indices(population_size, k=-1)
        original_values = W_h[lower_indices]
        
        # Calculate original mean and standard deviation
        original_mean = np.mean(original_values)
        original_sd = np.std(original_values, ddof=1)  # Sample standard deviation
        
        # Avoid division by zero
        if original_sd < 1e-15:
            original_sd = 1.0
        
        # Scale to target mean and standard deviation
        W_scaled = mu + (W_h - original_mean) * (sigma / original_sd)
        
        # Check if all scaled values are non-negative
        if np.all(W_scaled >= 0):
            condition = False
        
        # Check if maximum iterations reached
        if ctr >= num_tries:
            raise ValueError(
                f"Maximum number of tries ({num_tries}) reached without finding "
                f"a valid distance matrix. Try adjusting parameters: increasing "
                f"sigma, decreasing |scalar|, or adjusting runif_max."
            )
    
    # Undo the transformation
    D_new = inverse_kernel(W_scaled) / scalar
    
    # Set diagonal to 0 (distance from point to itself)
    np.fill_diagonal(D_new, 0)
    
    return D_new
