"""Utility functions for PEIRRS package."""

import numpy as np
import scipy
from scipy import special


def sort_sem(matrix_time):
    """Sort epidemic data by increasing removal times.
    
    Sorts the epidemic data matrix by column 2 (removal times) in ascending order.
    Handles both 2-column (basic epidemic) and 6-column (multitype epidemic) matrices.
    
    Parameters
    ----------
    matrix_time : ndarray
        Epidemic data matrix with shape (n_individuals, 2) or (n_individuals, 6).
        For 2-column format: [infection_times, removal_times].
        For 6-column format: [infection_times, removal_times, infection_class,
        infection_rate, removal_class, removal_rate].
    
    Returns
    -------
    sorted_data : ndarray
        Sorted epidemic data with same shape as input, sorted by removal times.
    
    Notes
    -----
    Non-infected individuals are denoted by infinite removal times and will appear
    last in the sorted output.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.utils import sort_sem
    >>> matrix_time = np.array([[1.0, 5.0], [0.5, 2.0], [2.0, 8.0]])
    >>> sorted_data = sort_sem(matrix_time)
    >>> sorted_data[:, 1]  # removal times in order
    array([2., 5., 8.])
    """
    matrix_time = np.asarray(matrix_time, dtype=float)
    
    if matrix_time.ndim != 2:
        raise ValueError("matrix_time must be 2-dimensional")
    
    n_cols = matrix_time.shape[1]
    if n_cols not in (2, 6):
        raise ValueError("matrix_time must have 2 or 6 columns")
    
    # Get sort indices based on removal times (column 1)
    removal_times = matrix_time[:, 1]
    sort_indices = np.argsort(removal_times)
    
    # Return sorted matrix
    return matrix_time[sort_indices, :]


def filter_sem(matrix_time):
    """Filter epidemic data to keep only infected individuals.
    
    Removes rows where both infection and removal times are non-finite (infinite or NaN).
    Non-infected individuals are denoted by infinite values and are filtered out.
    
    Parameters
    ----------
    matrix_time : ndarray
        Epidemic data matrix with shape (n_individuals, 2) or (n_individuals, 6).
        For 2-column format: [infection_times, removal_times].
        For 6-column format: [infection_times, removal_times, infection_class,
        infection_rate, removal_class, removal_rate].
    
    Returns
    -------
    filtered_data : ndarray
        Filtered epidemic data containing only infected individuals.
    
    Notes
    -----
    An individual is considered infected if either their infection time or removal
    time is finite (not infinite).
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.utils import filter_sem
    >>> matrix_time = np.array([[1.0, 5.0], [np.inf, np.inf], [2.0, 8.0]])
    >>> filtered = filter_sem(matrix_time)
    >>> filtered.shape
    (2, 2)
    """
    matrix_time = np.asarray(matrix_time, dtype=float)
    
    if matrix_time.ndim != 2:
        raise ValueError("matrix_time must be 2-dimensional")
    
    n_cols = matrix_time.shape[1]
    if n_cols not in (2, 6):
        raise ValueError("matrix_time must have 2 or 6 columns")
    
    # Filter by finite values in either infection time (col 0) or removal time (col 1)
    filter_mask = np.isfinite(matrix_time[:, 1]) | np.isfinite(matrix_time[:, 0])
    
    return matrix_time[filter_mask, :]


def decomplete_sem(matrix_time, prop_complete=0.0, prop_infection_missing=1.0):
    """Introduce missingness into epidemic data.
    
    Randomly inserts missing values (NaN) for infection and removal times to simulate
    incomplete observation of epidemics. For each individual, with probability
    1 - prop_complete, one time is set to NaN. When a time is set to NaN, it is the
    infection time with probability prop_infection_missing, otherwise the removal time.
    
    Parameters
    ----------
    matrix_time : ndarray
        Complete epidemic data matrix with shape (n_individuals, 2) or (n_individuals, 6).
        For 2-column format: [infection_times, removal_times].
        For 6-column format: [infection_times, removal_times, infection_class,
        infection_rate, removal_class, removal_rate].
    prop_complete : float, default 0.0
        Expected proportion of complete pairs (both infection and removal times observed).
        Must be in [0, 1]. With probability 1 - prop_complete, one time is set to NaN.
    prop_infection_missing : float, default 1.0
        Probability that the missing time is the infection time (rather than removal time).
        Must be in [0, 1].
    
    Returns
    -------
    incomplete_data : ndarray
        Epidemic data with randomly inserted NaN values.
    
    Notes
    -----
    The output array has the same shape as input. NaN values represent missing
    observations. Uses a random seed for reproducibility if set.
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.utils import decomplete_sem
    >>> np.random.seed(42)
    >>> matrix_time = np.array([[1.0, 5.0], [2.0, 8.0], [3.0, 10.0]])
    >>> incomplete = decomplete_sem(matrix_time, prop_complete=0.5, prop_infection_missing=0.8)
    >>> np.isnan(incomplete).any()
    True
    """
    matrix_time = np.asarray(matrix_time, dtype=float).copy()
    
    if matrix_time.ndim != 2:
        raise ValueError("matrix_time must be 2-dimensional")
    
    n_cols = matrix_time.shape[1]
    if n_cols not in (2, 6):
        raise ValueError("matrix_time must have 2 or 6 columns")
    
    if not (0.0 <= prop_complete <= 1.0):
        raise ValueError("prop_complete must be in [0, 1]")
    
    if not (0.0 <= prop_infection_missing <= 1.0):
        raise ValueError("prop_infection_missing must be in [0, 1]")
    
    n_individuals = matrix_time.shape[0]
    
    # For each individual, decide whether to introduce missingness and which type
    for i in range(n_individuals):
        # With probability 1 - prop_complete, introduce a missing value
        if np.random.binomial(1, 1.0 - prop_complete):
            # With probability prop_infection_missing, set infection time to NaN
            if np.random.binomial(1, prop_infection_missing):
                matrix_time[i, 0] = np.nan
            else:
                matrix_time[i, 1] = np.nan
    
    return matrix_time


def tau_moment(rk, rj, ik, ij, lambdak, lambdaj, lag=0.0):
    """Compute expected transmission duration between two individuals.
    
    Computes the conditional expectation of tau_{kj}, the transmission duration
    when individual k is infectious and j is susceptible, given observed infection
    and removal times. Handles 9 different missingness patterns in the data.
    
    Parameters
    ----------
    rk : float or np.nan
        Removal time of individual k. np.nan if unobserved.
    rj : float or np.nan
        Removal time of individual j. np.nan if unobserved.
    ik : float or np.nan
        Infection time of individual k. np.nan if unobserved.
    ij : float or np.nan
        Infection time of individual j. np.nan if unobserved.
    lambdak : float
        Removal rate (inverse mean infectious period) for individual k.
        Must be positive.
    lambdaj : float
        Removal rate for individual j. Must be positive.
    lag : float, default 0.0
        Fixed incubation period (exposed-to-infectious lag). Must be non-negative.
    
    Returns
    -------
    tau_expectation : float
        Conditional expectation of transmission duration tau_{kj}.
        Returns np.nan if neither k nor j is infected.
    
    Raises
    ------
    ValueError
        If both k and j times (either infection or removal) are missing.
        If more than one case pattern matches (internal error).
        If resulting tau is negative (violates theoretical constraints).
    
    Notes
    -----
    The function computes analytical integrals assuming exponential infectious
    period distributions at rates lambdak and lambdaj. Nine cases correspond to
    different missingness patterns:
    
    1. Only rk, rj observed (ik, ij missing)
    2. Only rj, ik observed (rk, ij missing)
    3. Only rk, ij observed (rj, ik missing)
    4. Only ik, ij observed (rk, rj missing)
    5. rj, ik, ij observed (rk missing)
    6. rk, ik, ij observed (rj missing)
    7. rk, rj, ij observed (ik missing)
    8. rk, rj, ik observed (ij missing)
    9. All four times observed (complete data)
    
    Examples
    --------
    >>> import numpy as np
    >>> from peirrs.utils import tau_moment
    >>> # Case 9: Complete data
    >>> tau = tau_moment(rk=5.0, rj=4.0, ik=2.0, ij=3.0,
    ...                   lambdak=1.0, lambdaj=1.0, lag=0)
    >>> tau
    0.0
    
    >>> # Case 1: Only removal times
    >>> tau = tau_moment(rk=5.0, rj=4.0, ik=np.nan, ij=np.nan,
    ...                   lambdak=1.0, lambdaj=1.5, lag=0)
    >>> tau > 0
    True
    """
    # Input validation
    if not np.isfinite(lambdak) or lambdak <= 0:
        raise ValueError("lambdak must be positive and finite")
    if not np.isfinite(lambdaj) or lambdaj <= 0:
        raise ValueError("lambdaj must be positive and finite")
    if lag < 0:
        raise ValueError("lag must be non-negative")
    
    # Check that individuals are actually infected
    if np.isnan(rk) and np.isnan(ik):
        raise ValueError("Both rk and ik are NaN: individual k not infected")
    if np.isnan(rj) and np.isnan(ij):
        raise ValueError("Both rj and ij are NaN: individual j not infected")
    
    # Adjust infection times for lag
    if np.isnan(ij):
        rj = rj - lag
    else:
        ij = ij - lag
    
    # Compute expected values for each observable pattern
    def e_tau_complete(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 9: All times observed."""
        if ij < ik:
            return 0.0
        elif ij > rk:
            return rk - ik
        else:
            return ij - ik
    
    def e_tau_rk_ik_ij(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 6: rk, ik, ij observed."""
        if ij < ik:
            return 0.0
        elif ij > rk:
            return rk - ik
        else:
            return ij - ik
    
    def e_tau_rj_ik_ij(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 5: rj, ik, ij observed."""
        if ij < ik:
            return 0.0
        elif ij > ik:
            h4 = scipy.stats.expon.sf(ij - ik, scale=1/lambdak)
            h5 = (np.exp(-lambdak * (ij - ik)) * (lambdak * (ik - ij) - 1) + 1) / lambdak
            return h4 + h5
        else:
            return 0.0
    
    def e_tau_rk_rj_ik(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 8: rk, rj, ik observed."""
        if rj < ik:
            return 0.0
        elif rj < rk:
            h6 = (np.exp(-lambdaj * (rj - ik)) - lambdaj * (ik - rj) - 1) / lambdaj
            return h6
        else:
            h8 = (rk - ik) * scipy.stats.expon.cdf(rj - rk, scale=1/lambdaj)
            h7 = (np.exp(-lambdaj * rj) * 
                  (np.exp(lambdaj * ik) + 
                   np.exp(lambdaj * rk) * 
                   (lambdaj * (rk - ik) - 1))) / lambdaj
            return h7 + h8
    
    def e_tau_rk_rj_ij(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 7: rk, rj, ij observed."""
        val = 1.0 / lambdak
        if ij < rk:
            val = val * np.exp(-lambdak * (rk - ij))
        return val
    
    def e_tau_rk_rj(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 1: Only rk, rj observed."""
        if rj < rk:
            h1 = (scipy.stats.expon.sf(rk - rj, scale=1/lambdak) * 
                  lambdaj / lambdak / (lambdak + lambdaj))
            return h1
        else:
            h2 = (scipy.stats.expon.sf(rj - rk, scale=1/lambdaj) * 
                  lambdaj / lambdak / (lambdak + lambdaj))
            h3 = scipy.stats.expon.cdf((rj - rk), scale=1/lambdaj) / lambdak
            return h2 + h3
    
    def e_tau_rk_ij(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 3: rk, ij observed (rj not useful)."""
        return e_tau_rk_rj_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    def e_tau_ik_ij(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 4: ik, ij observed (rk not useful)."""
        return e_tau_rj_ik_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    def e_tau_rj_ik(rk, rj, ik, ij, lambdak, lambdaj):
        """Case 2: rj, ik observed (complex integral case)."""
        if rj < ik:
            return 0.0
        
        # Helper functions for integral computation
        def func1(x, rate1, rate2):
            return (1 - (rate2 - rate1) * x) * np.exp((rate2 - rate1) * x)
        
        def func2(x, rate1, rate2):
            return np.exp((rate2 - rate1) * x) / (rate2 - rate1)
        
        if np.abs(lambdak - lambdaj) < 1e-10:
            h14 = lambdak * (rj**2 - ik**2) / 2
            h13 = rj - ik
            h27 = rj - ik
        else:
            h15 = (lambdak / ((lambdaj - lambdak)**2) * 
                   (func1(ik, lambdak, lambdaj) - func1(rj, lambdak, lambdaj)))
            h14 = h15
            h13 = (func2(rj, lambdak, lambdaj) - func2(ik, lambdak, lambdaj))
            h27 = ((np.exp((lambdaj - lambdak) * rj) - 
                    np.exp((lambdaj - lambdak) * ik)) / (lambdaj - lambdak))
        
        h12 = h13 + h14
        h11 = ((1 + lambdak * ik) * 
               np.exp(-lambdak * ik) * 
               (np.exp(lambdaj * rj) - np.exp(lambdaj * ik)) / lambdaj)
        h10 = (h11 - h12) / (lambdak**2)
            
        
        h9 = ((np.exp(lambdaj * rj) - np.exp(lambdaj * ik)) * 
              np.exp(-lambdak * ik) / lambdaj - h27) * ik / lambdak
        
        e1 = ((h10 - h9) * lambdak * lambdaj * 
              np.exp(-lambdaj * rj) * np.exp(lambdak * ik))
        
        # Second integral contribution
        h18 = (np.exp(-lambdak * rj) * 
               (func1(ik, 0, lambdaj) - func1(rj, 0, lambdaj)) / 
               lambdak / (lambdaj**2))
        h19 = (ik / lambdak / lambdaj * np.exp(-lambdak * rj) * 
               (np.exp(lambdaj * rj) - np.exp(lambdaj * ik)))
        h16 = (lambdak * lambdaj * np.exp(-lambdaj * rj) * 
               np.exp(lambdak * ik) * (h18 - h19))

         # ik < ij < rk < rj case
        if np.abs(lambdaj - lambdak) < 1e-10:
             h22 = (rj**2 - ik**2) / 2
             h25 = rj - ik
        else:
             h24 = ((func1(ik, lambdak, lambdaj) - func1(rj, lambdak, lambdaj)) /
                 ((lambdaj - lambdak)**2))
             h22 = h24
             h25 = ((np.exp((lambdaj - lambdak) * rj) -
                  np.exp((lambdaj - lambdak) * ik)) /
                 (lambdaj - lambdak))

        h26 = ((np.exp(lambdaj * rj) - np.exp(lambdaj * ik)) *
            np.exp(-lambdak * rj) /
            lambdaj)
        h21 = ik * (h25 - h26) / lambdak
        h23 = ((func1(ik, 0, lambdaj) - func1(rj, 0, lambdaj)) /
            (lambdaj**2) *
            np.exp(-lambdak * rj))
        h20 = (h22 - h23) / lambdak
        h17 = (lambdak * lambdaj *
            np.exp(-lambdaj * rj) *
            np.exp(lambdak * ik) *
            (h20 - h21))

        e2 = h16 + h17
        return e1 + e2
    
    # Determine which case and compute
    case_count = 0
    result = None
    
    # Case 1: Both ik and ij missing, both rk and rj observed
    if np.isnan(ij) and np.isnan(ik) and np.isfinite(rk) and np.isfinite(rj):
        case_count += 1
        result = e_tau_rk_rj(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 2: rj and ik observed, rk and ij missing
    if np.isnan(ij) and np.isnan(rk) and np.isfinite(rj) and np.isfinite(ik):
        case_count += 1
        result = e_tau_rj_ik(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 3: rk and ij observed, rj and ik missing
    if np.isnan(ik) and np.isnan(rj) and np.isfinite(rk) and np.isfinite(ij):
        case_count += 1
        result = e_tau_rk_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 4: ik and ij observed, rk and rj missing
    if np.isnan(rk) and np.isnan(rj) and np.isfinite(ik) and np.isfinite(ij):
        case_count += 1
        result = e_tau_ik_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 5: rj, ik, ij observed, rk missing
    if np.isnan(rk) and np.isfinite(rj) and np.isfinite(ik) and np.isfinite(ij):
        case_count += 1
        result = e_tau_rj_ik_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 6: rk, ik, ij observed, rj missing
    if np.isnan(rj) and np.isfinite(rk) and np.isfinite(ik) and np.isfinite(ij):
        case_count += 1
        result = e_tau_rk_ik_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 7: rk, rj, ij observed, ik missing
    if np.isnan(ik) and np.isfinite(rk) and np.isfinite(rj) and np.isfinite(ij):
        case_count += 1
        result = e_tau_rk_rj_ij(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 8: rk, rj, ik observed, ij missing
    if np.isnan(ij) and np.isfinite(rk) and np.isfinite(rj) and np.isfinite(ik):
        case_count += 1
        result = e_tau_rk_rj_ik(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Case 9: All times observed
    if (np.isfinite(rk) and np.isfinite(rj) and 
        np.isfinite(ik) and np.isfinite(ij)):
        case_count += 1
        result = e_tau_complete(rk, rj, ik, ij, lambdak, lambdaj)
    
    # Validate results
    if case_count == 0:
        print(rk,rj,ik,ij)
        raise ValueError("No valid missingness case matched the inputs")
    if case_count > 1:
        print(rk,rj,ik,ij)
        raise ValueError(f"Too many cases triggered ({case_count})")
    if result is None:
        print(rk,rj,ik,ij)
        raise ValueError("Computation resulted in None")
    if result < -1e-10:  # Small negative tolerance for numerical errors
        print(rk,rj,ik,ij)
        raise ValueError(f"tau is negative: {result}")
    
    # Return max(0, result) to handle small numerical negatives
    return max(0.0, result)
