import numpy as np

# simulation of stochastic epidemic models

def simulate_sem(beta, gamma, N, m=1, e=0.0):
    """
    Simulate a general stochastic epidemic model.

    Parameters
    ----------
    beta : float
        Infection rate.
    gamma : float
        Removal rate.
    N : int
        Population size.
    m : int, optional
        Positive shape parameter (number of renewal stages until removal), default = 1.
    e : float, optional
        Fixed exposure period, default = 0.

    Returns
    -------
    dict
        'matrix_time': (N x 2) NumPy array of infection and removal times.
        'matrix_record': (T x 5) NumPy array of (St, It, Et, Rt, Time) over time.
    """

    # Initialization
    t = 0.0
    betaN = beta / N
    i = np.full(N, np.inf)   # infection times
    r = np.full(N, np.inf)   # removal times
    M = np.zeros(N, dtype=int)  # renewal counters

    # Initial infection
    alpha = np.random.randint(0, N)
    i[alpha] = t

    # Initialize counts
    St = np.sum(np.isinf(i))
    It = np.sum(np.isfinite(i)) - np.sum(np.isfinite(r))
    Et = 0
    Rt = 0

    # Recording vectors
    Srecording = [St]
    Irecording = [It]
    Erecording = [Et]
    Rrecording = [Rt]
    Trecording = [t]
    ctr = 1

    # Main simulation loop
    while (It > 0) or (Et > 0):
        # Closest infectious time after exposure
        mask_infectious = np.isinf(r) & np.isfinite(i) & (i > t)
        min_time = np.min(i[mask_infectious]) if np.any(mask_infectious) else np.inf

        if It == 0:
            # No infecteds but there are exposeds
            t = min_time + np.finfo(float).eps
        else:
            # Simulate event time
            irate = betaN * It * St
            rrate = gamma * It
            t += np.random.exponential(1.0 / (irate + rrate))

            if t > min_time:
                # Update time to make an exposed infectious
                t = min_time + np.finfo(float).eps
            else:
                # Infection or removal occurs
                x = np.random.binomial(1, rrate / (irate + rrate))
                x = (x + 1) % 2  # x==1 means infection event

                if x:
                    # Infect a susceptible
                    susceptible_indices = np.where(np.isinf(i))[0]
                    if len(susceptible_indices) > 0:
                        argx = np.random.choice(susceptible_indices)
                        i[argx] = t + e  # infection time (after exposure)
                else:
                    # Remove an infected
                    infected_indices = np.where(np.isinf(r) & np.isfinite(i) & (i <= t))[0]
                    if len(infected_indices) == 0:
                        break  # No more infecteds
                    argx = np.random.choice(infected_indices)
                    M[argx] += 1
                    if M[argx] == m:
                        r[argx] = t

        # Update compartment counts
        St = np.sum(np.isinf(i))
        It = np.sum(np.isfinite(i) & (i <= t)) - np.sum(np.isfinite(r))
        Rt = np.sum(np.isfinite(i) & np.isfinite(r))
        Et = np.sum(np.isfinite(i) & (i > t))

        if (St + It + Et + Rt) != N:
            raise ValueError("S(t) + I(t) + E(t) + R(t) do not equal N")

        # Record values
        Srecording.append(St)
        Irecording.append(It)
        Erecording.append(Et)
        Rrecording.append(Rt)
        Trecording.append(t)
        ctr += 1
    
    # convert to structured arrays with named columns
    matrix_time = np.rec.fromarrays([i, r],
                                    names='infection_time,removal_time',
                                    formats='f8,f8')
    matrix_record = np.rec.fromarrays([Srecording, 
                                       Irecording, 
                                       Erecording, 
                                       Rrecording, 
                                       Trecording],
                                      names='S,I,E,R,Time',
                                      formats='i8,i8,i8,i8,f8')
    
    return {'matrix_time': matrix_time,
            'matrix_record': matrix_record}

# utility functions for epidemic data manipulation

def sort_sem(epi):
    """
    Sort epidemic structured array by increasing removal times.

    Parameters
    ----------
    epi : np.recarray or np.ndarray
        Structured array with fields:
        - 'infection_time' (infection time)
        - 'removal_time' (removal time)
        - optional 'classes'

    Returns
    -------
    np.recarray
        Sorted structured array (by 'removal_time'), preserving field names.
    """
    # Sort indices by removal times
    order = np.argsort(epi['removal_time'])
    
    # Apply ordering — preserves named fields
    epi_sorted = epi[order]
    
    return epi_sorted

def filter_sem(epi):
    """
    Filter epidemic data to keep only cases with finite removal times.

    Parameters
    ----------
    epi : np.recarray
        Structured array with fields:
        - 'infection_time' (infection times)
        - 'removal_time' (removal times)
        - optional 'classes'

    Returns
    -------
    np.recarray
        Structured array containing only rows where 'removal_time' is finite.
    """
    # Mask for finite removal times
    mask = np.isfinite(epi['removal_time'])
    
    # Apply mask — preserves all fields
    epi_filtered = epi[mask]
    
    return epi_filtered


def decomplete_sem(epi, p, q=1.0):
    """
    Randomly introduce missing infection or removal times to simulate incomplete epidemic data.

    Parameters
    ----------
    epi : np.recarray
        Structured array with fields:
        - 'infection_time' (infection times)
        - 'removal_time' (removal times)
        - optional 'classes'
    p : float
        Expected proportion of complete pairs observed (probability that both infection
        and removal times are present).
    q : float, optional
        Probability that infection time is missing (given that one is missing).
        Default is 1 (infection time always missing when incomplete).

    Returns
    -------
    np.recarray
        Structured array with some infection and/or removal times set to np.nan.
    """
    # Make a copy to avoid modifying input in place
    epi_out = epi.copy()

    n = len(epi_out)
    for j in range(n):
        # With probability (1 - p), make data incomplete
        if np.random.binomial(1, 1 - p):
            # With probability q, remove infection time; else, remove removal time
            if np.random.binomial(1, q):
                epi_out['infection_time'][j] = np.nan
            else:
                epi_out['removal_time'][j] = np.nan

    return epi_out

import numpy as np

def simulate(beta, gamma, N, m=1, e=0.0, p=0.0, q=1.0):
    """
    Simulate a general stochastic epidemic model and format the output.

    Parameters
    ----------
    beta : float
        Infection rate.
    gamma : float
        Removal rate.
    N : int
        Population size.
    m : int, optional
        Positive shape parameter (number of renewals until removal), default = 1.
    e : float, optional
        Fixed exposure period, default = 0.
    p : float, optional
        Expected proportion of complete pairs observed (1 - p fraction missing), default = 0.
    q : float, optional
        Probability infection time missing (given incomplete), default = 1.

    Returns
    -------
    dict
        {
            'matrix_time': structured array with infection/removal times (and optional classes),
            'matrix_record': ndarray of (St, It, Et, Rt, Time)
        }
    """

    # --- Step 1: Simulate epidemic ---
    epi = simulate_sem(beta, gamma, N, m, e)

    # --- Step 2: Introduce missing values ---
    epi_time = decomplete_sem(epi_time, p, q)

    # --- Step 3: Filter cases with finite removal times ---
    epi_time = filter_sem(epi_time)

    # --- Step 4: Sort by removal times ---
    epi_time = sort_sem(epi_time)

    # --- Step 5: Return formatted result ---
    return {
        "matrix_time": epi_time,
        "matrix_record": epi["matrix_record"]
    }

