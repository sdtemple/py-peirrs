"""Tests for simulation functions in peirrs.simulate."""

import pytest
import numpy as np
from peirrs.simulate import simulate_sem, simulator


class TestSimulateSem:
    """Test cases for simulate_sem function."""
    
    def test_simulate_sem_generates_array(self):
        """Test that simulate_sem returns array with correct shape."""
        np.random.seed(123)
        X = simulate_sem(beta=2.0, gamma=1.0, population_size=100)['matrix_time']
        
        assert isinstance(X, np.ndarray)
        assert X.ndim == 2
        assert X.shape[1] == 2  # infection and removal times
        assert X.shape[0] > 0
    
    def test_simulate_sem_infection_before_removal(self):
        """Test that infection times are always before removal times."""
        np.random.seed(456)
        epi= simulate_sem(beta=2.0, gamma=1.0, population_size=100)
        X = epi['matrix_time']
        
        # For all infected individuals, infection < removal
        finite_mask = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])
        assert np.all(X[finite_mask, 0] < X[finite_mask, 1])
    
    def test_simulate_sem_times_non_negative(self):
        """Test that all event times are non-negative."""
        np.random.seed(789)
        epi= simulate_sem(beta=2.0, gamma=1.0, population_size=100)
        X = epi['matrix_time']
        
        # All finite times should be non-negative
        finite_times = X[np.isfinite(X)]
        assert np.all(finite_times >= 0)
    
    def test_simulate_sem_returns_valid_population_count(self):
        """Test that population count matches or close to input."""
        np.random.seed(101)
        pop_size = 100
        epi = simulate_sem(beta=2.0, gamma=1.0, population_size=pop_size)
        N = epi['matrix_time'].shape[0]
        
        # N should be <= population_size (not all may be infected)
        assert N <= pop_size
        assert N >= 1


class TestSimulator:
    """Test cases for simulator wrapper function."""
    
    def test_simulator_respects_min_epidemic_size(self):
        """Test that simulator respects minimum epidemic size constraint."""
        np.random.seed(123)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=20, max_epidemic_size=np.inf,
                           prop_complete=1.0)
        
        X = epidemic['matrix_time']
        epidemic_size = X.shape[0]
        
        assert epidemic_size >= 20
        assert X.shape[1] == 2
    
    def test_simulator_respects_max_epidemic_size(self):
        """Test that simulator respects maximum epidemic size constraint."""
        np.random.seed(456)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=5, max_epidemic_size=30,
                           prop_complete=1.0)
        
        X = epidemic['matrix_time']
        epidemic_size = X.shape[0]
        
        assert epidemic_size <= 30
        assert epidemic_size >= 5
    
    def test_simulator_creates_missing_values(self):
        """Test that simulator creates missing values with prop_complete < 1."""
        np.random.seed(789)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           prop_complete=0.3, prop_infection_missing=0.5,
                           min_epidemic_size=50)
        
        X = epidemic['matrix_time']
        
        # Count missing values
        num_missing_inf = np.sum(np.isnan(X[:, 0]))
        num_missing_rem = np.sum(np.isnan(X[:, 1]))
        
        # Should have some missing values when prop_complete < 1
        assert num_missing_inf > 0 or num_missing_rem > 0
    
    def test_simulator_respects_prop_complete(self):
        """Test that observed proportion of complete pairs matches expectation."""
        np.random.seed(101)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           prop_complete=0.5, min_epidemic_size=50)
        
        X = epidemic['matrix_time']
        complete_mask = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])
        observed_prop = np.sum(complete_mask) / X.shape[0]
        
        # Allow ±20% deviation from expected
        assert 0.3 < observed_prop < 0.7
    
    def test_simulator_prop_infection_missing_1(self):
        """Test that prop_infection_missing=1 only removes infection times."""
        np.random.seed(202)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           prop_complete=0.2, prop_infection_missing=1.0,
                           min_epidemic_size=50)
        
        X = epidemic['matrix_time']
        
        # Count which times are missing
        inf_missing = np.isnan(X[:, 0]) & np.isfinite(X[:, 1])
        rem_missing = np.isfinite(X[:, 0]) & np.isnan(X[:, 1])
        
        # Should have infection missing, no removal missing
        assert np.sum(inf_missing) > 0
        assert np.sum(rem_missing) == 0
    
    def test_simulator_prop_infection_missing_0(self):
        """Test that prop_infection_missing=0 only removes removal times."""
        np.random.seed(303)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           prop_complete=0.2, prop_infection_missing=0.0,
                           min_epidemic_size=50)
        
        X = epidemic['matrix_time']
        
        # Count which times are missing
        inf_missing = np.isnan(X[:, 0]) & np.isfinite(X[:, 1])
        rem_missing = np.isfinite(X[:, 0]) & np.isnan(X[:, 1])
        
        # Should have removal missing, no infection missing
        assert np.sum(rem_missing) > 0
        assert np.sum(inf_missing) == 0
    
    def test_simulator_returns_dict_structure(self):
        """Test that simulator returns dict with expected keys."""
        np.random.seed(404)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=20)
        
        assert isinstance(epidemic, dict)
        assert 'matrix_time' in epidemic
        assert isinstance(epidemic['matrix_time'], np.ndarray)
    
    def test_simulator_consistent_with_seed(self):
        """Test that simulator produces consistent results with same seed."""
        np.random.seed(505)
        epidemic1 = simulator(beta=2.0, gamma=1.0, population_size=100,
                            min_epidemic_size=20, max_epidemic_size=80)
        
        np.random.seed(505)
        epidemic2 = simulator(beta=2.0, gamma=1.0, population_size=100,
                            min_epidemic_size=20, max_epidemic_size=80)
        
        # Should produce identical results
        assert np.allclose(epidemic1['matrix_time'], epidemic2['matrix_time'],
                          equal_nan=True)
    
    def test_simulator_different_beta_values(self):
        """Test that simulator accepts different beta values."""
        np.random.seed(606)
        epidemic_low = simulator(beta=0.5, gamma=1.0, population_size=200,
                               min_epidemic_size=5, max_epidemic_size=100)
        
        np.random.seed(707)
        epidemic_high = simulator(beta=3.0, gamma=1.0, population_size=200,
                                min_epidemic_size=5, max_epidemic_size=100)
        
        assert isinstance(epidemic_low, dict)
        assert isinstance(epidemic_high, dict)
    
    def test_simulator_valid_infection_removal_ordering(self):
        """Test that infection times are before removal times."""
        np.random.seed(808)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           prop_complete=1.0, min_epidemic_size=30)
        
        X = epidemic['matrix_time']
        # All finite times should satisfy infection < removal
        finite_mask = np.isfinite(X[:, 0]) & np.isfinite(X[:, 1])
        assert np.all(X[finite_mask, 0] < X[finite_mask, 1])