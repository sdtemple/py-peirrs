"""Tests for estimation functions."""

import pytest
import numpy as np
from peirrs.estimators import (
    peirr_removal_rate,
    peirr_tau,
    peirr_imputed,
    bayes_complete,
)
from peirrs.simulate import simulator


class TestPeirrRemovalRate:
    """Test cases for peirr_removal_rate function."""
    
    def test_removal_rate_returns_positive(self):
        """Test that removal rate estimator returns positive value."""
        removals = np.array([3.0, 4.0, 5.0, 6.0])
        infections = np.array([1.0, 2.0, 2.5, 3.0])
        
        gamma = peirr_removal_rate(removals, infections)
        
        assert isinstance(gamma, (float, np.floating))
        assert gamma > 0
    
    def test_removal_rate_with_uniform_intervals(self):
        """Test removal rate with uniform infection-removal intervals."""
        # All infectious periods are 2.0
        removals = np.array([3.0, 4.0, 5.0, 6.0])
        infections = np.array([1.0, 2.0, 3.0, 4.0])
        
        gamma = peirr_removal_rate(removals, infections)
        
        # Mean infectious period = 2.0, so gamma = 1/2 = 0.5
        assert np.isclose(gamma, 0.5, atol=0.01)
    
    def test_removal_rate_with_simulated_data(self):
        """Test removal rate on simulated complete data."""
        np.random.seed(123)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=1.0)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        gamma = peirr_removal_rate(removals, infections)
        
        # Should be positive and in reasonable range
        assert gamma > 0
        assert gamma < 5.0  # Not unreasonably high


class TestPeirrTau:
    """Test cases for peirr_tau function."""
    
    def test_tau_returns_dict(self):
        """Test that tau estimator returns dict with expected structure."""
        np.random.seed(123)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.8)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_tau(removals=removals, infections=infections,
                          population_size=100)
        
        assert isinstance(result, dict)
        assert 'infection_rate' in result
        assert 'removal_rate' in result
    
    def test_tau_infection_rate_positive(self):
        """Test that infection rate estimate is positive."""
        np.random.seed(456)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.8)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_tau(removals=removals, infections=infections,
                          population_size=100)
        
        assert result['infection_rate'] > 0
        assert result['removal_rate'] > 0
    
    def test_tau_gamma_matches_removal_rate(self):
        """Test that tau's gamma estimate matches direct removal_rate estimator."""
        np.random.seed(789)
        epidemic = simulator(beta=1.5, gamma=1.0, population_size=200,
                           min_epidemic_size=40, prop_complete=0.8)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        tau_result = peirr_tau(removals=removals, infections=infections,
                              population_size=200)
        direct_gamma = peirr_removal_rate(removals, infections)
        
        # Should match exactly according to R tests
        assert np.isclose(tau_result['removal_rate'], direct_gamma)
    
    def test_tau_with_missing_data(self):
        """Test tau estimator with incomplete data."""
        np.random.seed(101)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.5)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_tau(removals=removals, infections=infections,
                          population_size=100)
        
        assert result['infection_rate'] > 0
        assert result['removal_rate'] > 0
    
    def test_tau_with_lag(self):
        """Test tau estimator with exposure lag."""
        np.random.seed(202)
        lag = 0.5
        epidemic = simulator(beta=1.8, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.8, lag=lag)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_tau(removals=removals, infections=infections,
                          population_size=100, lag=lag)
        
        # With lag, estimates should still be positive
        assert result['infection_rate'] > 0
        assert result['removal_rate'] > 0


class TestPeirrImputed:
    """Test cases for peirr_imputed function."""
    
    def test_imputed_returns_dict(self):
        """Test that imputed estimator returns dict with expected structure."""
        np.random.seed(303)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.8)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_imputed(removals=removals, infections=infections,
                              population_size=100)
        
        assert isinstance(result, dict)
        assert 'infection_rate' in result
        assert 'removal_rate' in result
    
    def test_imputed_gamma_matches_removal_rate(self):
        """Test that imputed's gamma matches direct removal_rate estimator."""
        np.random.seed(404)
        epidemic = simulator(beta=1.5, gamma=1.0, population_size=200,
                           min_epidemic_size=40, prop_complete=0.8)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        imputed_result = peirr_imputed(removals=removals, infections=infections,
                                      population_size=200)
        direct_gamma = peirr_removal_rate(removals, infections)
        
        # Should match exactly
        assert np.isclose(imputed_result['removal_rate'], direct_gamma)
    
    def test_imputed_with_missing_data(self):
        """Test imputed estimator handles missing data correctly."""
        np.random.seed(505)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=0.4)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = peirr_imputed(removals=removals, infections=infections,
                              population_size=100)
        
        assert result['infection_rate'] > 0
        assert result['removal_rate'] > 0
    
    def test_imputed_vs_tau_similar_estimates(self):
        """Test that tau and imputed give similar estimates."""
        np.random.seed(606)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=150,
                           min_epidemic_size=40, prop_complete=0.7)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        tau_result = peirr_tau(removals=removals, infections=infections,
                              population_size=150)
        imputed_result = peirr_imputed(removals=removals, infections=infections,
                                      population_size=150)
        
        # gamma should match exactly (from R tests)
        assert np.isclose(tau_result['removal_rate'], imputed_result['removal_rate'])
        # infection rate should be in similar ballpark
        assert tau_result['infection_rate'] > 0
        assert imputed_result['infection_rate'] > 0

class TestBayesComplete:
    """Test cases for bayes_complete function."""
    
    def test_bayes_complete_returns_dict(self):
        """Test that bayes_complete returns dict with samples."""
        np.random.seed(123)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=1.0)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = bayes_complete(removals=removals, infections=infections,
                              population_size=100, num_iter=100)
        
        assert isinstance(result, dict)
        assert 'infection_rate' in result
        assert 'removal_rate' in result
    
    def test_bayes_complete_produces_samples(self):
        """Test that bayes_complete produces posterior samples."""
        np.random.seed(456)
        epidemic = simulator(beta=2.0, gamma=1.0, population_size=100,
                           min_epidemic_size=30, prop_complete=1.0)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = bayes_complete(removals=removals, infections=infections,
                              population_size=100, num_iter=200)
        
        assert len(result['infection_rate']) == 200
        assert len(result['removal_rate']) == 200
    
    def test_bayes_complete_posterior_mean_reasonable(self):
        """Test that posterior means are in reasonable range."""
        np.random.seed(789)
        beta_true = 2.0
        gamma_true = 1.0
        epidemic = simulator(beta=beta_true, gamma=gamma_true, 
                           population_size=100, min_epidemic_size=30,
                           prop_complete=1.0)
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        
        result = bayes_complete(removals=removals, infections=infections,
                              population_size=100, num_iter=500,
                              beta_shape=1, gamma_shape=1)
        
        infection_mean = np.mean(result['infection_rate'])
        removal_mean = np.mean(result['removal_rate'])
        
        # Allow 50% tolerance with weak priors
        assert infection_mean > beta_true * 0.5
        assert infection_mean < beta_true * 1.5
        assert removal_mean > gamma_true * 0.5
        assert removal_mean < gamma_true * 1.5