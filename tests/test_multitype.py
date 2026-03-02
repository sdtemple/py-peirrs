"""Tests for multitype estimation functions."""

import pytest
import numpy as np
from peirrs.multitype.estimators import (
    peirr_tau_multitype,
    bayes_complete_multitype,
)
from peirrs.multitype.simulate import simulator_multitype


class TestPeirrTauMultitype:
    """Test cases for peirr_tau_multitype function."""
    
    def test_tau_multitype_returns_dict(self):
        """Test that multitype tau estimator returns dict."""
        np.random.seed(123)
        beta = [1.5, 2.0]
        gamma = [1.0, 1.0]
        infection_class_sizes = (50, 50)
        removal_class_sizes = (50, 50)
        
        epidemic = simulator_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes,
            min_epidemic_size=30,
            prop_complete=0.8
        )
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        infection_classes = X[:, 2]
        removal_classes = X[:, 4]
        
        result = peirr_tau_multitype(
            removals=removals,
            infections=infections,
            removal_classes=removal_classes,
            infection_classes=infection_classes,
            infection_class_sizes=infection_class_sizes,
            lag=0
        )
        
        assert isinstance(result, dict)
        assert 'infection_rate' in result
        assert 'removal_rate' in result
    
    def test_tau_multitype_positive_estimates(self):
        """Test that multitype tau estimates are positive."""
        np.random.seed(456)
        beta = [2.0, 2.5]
        gamma = [1.0, 1.2]
        infection_class_sizes = (50, 50)
        removal_class_sizes = (50, 50)
        
        epidemic = simulator_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes,
            min_epidemic_size=30,
            prop_complete=0.8
        )
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        infection_classes = X[:, 2]
        removal_classes = X[:, 4]
        
        result = peirr_tau_multitype(
            removals=removals,
            infections=infections,
            removal_classes=removal_classes,
            infection_classes=infection_classes,
            infection_class_sizes=infection_class_sizes,
            lag=0
        )
        
        assert np.all(result['infection_rate'] > 0)
        assert np.all(result['removal_rate'] > 0)


class TestBayesCompleteMultitype:
    """Test cases for bayes_complete_multitype function."""
    
    def test_bayes_multitype_returns_samples(self):
        """Test that multitype bayes_complete returns samples."""
        np.random.seed(202)
        beta = [2.0, 2.2]
        gamma = [1.0, 1.0]
        infection_class_sizes = (50, 50)
        removal_class_sizes = (50, 50)
        
        epidemic = simulator_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes,
            min_epidemic_size=30,
            prop_complete=1.0
        )
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        infection_classes = X[:, 2]
        removal_classes = X[:, 4]
        
        result = bayes_complete_multitype(
            removals=removals,
            infections=infections,
            removal_classes=removal_classes,
            infection_classes=infection_classes,
            infection_class_sizes=infection_class_sizes,
            beta_init=beta,
            beta_shape=[1, 1],
            gamma_init=gamma,
            gamma_shape=[1, 1],
            num_iter=100
        )
        
        assert isinstance(result, dict)
        assert 'infection_rate' in result
        assert 'removal_rate' in result
        assert result['infection_rate'].shape == (2, 100)
        assert result['removal_rate'].shape == (2, 100)
    
    def test_bayes_multitype_posterior_positive(self):
        """Test that posterior samples are positive."""
        np.random.seed(303)
        beta = [2.0, 2.5]
        gamma = [1.0, 1.2]
        infection_class_sizes = (50, 50)
        removal_class_sizes = (50, 50)
        
        epidemic = simulator_multitype(
            beta=beta,
            gamma=gamma,
            infection_class_sizes=infection_class_sizes,
            removal_class_sizes=removal_class_sizes,
            min_epidemic_size=30,
            prop_complete=1.0
        )
        
        X = epidemic['matrix_time']
        removals = X[:, 1]
        infections = X[:, 0]
        infection_classes = X[:, 2]
        removal_classes = X[:, 4]
        
        result = bayes_complete_multitype(
            removals=removals,
            infections=infections,
            removal_classes=removal_classes,
            infection_classes=infection_classes,
            infection_class_sizes=infection_class_sizes,
            beta_init=beta,
            beta_shape=[1, 1],
            gamma_init=gamma,
            gamma_shape=[1, 1],
            num_iter=100
        )
        
        assert np.all(result['infection_rate'] > 0)
        assert np.all(result['removal_rate'] > 0)