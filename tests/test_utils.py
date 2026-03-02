"""Tests for utility functions in peirrs.utils."""

import pytest
import numpy as np
from peirrs.utils import sort_sem, filter_sem, decomplete_sem, tau_moment


class TestSortSem:
    """Test cases for sort_sem function."""
    
    def test_sort_sem_sorts_2col_by_removal_times(self):
        """Test that sort_sem correctly sorts 2-column matrix by removal times."""
        infections = np.array([3, 1, 5, 2, 4], dtype=float)
        removals = np.array([7, 4, 9, 5, 8], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        result = sort_sem(matrix_time)
        
        # Check that removals are sorted
        assert np.allclose(result[:, 1], np.array([4, 5, 7, 8, 9]))
        # Check that infections were reordered correctly
        assert np.allclose(result[:, 0], np.array([1, 2, 3, 4, 5]))
    
    def test_sort_sem_preserves_shape_2col(self):
        """Test that sort_sem preserves shape for 2-column input."""
        infections = np.array([1, 2, 3], dtype=float)
        removals = np.array([5, 4, 6], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        result = sort_sem(matrix_time)
        
        assert result.shape == (3, 2)
        assert isinstance(result, np.ndarray)
    
    def test_sort_sem_sorts_6col_by_removal_times(self):
        """Test that sort_sem correctly sorts 6-column multitype matrix."""
        infections = np.array([3, 1, 5, 2, 4], dtype=float)
        removals = np.array([7, 4, 9, 5, 8], dtype=float)
        inf_classes = np.array([1, 2, 1, 2, 1], dtype=float)
        inf_rates = np.array([2.5, 3.0, 2.5, 3.0, 2.5], dtype=float)
        rem_classes = np.array([1, 1, 2, 2, 1], dtype=float)
        rem_rates = np.array([1.0, 1.0, 1.5, 1.5, 1.0], dtype=float)
        
        matrix_time = np.column_stack([infections, removals, inf_classes,
                                       inf_rates, rem_classes, rem_rates])
        
        result = sort_sem(matrix_time)
        
        # Check that removals are sorted (column 1)
        assert np.allclose(result[:, 1], np.array([4, 5, 7, 8, 9]))
        # Check that infections were reordered consistently
        assert np.allclose(result[:, 0], np.array([1, 2, 3, 4, 5]))
        assert result.shape == (5, 6)
    
    def test_sort_sem_rejects_invalid_dimensions(self):
        """Test that sort_sem rejects non-2D input."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            sort_sem(data)
    
    def test_sort_sem_rejects_invalid_columns(self):
        """Test that sort_sem rejects matrices with wrong column count."""
        data = np.array([[1, 2, 3, 4, 5]])
        with pytest.raises(ValueError):
            sort_sem(data)


class TestFilterSem:
    """Test cases for filter_sem function."""
    
    def test_filter_sem_removes_all_inf_2col(self):
        """Test that filter_sem removes rows with all Inf values."""
        infections = np.array([1, 2, np.inf, 4, np.inf], dtype=float)
        removals = np.array([3, 4, np.inf, 6, np.inf], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        result = filter_sem(matrix_time)
        
        # Should keep only rows where at least one time is finite
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], np.array([1, 2, 4]))
        assert np.allclose(result[:, 1], np.array([3, 4, 6]))
    
    def test_filter_sem_preserves_shape_2col(self):
        """Test that filter_sem preserves shape for valid 2-column input."""
        infections = np.array([1, 2, 3], dtype=float)
        removals = np.array([5, np.inf, 6], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        result = filter_sem(matrix_time)
        
        # Should keep all rows since infection time is finite for row 2
        assert result.shape == (3, 2)
        assert isinstance(result, np.ndarray)
    
    def test_filter_sem_removes_all_inf_6col(self):
        """Test that filter_sem correctly removes rows from 6-column input."""
        infections = np.array([1, 2, np.inf, 4, np.inf], dtype=float)
        removals = np.array([3, 4, np.inf, 6, np.inf], dtype=float)
        inf_classes = np.array([1, 1, 2, 2, 1], dtype=float)
        inf_rates = np.array([2.5, 2.5, 3.0, 3.0, 2.5], dtype=float)
        rem_classes = np.array([1, 1, 2, 2, 1], dtype=float)
        rem_rates = np.array([1.0, 1.0, 1.5, 1.5, 1.0], dtype=float)
        
        matrix_time = np.column_stack([infections, removals, inf_classes,
                                       inf_rates, rem_classes, rem_rates])
        
        result = filter_sem(matrix_time)
        
        # Should filter out rows 3 and 5 (both Inf times)
        assert result.shape[0] == 3
        assert result.shape[1] == 6
    
    def test_filter_sem_rejects_invalid_dimensions(self):
        """Test that filter_sem rejects non-2D input."""
        data = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            filter_sem(data)


class TestDecompleteSem:
    """Test cases for decomplete_sem function."""
    
    def test_decomplete_sem_default_keeps_complete(self):
        """Test that decomplete_sem with default params keeps data complete."""
        infections = np.array([1, 2, 3], dtype=float)
        removals = np.array([5, 6, 7], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        result = decomplete_sem(matrix_time, prop_complete=1)
        
        # With default prop_complete=1.0, all data should remain complete
        assert np.all(np.isfinite(result))
    
    def test_decomplete_sem_introduces_missing(self):
        """Test that decomplete_sem introduces NaN with prop_complete < 1."""
        infections = np.array([1, 2, 3], dtype=float)
        removals = np.array([5, 6, 7], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        np.random.seed(42)
        result = decomplete_sem(matrix_time, prop_complete=0.0, 
                               prop_infection_missing=0.5)
        
        # With prop_complete=0.0, some NaN values should be introduced
        assert np.any(np.isnan(result))
    
    def test_decomplete_sem_respects_prop_complete(self):
        """Test that observed proportion of complete pairs matches expectation."""
        infections = np.arange(1, 101, dtype=float)
        removals = np.arange(10, 110, dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        np.random.seed(42)
        result = decomplete_sem(matrix_time, prop_complete=0.5)
        
        # Count complete pairs
        complete_mask = np.isfinite(result[:, 0]) & np.isfinite(result[:, 1])
        observed_prop = np.sum(complete_mask) / len(complete_mask)
        
        # Should be approximately 0.5 (with some tolerance due to randomness)
        assert 0.3 < observed_prop < 0.7
    
    def test_decomplete_sem_respects_prop_infection_missing(self):
        """Test that prop_infection_missing controls which time is missing."""
        infections = np.arange(1, 101, dtype=float)
        removals = np.arange(10, 110, dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        np.random.seed(42)
        result = decomplete_sem(matrix_time, prop_complete=0.0,
                               prop_infection_missing=1.0)
        
        # With prop_infection_missing=1.0, only infection times should be NaN
        # Count infections missing vs removals missing
        inf_missing = np.isnan(result[:, 0]) & np.isfinite(result[:, 1])
        rem_missing = np.isfinite(result[:, 0]) & np.isnan(result[:, 1])
        
        # Should have some infection missing, none removal missing
        assert np.sum(inf_missing) > 0
        assert np.sum(rem_missing) == 0
    
    def test_decomplete_sem_rejects_invalid_params(self):
        """Test that decomplete_sem validates input parameters."""
        infections = np.array([1, 2, 3], dtype=float)
        removals = np.array([5, 6, 7], dtype=float)
        matrix_time = np.column_stack([infections, removals])
        
        with pytest.raises(ValueError):
            decomplete_sem(matrix_time, prop_complete=1.5)
        
        with pytest.raises(ValueError):
            decomplete_sem(matrix_time, prop_infection_missing=-0.1)


# check this
class TestTauMoment:
    """Test cases for tau_moment function."""
    
    def test_tau_moment_complete_data_ij_before_ik(self):
        """Test tau_moment with complete data where ij < ik."""
        # If j is infected before k is infectious, tau = 0
        tau = tau_moment(rk=5.0, rj=4.0, ik=2.0, ij=1.0,
                        lambdak=1.0, lambdaj=1.0)
        assert tau == 0.0
    
    def test_tau_moment_complete_data_ij_after_rk(self):
        """Test tau_moment with complete data where ij > rk."""
        # If j is infected after k recovers, tau = rk - ik
        tau = tau_moment(rk=3.0, rj=5.0, ik=1.0, ij=4.0,
                        lambdak=1.0, lambdaj=1.0)
        assert np.isclose(tau, 2.0)  # rk - ik = 3 - 1 = 2
    
    def test_tau_moment_complete_data_ij_between_ik_rk(self):
        """Test tau_moment with complete data where ik < ij < rk."""
        # tau should be the time from ij to ij (i.e. 0) or up to rk
        tau = tau_moment(rk=5.0, rj=6.0, ik=2.0, ij=3.0,
                        lambdak=1.0, lambdaj=1.0)
        # tau = ij - ik = 3 - 2 = 1
        assert np.isclose(tau, 1.0)
    
    def test_tau_moment_rejects_both_rk_ik_missing(self):
        """Test that tau_moment rejects when both rk and ik are missing."""
        with pytest.raises(ValueError, match="Both rk and ik"):
            tau_moment(rk=np.nan, rj=4.0, ik=np.nan, ij=3.0,
                      lambdak=1.0, lambdaj=1.0)
    
    def test_tau_moment_rejects_both_rj_ij_missing(self):
        """Test that tau_moment rejects when both rj and ij are missing."""
        with pytest.raises(ValueError, match="Both rj and ij"):
            tau_moment(rk=5.0, rj=np.nan, ik=2.0, ij=np.nan,
                      lambdak=1.0, lambdaj=1.0)
    
    def test_tau_moment_rejects_invalid_rates(self):
        """Test that tau_moment rejects invalid rate parameters."""
        with pytest.raises(ValueError, match="lambdak must be positive"):
            tau_moment(rk=5.0, rj=4.0, ik=2.0, ij=3.0,
                      lambdak=-1.0, lambdaj=1.0)
        
        with pytest.raises(ValueError, match="lambdaj must be positive"):
            tau_moment(rk=5.0, rj=4.0, ik=2.0, ij=3.0,
                      lambdak=1.0, lambdaj=0.0)
    
    def test_tau_moment_rejects_negative_lag(self):
        """Test that tau_moment rejects negative lag."""
        with pytest.raises(ValueError, match="lag must be non-negative"):
            tau_moment(rk=5.0, rj=4.0, ik=2.0, ij=3.0,
                      lambdak=1.0, lambdaj=1.0, lag=-0.5)
    
    def test_tau_moment_with_lag(self):
        """Test that tau_moment handles lag parameter correctly."""
        # tau_moment adjusts infection times when lag > 0
        tau = tau_moment(rk=5.0, rj=np.nan, ik=2.0, ij=3.0,
                        lambdak=1.0, lambdaj=1.0, lag=0.5)
        # Adjusted ij = 3.0 - 0.5 = 2.5
        assert tau >= 0  # Should be non-negative
    
    def test_tau_moment_returns_non_negative(self):
        """Test that tau_moment always returns non-negative values."""
        # Test various missingness patterns
        patterns = [
            # (rk, rj, ik, ij)
            (5.0, 4.0, 2.0, 3.0),  # Complete
            (5.0, 4.0, np.nan, np.nan),  # Both times missing
            (np.nan, 4.0, 2.0, np.nan),  # rk, ij missing (Case 2)
            (5.0, 4.0, 2.0, np.nan),  # ij missing
        ]
        
        for rk, rj, ik, ij in patterns:
            tau = tau_moment(rk, rj, ik, ij, lambdak=1.0, lambdaj=1.0)
            assert tau >= 0, f"tau is negative for pattern ({rk}, {rj}, {ik}, {ij})"
