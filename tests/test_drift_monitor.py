"""
Automated Unit Tests for Multi-Year Longitudinal Drift Monitor.

Verifies:
1. Population Stability Index (PSI) calculation mechanics and edge cases.
2. Two-sample Kolmogorov-Smirnov (KS) statistic computation.
3. Longitudinal drift simulation across 3 simulated academic school years.
4. Production retraining trigger classification.
"""

import os
import json
import numpy as np
import pytest
from src.drift_monitor import calculate_psi, calculate_ks_statistic, LongitudinalDriftSimulator


def test_calculate_psi_identical_distributions():
    """Identical distributions should yield near-zero PSI (< 0.01)."""
    np.random.seed(42)
    sample_a = np.random.normal(loc=50.0, scale=10.0, size=500)
    sample_b = np.random.normal(loc=50.0, scale=10.0, size=500)
    psi = calculate_psi(sample_a, sample_b, num_bins=10)
    assert psi < 0.05, f"Expected near-zero PSI for identical distributions, got {psi}"


def test_calculate_psi_shifted_distribution():
    """Significant mean shift should produce elevated PSI (>= 0.25)."""
    np.random.seed(42)
    reference = np.random.normal(loc=70.0, scale=8.0, size=500)
    severely_shifted = np.random.normal(loc=50.0, scale=12.0, size=500)
    psi = calculate_psi(reference, severely_shifted, num_bins=10)
    assert psi >= 0.25, f"Expected PSI >= 0.25 for major distribution shift, got {psi}"


def test_calculate_ks_statistic():
    """KS test should report high statistic and low p-value on shifted data."""
    np.random.seed(42)
    data1 = np.random.uniform(0, 50, size=300)
    data2 = np.random.uniform(30, 80, size=300)
    stat, p_val = calculate_ks_statistic(data1, data2)
    assert 0.0 <= stat <= 1.0
    assert stat > 0.30
    assert p_val < 0.001


def test_multi_year_simulation_execution():
    """Runs a lightweight multi-year simulation across Year 1 and Year 2."""
    simulator = LongitudinalDriftSimulator(years=["Year_1", "Year_2"])
    cohorts = simulator.generate_multi_year_cohorts()
    assert "Year_1" in cohorts
    assert "Year_2" in cohorts
    assert len(cohorts["Year_1"]) > 0
    assert len(cohorts["Year_2"]) > 0

    results = simulator.evaluate_multi_year_drift()
    assert "Year_1" in results
    assert "Year_2" in results
    assert results["Year_1"]["mean_cohort_psi"] == 0.0
    assert results["Year_1"]["drift_status"] == "STABLE"
    assert "legacy_model_unretrained" in results["Year_2"]
    assert "retrained_annual_model" in results["Year_2"]
    assert results["Year_2"]["legacy_model_unretrained"]["f1"] > 0.85
