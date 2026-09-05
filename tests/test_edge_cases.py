"""
Automated Test Suite for Edge and Failure Cases.

Tests the 3 required edge cases:
1. Transfer Student with Sparse/Missing Historical Records (Widening Uncertainty Band).
2. Signal Gamer (Excessive Logins but Zero Comprehension / Content Time).
3. Temporary Acute Shock (Family Emergency with Rapid Rebound - Non-Overreaction).
"""

import pytest
import numpy as np
import pandas as pd
from data.synthetic_data_generator import generate_cohort
from src.features import prepare_feature_matrix, extract_student_features
from src.baseline_model import LaggingAttendanceMarksBaseline
from src.main_model import TransparentMultiSignalModel
from src.uncertainty import BootstrappedUncertaintyEstimator


@pytest.fixture(scope="module")
def trained_models():
    """Generates cohort, splits temporally, and trains models on weeks 1-10."""
    df = generate_cohort(n_students=500, n_weeks=16, seed=42)
    df_train = df[df["week"] <= 10].copy()
    X_train, y_train, _ = prepare_feature_matrix(df_train)

    main_model = TransparentMultiSignalModel(random_state=42)
    main_model.fit(X_train, y_train)

    uncertainty_est = BootstrappedUncertaintyEstimator(n_bootstraps=8, random_state=42)
    uncertainty_est.fit(X_train, y_train)

    baseline = LaggingAttendanceMarksBaseline()

    return {
        "df": df,
        "main_model": main_model,
        "uncertainty_est": uncertainty_est,
        "baseline": baseline,
    }


def test_edge_case_1_transfer_student_uncertainty(trained_models):
    """
    Edge Case 1: Transfer student with sparse data (enrolled late at week 7).
    Model must handle missing historical weeks gracefully and widen the
    uncertainty interval to signal lower confidence to counselors.
    """
    df = trained_models["df"]
    uncertainty_est = trained_models["uncertainty_est"]

    # Filter for transfer student archetype
    transfer_df = df[df["archetype"] == "transfer_student"].copy()
    assert not transfer_df.empty, "Transfer students must be present in cohort."

    # Verify student has no records for weeks 1-6
    min_week = transfer_df["week"].min()
    assert min_week == 7, f"Transfer student should begin at week 7, found week {min_week}."

    # Extract features for a transfer student at week 8 (only 2 weeks of observations)
    s_id = transfer_df["student_id"].iloc[0]
    s_records = transfer_df[transfer_df["student_id"] == s_id].sort_values("week")
    s_feats = extract_student_features(s_records)

    row_wk8 = s_feats[s_feats["week"] == 8].iloc[0]
    uncertainty_info = uncertainty_est.compute_instance_uncertainty(row_wk8, weeks_available=2)

    # Uncertainty band must be wide due to sparse history penalty
    assert uncertainty_info["sparse_data_flag"] is True
    assert uncertainty_info["interval_width"] >= 0.20, (
        f"Expected wide interval for sparse transfer student, got {uncertainty_info['interval_width']}"
    )
    assert "Low Confidence" in uncertainty_info["confidence_label"] or "Moderate" in uncertainty_info["confidence_label"]


def test_edge_case_2_signal_gamer_detection(trained_models):
    """
    Edge Case 2: Signal gamer who attempts to game activity metrics by opening 60+ LMS logins
    per week, but spends negligible time actually studying (5-15 mins) and fails quizzes.
    The multi-signal model must detect the divergence via cross-signal features and flag risk.
    """
    df = trained_models["df"]
    main_model = trained_models["main_model"]

    gamer_df = df[df["archetype"] == "signal_gamer"].copy()
    assert not gamer_df.empty, "Signal gamer students must be present in cohort."

    s_id = gamer_df["student_id"].iloc[0]
    s_records = gamer_df[gamer_df["student_id"] == s_id].sort_values("week")
    s_feats = extract_student_features(s_records)

    # In week 12, check model prediction on this gamer
    wk12_row = s_feats[s_feats["week"] == 12].iloc[0]
    X_single = wk12_row[main_model.feature_names].to_frame().T

    risk_score = float(main_model.predict_risk_score(X_single)[0])
    # The divergence feature logins_per_active_hour should be very high
    assert wk12_row["logins_per_active_hour"] > 100.0, (
        f"Gamer should exhibit extreme logins per active hour, got {wk12_row['logins_per_active_hour']}"
    )
    # The model must flag high risk (>= 0.50) despite the high login count
    assert risk_score >= 0.50, f"Expected gamer to be flagged at-risk, got risk score {risk_score}"


def test_edge_case_3_temporary_acute_shock_non_overreaction(trained_models):
    """
    Edge Case 3: Acute temporary shock (e.g. family emergency) at Week 7.
    Attendance and quiz score drop sharply for 1 week, but rebound in weeks 8-9,
    accompanied by an explanatory teacher note.
    The system must NOT permanently flag this student as chronically disengaged.
    """
    df = trained_models["df"]
    main_model = trained_models["main_model"]

    shock_df = df[df["archetype"] == "acute_shock"].copy()
    assert not shock_df.empty, "Acute shock students must be present in cohort."

    s_id = shock_df["student_id"].iloc[0]
    s_records = shock_df[shock_df["student_id"] == s_id].sort_values("week")
    s_feats = extract_student_features(s_records)

    # At week 7, attendance drops to 0 or 1
    wk7_row = s_records[s_records["week"] == 7].iloc[0]
    assert wk7_row["days_present"] <= 1, "Acute shock must exhibit 1-week attendance drop."

    # By week 10, the student has rebounded
    wk10_row = s_feats[s_feats["week"] == 10].iloc[0]
    X_wk10 = wk10_row[main_model.feature_names].to_frame().T
    risk_score_wk10 = float(main_model.predict_risk_score(X_wk10)[0])

    # Chronic disengagement requires sustained withdrawal; post-recovery risk should be low (< 0.40)
    assert risk_score_wk10 < 0.40, (
        f"Model should not overreact to transient shock once student recovers, got {risk_score_wk10}"
    )


def test_information_barrier_enforcement():
    """
    Verifies that passing any ground-truth column to the model or feature matrix
    strictly raises an Information Barrier Exception.
    """
    from src.schema import assert_no_ground_truth_leakage

    # Allowed features
    valid_cols = ["days_present", "lms_logins", "quiz_score", "survey_sentiment"]
    assert_no_ground_truth_leakage(valid_cols)  # Should not raise

    # Leaked columns must raise ValueError
    with pytest.raises(ValueError, match="CRITICAL INFORMATION BARRIER VIOLATION"):
        assert_no_ground_truth_leakage(["days_present", "latent_engagement"])

    with pytest.raises(ValueError, match="CRITICAL INFORMATION BARRIER VIOLATION"):
        assert_no_ground_truth_leakage(["archetype", "lms_logins"])

    with pytest.raises(ValueError, match="CRITICAL INFORMATION BARRIER VIOLATION"):
        assert_no_ground_truth_leakage(["is_disengaged", "quiz_score"])
