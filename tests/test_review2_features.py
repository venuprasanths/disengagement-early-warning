"""
Automated Test Suite for Review 2 Features:
1. Per-student local explainability with plain-language labels & SHAP
2. Compassionate Restorative Wellness Triage safeguard
3. FERPA-compliant consultation memorandum generator
4. Sensitivity & stress testing framework
5. Deprecated Streamlit width parameter cleanup
6. Probabilistic calibration & reliability curve computation
"""

import os
import pytest
import numpy as np
import pandas as pd

from data.synthetic_data_generator import generate_cohort
from src.features import prepare_feature_matrix, ENGINEERED_FEATURE_NAMES
from src.main_model import (
    TransparentMultiSignalModel,
    FEATURE_PLAIN_LANGUAGE_MAPPING,
    SIGNAL_FAMILY_MAPPING,
    format_plain_interpretation,
)
from src.sensitivity_analysis import (
    evaluate_cohort_scenario,
    SCENARIO_CONFIGURATIONS,
)
from src.backtest import compute_calibration_curve


@pytest.fixture(scope="module")
def trained_model_and_test_data():
    df = generate_cohort(n_students=200, n_weeks=16, seed=42)
    df_train = df[df["week"] <= 10]
    df_test = df[df["week"] >= 11]

    X_train, y_train, _ = prepare_feature_matrix(df_train)
    X_test, y_test, audit_test = prepare_feature_matrix(df_test)

    model = TransparentMultiSignalModel(random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, audit_test


def test_plain_language_mapping_coverage():
    """Verifies every engineered feature has a human-readable plain-language label."""
    for feat in ENGINEERED_FEATURE_NAMES:
        assert feat in FEATURE_PLAIN_LANGUAGE_MAPPING, f"Missing plain-language mapping for feature: {feat}"
        assert len(FEATURE_PLAIN_LANGUAGE_MAPPING[feat]) > 5, f"Label too short for {feat}"
        # Ensure it doesn't contain underscores (plain English for educators)
        assert "_" not in FEATURE_PLAIN_LANGUAGE_MAPPING[feat], f"Label contains raw underscore: {FEATURE_PLAIN_LANGUAGE_MAPPING[feat]}"


def test_plain_language_shap_explainability(trained_model_and_test_data):
    """Verifies explain_instance returns plain-language attributes, impacts, and interpretations."""
    model, X_test, _, _ = trained_model_and_test_data
    instance = X_test.iloc[0]

    explanation = model.explain_instance(instance, background_X=X_test)

    assert "risk_score" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_protective_factors" in explanation
    assert "all_feature_contributions" in explanation
    assert "primary_driver_family" in explanation
    assert explanation["primary_driver_family"] in ["Attendance", "Activity", "Assessment", "Help-Seeking", "Feedback & Sentiment"]

    # Verify top risk drivers structure
    for r in explanation["top_risk_drivers"]:
        assert "plain_feature" in r
        assert "interpretation" in r
        assert "impact" in r
        assert "family" in r
        assert r["impact"] > 0

    # Verify top protective factors structure
    for p in explanation["top_protective_factors"]:
        assert "plain_feature" in p
        assert "interpretation" in p
        assert "impact" in p
        assert "family" in p
        assert p["impact"] < 0


def test_restorative_wellness_triage_safeguard(trained_model_and_test_data):
    """
    Verifies that a student with emotional distress (low pulse morale) but strong attendance
    and zero unexcused absences is classified as RESTO_WELLNESS_CHECK rather than ACADEMIC_INTERVENTION.
    """
    model, X_test, _, _ = trained_model_and_test_data
    # Construct a student profile in emotional distress but perfect attendance
    stu = X_test.iloc[0].copy()
    stu["survey_sentiment_recent"] = -0.75
    stu["confidence_rating_recent"] = 1.0
    stu["att_3wk_mean"] = 5.0
    stu["unexcused_absences_sum"] = 0
    stu["tardy_rate"] = 0.0

    explanation = model.explain_instance(stu, background_X=X_test)
    assert explanation["intervention_pathway"] == "RESTO_WELLNESS_CHECK"
    assert "Wellness" in explanation["pathway_label"]
    assert "emotional well-being" in explanation["outreach_guidance"]


def test_ferpa_counselor_audit_record_export(trained_model_and_test_data):
    """Verifies that export_counselor_audit_record generates a valid FERPA-compliant memorandum."""
    model, X_test, _, _ = trained_model_and_test_data
    stu = X_test.iloc[0]
    explanation = model.explain_instance(stu, background_X=X_test)

    memo = model.export_counselor_audit_record("STU_9999", stu, explanation)
    assert "STU_9999" in memo
    assert "CONFIDENTIAL STUDENT SUPPORT CONSULTATION MEMORANDUM" in memo
    assert "PRIMARY OBSERVATIONAL DRIVERS" in memo
    assert "DEMONSTRATED PROTECTIVE STRENGTHS" in memo
    assert "FERPA & PRIVACY NOTICE" in memo
    # Verify no raw unmapped machine learning column names appear in bullets
    assert "att_3wk_mean:" not in memo
    assert "lms_logins_3wk_mean:" not in memo


def test_sensitivity_scenario_execution():
    """Verifies sensitivity stress testing executes and computes all required performance fields."""
    cfg = {
        "name": "Mini Stress Test",
        "description": "Mini cohort test",
        "archetype_counts": {
            "consistently_engaged": 40,
            "quietly_struggling": 30,
            "checked_out": 20,
            "genuinely_improving": 10,
        },
    }
    res = evaluate_cohort_scenario("mini_test", cfg, seed=42)

    assert "main_model" in res
    assert "baseline_model" in res
    assert "f1" in res["main_model"]
    assert "recall" in res["main_model"]
    assert "precision" in res["main_model"]
    assert "false_alarm_rate" in res["main_model"]
    assert "brier_score" in res["main_model"]
    assert "archetype_breakdown" in res
    assert res["main_model"]["f1"] > res["baseline_model"]["f1"]


def test_no_deprecated_use_container_width():
    """Verifies maintenance cleanup: zero occurrences of use_container_width in dashboard/app.py."""
    app_path = os.path.join("dashboard", "app.py")
    assert os.path.exists(app_path)
    with open(app_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "use_container_width" not in content, "Found deprecated use_container_width in dashboard/app.py!"
    assert 'width="stretch"' in content, "Missing updated width='stretch' in dashboard/app.py!"


def test_probabilistic_calibration_computation():
    """Verifies Expected Calibration Error (ECE) and reliability bin statistics."""
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate([np.linspace(0.05, 0.45, 50), np.linspace(0.55, 0.95, 50)])

    calib = compute_calibration_curve(y_true, y_prob, n_bins=10)
    assert 0.0 <= calib["ece"] <= 0.30
    assert 0.0 <= calib["brier_score"] <= 0.25
    assert len(calib["bins"]) > 0
    for b in calib["bins"]:
        assert 0.0 <= b["avg_confidence"] <= 1.0
        assert 0.0 <= b["true_frequency"] <= 1.0
