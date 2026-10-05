"""
Multi-Year Longitudinal Population Drift & Performance Degradation Simulator.

Simulates 3 consecutive academic school years (Years 1, 2, and 3) to model:
1. Grade-level demographic transitions (e.g. 9th-grade transition shock).
2. Curriculum policy shifts and changing digital learning tool adoption.
3. Covariate feature drift using Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests.
4. Model performance decay when operating unretrained vs. under an active annual retraining policy.
5. Production retraining triggers and MLOps alert thresholds.
"""

import os
import json
import numpy as np
import sys
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, recall_score, precision_score, brier_score_loss, roc_auc_score

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.synthetic_data_generator import generate_cohort
from src.features import prepare_feature_matrix, ENGINEERED_FEATURE_NAMES
from src.main_model import TransparentMultiSignalModel
from src.baseline_model import LaggingAttendanceMarksBaseline


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Computes the Population Stability Index (PSI) between a reference and target distribution.

    Mathematical Formulation:
        PSI = sum_{k=1}^K (Actual_k - Expected_k) * ln(Actual_k / Expected_k)

    Standard Educational MLOps Thresholds:
        - PSI < 0.10: Insignificant drift (Model stable, no action required)
        - 0.10 <= PSI < 0.25: Moderate drift (Monitor closely, schedule scheduled retrain)
        - PSI >= 0.25: Significant drift (Critical trigger, mandatory immediate retraining)
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantiles based on reference (expected) distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Handle duplicate bin edges in discrete data
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) <= 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=unique_edges)
    act_counts, _ = np.histogram(actual, bins=unique_edges)

    exp_pct = exp_counts / float(len(expected))
    act_pct = act_counts / float(len(actual))

    # Add epsilon to prevent log(0) or division by zero
    exp_pct = np.clip(exp_pct, epsilon, None)
    act_pct = np.clip(act_pct, epsilon, None)

    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.round(psi_val, 4))


def calculate_ks_statistic(expected: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    """
    Computes two-sample Kolmogorov-Smirnov test statistic and p-value.
    Measures the maximum distance between cumulative empirical distributions.
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, 1.0
    res = ks_2samp(expected, actual)
    return float(np.round(res.statistic, 4)), float(np.round(res.pvalue, 6))


# Cohort configurations representing 3 consecutive academic years
MULTI_YEAR_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Year_1": {
        "name": "Academic Year 1 (Baseline Reference Cohort)",
        "description": "Standard balanced high school cohort composition.",
        "seed": 101,
        "archetype_counts": {
            "consistently_engaged": 250,
            "quietly_struggling": 100,
            "checked_out": 75,
            "genuinely_improving": 50,
            "transfer_student": 10,
            "signal_gamer": 8,
            "acute_shock": 7,
        },
    },
    "Year_2": {
        "name": "Academic Year 2 (Curriculum Rigor Increase & Remote Hybrid Shifts)",
        "description": "Increased STEM difficulty increases quiet struggle by 25%; digital tool usage expands.",
        "seed": 202,
        "archetype_counts": {
            "consistently_engaged": 220,
            "quietly_struggling": 125,  # +25% quiet struggle
            "checked_out": 80,
            "genuinely_improving": 45,
            "transfer_student": 12,
            "signal_gamer": 10,
            "acute_shock": 8,
        },
    },
    "Year_3": {
        "name": "Academic Year 3 (Grade 9 Transition Shock & Post-Disruption Cohort)",
        "description": "Freshman transition shock: elevated quiet struggle (155) and acute stress events (25).",
        "seed": 303,
        "archetype_counts": {
            "consistently_engaged": 190,
            "quietly_struggling": 155,  # +55% quiet struggle vs Y1
            "checked_out": 85,
            "genuinely_improving": 35,
            "transfer_student": 15,
            "signal_gamer": 12,
            "acute_shock": 25,  # 3.5x acute shock
        },
    },
}


class LongitudinalDriftSimulator:
    """
    Multi-year simulator and MLOps drift monitoring engine.
    """

    def __init__(self, years: Optional[List[str]] = None):
        self.years = years or ["Year_1", "Year_2", "Year_3"]
        self.cohort_data: Dict[str, pd.DataFrame] = {}
        self.feature_matrices: Dict[str, Tuple[pd.DataFrame, pd.Series, pd.DataFrame]] = {}

    def generate_multi_year_cohorts(self) -> Dict[str, pd.DataFrame]:
        """Generates synthetic cohort datasets across the simulated academic years."""
        for yr_key in self.years:
            cfg = MULTI_YEAR_CONFIGS[yr_key]
            df_yr = generate_cohort(seed=cfg["seed"], archetype_counts=cfg["archetype_counts"])
            self.cohort_data[yr_key] = df_yr
            self.feature_matrices[yr_key] = prepare_feature_matrix(df_yr)
        return self.cohort_data

    def evaluate_multi_year_drift(self) -> Dict[str, Any]:
        """
        Executes drift analysis across Year 1 (reference) -> Year 2 -> Year 3.
        Compares:
          1. Feature-level PSI and KS drift against Year 1 reference.
          2. Legacy (Unretrained Year 1 Model) performance across all years.
          3. Retrained (In-Domain Annual Retrained Model) performance across all years.
        """
        if not self.cohort_data:
            self.generate_multi_year_cohorts()

        # Step 1: Train reference Year 1 model on Weeks 1-10
        y1_df = self.cohort_data["Year_1"]
        y1_train = y1_df[y1_df["week"] <= 10]
        X1_train, y1_train_labels, _ = prepare_feature_matrix(y1_train)

        legacy_model = TransparentMultiSignalModel(random_state=42)
        legacy_model.fit(X1_train, y1_train_labels)

        X1_full, _, _ = self.feature_matrices["Year_1"]

        yearly_results = {}

        for yr_key in self.years:
            cfg = MULTI_YEAR_CONFIGS[yr_key]
            df_curr = self.cohort_data[yr_key]
            X_curr, y_curr, audit_curr = self.feature_matrices[yr_key]

            # Holdout test set for this academic year (Weeks 11–16)
            test_mask = df_curr["week"] >= 11
            X_test_yr = X_curr.loc[test_mask]
            y_test_yr = y_curr.loc[test_mask].to_numpy()

            # Baseline performance on this year
            df_test_curr = df_curr.loc[test_mask]
            baseline = LaggingAttendanceMarksBaseline()
            b_preds = baseline.predict(df_test_curr)
            b_risk = baseline.predict_risk_score(df_test_curr)
            b_f1 = float(f1_score(y_test_yr, b_preds, zero_division=0))
            b_rec = float(recall_score(y_test_yr, b_preds, zero_division=0))
            b_prec = float(precision_score(y_test_yr, b_preds, zero_division=0))

            # Paradigm A: Unretrained Legacy Model (Trained on Year 1 only)
            legacy_scores = legacy_model.predict_risk_score(X_test_yr)
            legacy_preds = legacy_scores >= 0.50
            leg_f1 = float(f1_score(y_test_yr, legacy_preds, zero_division=0))
            leg_rec = float(recall_score(y_test_yr, legacy_preds, zero_division=0))
            leg_prec = float(precision_score(y_test_yr, legacy_preds, zero_division=0))
            leg_brier = float(brier_score_loss(y_test_yr, legacy_scores))
            leg_auc = float(roc_auc_score(y_test_yr, legacy_scores))
            neg_mask = y_test_yr == 0
            leg_fpr = float(np.mean(legacy_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0

            # Paradigm B: Annual Retrained Model (Trained on current year Weeks 1–10)
            train_mask = df_curr["week"] <= 10
            X_train_yr = X_curr.loc[train_mask]
            y_train_yr = y_curr.loc[train_mask]

            retrained_model = TransparentMultiSignalModel(random_state=cfg["seed"])
            retrained_model.fit(X_train_yr, y_train_yr)
            retrained_scores = retrained_model.predict_risk_score(X_test_yr)
            retrained_preds = retrained_scores >= 0.50
            ret_f1 = float(f1_score(y_test_yr, retrained_preds, zero_division=0))
            ret_rec = float(recall_score(y_test_yr, retrained_preds, zero_division=0))
            ret_prec = float(precision_score(y_test_yr, retrained_preds, zero_division=0))
            ret_brier = float(brier_score_loss(y_test_yr, retrained_scores))
            ret_auc = float(roc_auc_score(y_test_yr, retrained_scores))
            ret_fpr = float(np.mean(retrained_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0

            # Feature Drift vs. Year 1 Reference
            feature_drift_dict = {}
            feature_psis = []
            for feat in ENGINEERED_FEATURE_NAMES:
                exp_vals = X1_full[feat].to_numpy()
                act_vals = X_curr[feat].to_numpy()
                psi_f = calculate_psi(exp_vals, act_vals)
                ks_stat, p_val = calculate_ks_statistic(exp_vals, act_vals)
                feature_drift_dict[feat] = {
                    "psi": psi_f,
                    "ks_statistic": ks_stat,
                    "ks_pvalue": p_val,
                }
                feature_psis.append(psi_f)

            mean_cohort_psi = float(np.round(np.mean(feature_psis), 4))
            max_drift_feature = max(feature_drift_dict.items(), key=lambda item: item[1]["psi"])

            # Determine Retraining Trigger Status
            if mean_cohort_psi < 0.10:
                drift_status = "STABLE"
                retrain_action = "CONTINUE_MONITORING"
            elif mean_cohort_psi < 0.20:
                drift_status = "MODERATE_DRIFT"
                retrain_action = "SCHEDULE_SEMESTER_RETRAIN"
            else:
                drift_status = "CRITICAL_DRIFT"
                retrain_action = "MANDATORY_IMMEDIATE_RETRAIN"

            yearly_results[yr_key] = {
                "name": cfg["name"],
                "description": cfg["description"],
                "disengagement_base_rate": float(np.round(np.mean(y_test_yr), 3)),
                "mean_cohort_psi": mean_cohort_psi,
                "drift_status": drift_status,
                "retrain_action": retrain_action,
                "max_drift_feature": {
                    "feature": max_drift_feature[0],
                    "psi": max_drift_feature[1]["psi"],
                    "ks_stat": max_drift_feature[1]["ks_statistic"],
                },
                "legacy_model_unretrained": {
                    "f1": round(leg_f1, 4),
                    "recall": round(leg_rec, 4),
                    "precision": round(leg_prec, 4),
                    "false_alarm_rate": round(leg_fpr, 4),
                    "brier_score": round(leg_brier, 4),
                    "roc_auc": round(leg_auc, 4),
                },
                "retrained_annual_model": {
                    "f1": round(ret_f1, 4),
                    "recall": round(ret_rec, 4),
                    "precision": round(ret_prec, 4),
                    "false_alarm_rate": round(ret_fpr, 4),
                    "brier_score": round(ret_brier, 4),
                    "roc_auc": round(ret_auc, 4),
                },
                "lagging_baseline": {
                    "f1": round(b_f1, 4),
                    "recall": round(b_rec, 4),
                    "precision": round(b_prec, 4),
                },
            }

        return yearly_results


def run_drift_analysis(output_path: str = "data/drift_analysis_results.json") -> Dict[str, Any]:
    """Runs longitudinal simulation and writes results to JSON artifact."""
    simulator = LongitudinalDriftSimulator()
    results = simulator.evaluate_multi_year_drift()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = run_drift_analysis()
    print("Multi-Year Longitudinal Drift Analysis Completed Successfully:")
    for yr, res in out.items():
        print(f"[{yr}] Mean PSI: {res['mean_cohort_psi']} | Status: {res['drift_status']} | Legacy F1: {res['legacy_model_unretrained']['f1']} -> Retrained F1: {res['retrained_annual_model']['f1']}")
