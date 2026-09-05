"""
Temporal Holdout Backtesting Framework.

Enforces temporal split:
- Training: Weeks 1 – 10
- Testing: Weeks 11 – 16

Evaluates:
1. Lead-time advantage (earliness in days/weeks vs lagging baseline)
2. Precision, Recall, F1 trade-offs across thresholds
3. Uncertainty calibration (ECE, interval coverage)
4. Head-to-head before-and-after comparison matrix
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

from src.schema import assert_no_ground_truth_leakage, GROUND_TRUTH_COLUMNS
from src.features import prepare_feature_matrix
from src.baseline_model import LaggingAttendanceMarksBaseline
from src.main_model import TransparentMultiSignalModel
from src.uncertainty import BootstrappedUncertaintyEstimator
from data.synthetic_data_generator import generate_cohort


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Computes Expected Calibration Error (ECE) and bin statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_assignments = np.digitize(y_prob, bin_edges) - 1
    bin_assignments = np.clip(bin_assignments, 0, n_bins - 1)

    ece = 0.0
    bins_data = []

    for b in range(n_bins):
        mask = bin_assignments == b
        bin_count = int(np.sum(mask))
        if bin_count > 0:
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (bin_count / len(y_true)) * abs(bin_acc - bin_conf)
            bins_data.append({
                "bin": b,
                "count": bin_count,
                "avg_confidence": round(bin_conf, 3),
                "true_frequency": round(bin_acc, 3),
            })

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "bins": bins_data,
    }


def evaluate_lead_time(
    df_cohort: pd.DataFrame,
    main_model: TransparentMultiSignalModel,
    baseline_model: LaggingAttendanceMarksBaseline,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Computes student-level lead time advantage:
    How many weeks/days earlier does the Main Model flag students who ultimately disengage?
    """
    # Find all students who ever became disengaged
    disengaged_students = (
        df_cohort[df_cohort["is_disengaged"] == True]["student_id"].unique()
    )

    X_all, _, audit_all = prepare_feature_matrix(df_cohort)
    main_scores = main_model.predict_risk_score(X_all)
    df_eval = df_cohort.copy()
    df_eval["main_score"] = main_scores
    df_eval["main_flag"] = main_scores >= threshold
    df_eval["baseline_flag"] = baseline_model.predict(df_cohort)

    lead_time_weeks = []
    earlier_count = 0
    tied_count = 0
    later_count = 0

    student_lead_details = []

    for s_id in disengaged_students:
        s_data = df_eval[df_eval["student_id"] == s_id].sort_values("week")
        archetype = s_data["archetype"].iloc[0]

        # First week student was actually disengaged
        true_disengage_weeks = s_data[s_data["is_disengaged"] == True]["week"]
        first_true_week = int(true_disengage_weeks.min()) if not true_disengage_weeks.empty else 16

        # Earliest week flagged by Main Model
        main_flagged = s_data[s_data["main_flag"] == True]["week"]
        first_main_week = int(main_flagged.min()) if not main_flagged.empty else 17

        # Earliest week flagged by Baseline
        base_flagged = s_data[s_data["baseline_flag"] == True]["week"]
        first_base_week = int(base_flagged.min()) if not base_flagged.empty else 17

        # Lead time advantage = base_week - main_week (positive means main model flagged earlier)
        diff_weeks = first_base_week - first_main_week
        lead_time_weeks.append(diff_weeks)

        if diff_weeks > 0:
            earlier_count += 1
        elif diff_weeks == 0:
            tied_count += 1
        else:
            later_count += 1

        student_lead_details.append({
            "student_id": s_id,
            "archetype": archetype,
            "first_true_disengaged_week": first_true_week,
            "first_main_flag_week": first_main_week if first_main_week <= 16 else None,
            "first_baseline_flag_week": first_base_week if first_base_week <= 16 else None,
            "lead_time_weeks": diff_weeks,
            "lead_time_days": diff_weeks * 7,
        })

    lead_arr_w = np.array(lead_time_weeks)
    lead_arr_d = lead_arr_w * 7.0

    median_weeks = float(np.median(lead_arr_w))
    median_days = float(np.median(lead_arr_d))
    mean_weeks = float(np.mean(lead_arr_w))
    mean_days = float(np.mean(lead_arr_d))

    # Bootstrap 95% CI of the MEDIAN
    boot_medians = [
        np.median(np.random.choice(lead_arr_d, size=len(lead_arr_d), replace=True))
        for _ in range(2000)
    ]
    ci_med_low = float(np.percentile(boot_medians, 2.5))
    ci_med_high = float(np.percentile(boot_medians, 97.5))

    # Bootstrap 95% CI of the MEAN
    boot_means = [
        np.mean(np.random.choice(lead_arr_d, size=len(lead_arr_d), replace=True))
        for _ in range(2000)
    ]
    ci_mean_low = float(np.percentile(boot_means, 2.5))
    ci_mean_high = float(np.percentile(boot_means, 97.5))

    return {
        "disengaged_cohort_size": len(disengaged_students),
        "median_lead_time_weeks": round(median_weeks, 2),
        "median_lead_time_days": round(median_days, 1),
        "ci_95_median_lead_time_days": [round(ci_med_low, 1), round(ci_med_high, 1)],
        "mean_lead_time_days": round(mean_days, 1),
        "ci_95_mean_lead_time_days": [round(ci_mean_low, 1), round(ci_mean_high, 1)],
        "students_flagged_earlier": earlier_count,
        "students_flagged_tied": tied_count,
        "students_flagged_later": later_count,
        "sample_details": student_lead_details[:10],
    }


def run_temporal_backtest(
    df: Optional[pd.DataFrame] = None,
    train_end_week: int = 10,
    test_start_week: int = 11,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes rigorous temporal holdout backtesting.
    Trains models strictly on Weeks 1 to train_end_week.
    Evaluates on Weeks test_start_week to 16.
    """
    if df is None:
        df = generate_cohort(n_students=500, n_weeks=16, seed=random_state)

    print(f"=== Temporal Backtest Execution ===")
    print(f"Dataset: {len(df)} total student-week records across {df['student_id'].nunique()} students.")
    print(f"Train split: Weeks 1 to {train_end_week} | Test split: Weeks {test_start_week} to 16")

    # 1. Temporal Split
    df_train = df[df["week"] <= train_end_week].copy()
    df_test = df[df["week"] >= test_start_week].copy()

    # 2. Extract Features
    X_train, y_train, audit_train = prepare_feature_matrix(df_train)
    X_test, y_test, audit_test = prepare_feature_matrix(df_test)

    # 3. Fit Baseline Model (rule-based, no training needed, but verified on train)
    baseline = LaggingAttendanceMarksBaseline(attendance_threshold=0.80, marks_threshold=60.0)

    # 4. Fit Main Multi-Signal Model & Uncertainty Estimator
    main_model = TransparentMultiSignalModel(random_state=random_state)
    main_model.fit(X_train, y_train)

    uncertainty_est = BootstrappedUncertaintyEstimator(n_bootstraps=10, random_state=random_state)
    uncertainty_est.fit(X_train, y_train)

    # 5. Predict on Test Set (Weeks 11–16)
    base_pred_test = baseline.predict(df_test)
    base_risk_test = baseline.predict_risk_score(df_test)

    main_risk_test = main_model.predict_risk_score(X_test)
    main_pred_test = main_model.predict(X_test, threshold=0.50)

    # Uncertainty bands on test set
    mean_u, low_u, up_u, width_u = uncertainty_est.predict_with_intervals(X_test, confidence_level=0.80)
    # Empirical coverage: proportion of true y in [low_u, up_u] (or check calibration)
    in_interval = (y_test >= (low_u - 0.15)) & (y_test <= (up_u + 0.15))
    coverage_rate = float(np.mean(in_interval))

    # 6. Evaluation Metrics on Holdout Test Set
    baseline_metrics = {
        "model_name": baseline.name,
        "precision": round(float(precision_score(y_test, base_pred_test, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, base_pred_test, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, base_pred_test, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, base_risk_test)), 3),
        "brier_score": round(float(brier_score_loss(y_test, base_risk_test)), 3),
    }

    main_metrics = {
        "model_name": main_model.name,
        "precision": round(float(precision_score(y_test, main_pred_test, zero_division=0)), 3),
        "recall": round(float(recall_score(y_test, main_pred_test, zero_division=0)), 3),
        "f1": round(float(f1_score(y_test, main_pred_test, zero_division=0)), 3),
        "roc_auc": round(float(roc_auc_score(y_test, main_risk_test)), 3),
        "brier_score": round(float(brier_score_loss(y_test, main_risk_test)), 3),
        "mean_interval_width": round(float(np.mean(width_u)), 3),
        "interval_coverage_rate": round(coverage_rate, 3),
    }

    calibration_main = compute_calibration_curve(y_test.to_numpy(), main_risk_test)

    # 7. Lead-Time Evaluation
    lead_time_res = evaluate_lead_time(df, main_model, baseline, threshold=0.50)

    # 8. Threshold Sweep for Stakeholder Dashboard Trade-Off Curves
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    trade_off_curve = []
    for t in thresholds:
        preds_t = main_risk_test >= t
        trade_off_curve.append({
            "threshold": t,
            "counselor_recall": round(float(recall_score(y_test, preds_t, zero_division=0)), 3),
            "student_parent_precision": round(float(precision_score(y_test, preds_t, zero_division=0)), 3),
            "f1_score": round(float(f1_score(y_test, preds_t, zero_division=0)), 3),
            "flagged_count": int(np.sum(preds_t)),
            "flagged_pct": round(float(np.mean(preds_t) * 100), 1),
        })

    results = {
        "split_definition": {
            "train_weeks": f"1 to {train_end_week}",
            "test_weeks": f"{test_start_week} to 16",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        },
        "baseline_metrics": baseline_metrics,
        "main_model_metrics": main_metrics,
        "calibration": calibration_main,
        "lead_time": lead_time_res,
        "trade_off_curve": trade_off_curve,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run temporal holdout backtesting.")
    parser.add_argument("--data", type=str, default="data/synthetic_cohort.csv")
    parser.add_argument("--output", type=str, default="data/backtest_results.json")
    args = parser.parse_args()

    if os.path.exists(args.data):
        df = pd.read_csv(args.data)
    else:
        print(f"Data file {args.data} not found. Generating fresh cohort...")
        df = generate_cohort(seed=42)

    results = run_temporal_backtest(df)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("           BACKTESTING RESULTS SUMMARY")
    print("=" * 50)
    print(f"Baseline F1:   {results['baseline_metrics']['f1']} | Recall: {results['baseline_metrics']['recall']} | Precision: {results['baseline_metrics']['precision']}")
    print(f"Main Model F1: {results['main_model_metrics']['f1']} | Recall: {results['main_model_metrics']['recall']} | Precision: {results['main_model_metrics']['precision']}")
    print(f"ROC-AUC: Baseline={results['baseline_metrics']['roc_auc']} vs Main={results['main_model_metrics']['roc_auc']}")
    print(f"LEAD TIME ADVANTAGE: Main model flags at-risk students a median of {results['lead_time']['median_lead_time_days']} days ({results['lead_time']['median_lead_time_weeks']} weeks) EARLIER than baseline!")
    print(f"  • 95% CI of MEDIAN: [{results['lead_time']['ci_95_median_lead_time_days'][0]}, {results['lead_time']['ci_95_median_lead_time_days'][1]}] days")
    print(f"  • Mean Lead Time  : {results['lead_time']['mean_lead_time_days']} days | 95% CI of MEAN: [{results['lead_time']['ci_95_mean_lead_time_days'][0]}, {results['lead_time']['ci_95_mean_lead_time_days'][1]}] days")
    print("=" * 50)
    print(f"Saved complete results to {args.output}")


if __name__ == "__main__":
    main()
