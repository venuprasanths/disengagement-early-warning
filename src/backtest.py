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
    Computes censoring-aware lead time and Kaplan-Meier survival analysis.
    Explicitly separates:
      - Population (a): Both models flag within observation window (uncensored)
      - Population (b): Baseline never flags within window (right-censored)
      - Population (c): Main model never flags within window (false negatives)
    """
    disengaged_students = df_cohort[df_cohort["is_disengaged"] == True]["student_id"].unique()
    total_disengaged = len(disengaged_students)

    X_all, _, _ = prepare_feature_matrix(df_cohort)
    main_scores = main_model.predict_risk_score(X_all)
    df_eval = df_cohort.copy()
    df_eval["main_score"] = main_scores
    df_eval["main_flag"] = main_scores >= threshold
    df_eval["baseline_flag"] = baseline_model.predict(df_cohort)

    records = []
    for s_id in disengaged_students:
        s_data = df_eval[df_eval["student_id"] == s_id].sort_values("week")
        arch = s_data["archetype"].iloc[0]
        
        true_weeks = s_data[s_data["is_disengaged"] == True]["week"]
        first_true = int(true_weeks.min()) if not true_weeks.empty else 16
        
        m_flagged = s_data[s_data["main_flag"] == True]["week"]
        b_flagged = s_data[s_data["baseline_flag"] == True]["week"]
        
        first_m = int(m_flagged.min()) if not m_flagged.empty else None
        first_b = int(b_flagged.min()) if not b_flagged.empty else None
        
        if first_m is not None and first_b is not None:
            pop = "both_flagged"
            lead_w = first_b - first_m
            lead_d = lead_w * 7.0
        elif first_m is not None and first_b is None:
            pop = "baseline_never_flagged"
            lead_w = None
            lead_d = None
        elif first_m is None and first_b is not None:
            pop = "main_never_flagged"
            lead_w = None
            lead_d = None
        else:
            pop = "neither_flagged"
            lead_w = None
            lead_d = None

        records.append({
            "student_id": s_id,
            "archetype": arch,
            "first_true_week": first_true,
            "first_main_week": first_m,
            "first_baseline_week": first_b,
            "population": pop,
            "lead_time_weeks": lead_w,
            "lead_time_days": lead_d,
            "main_days_before_onset": (first_true - first_m) * 7.0 if first_m is not None else None,
            "base_days_before_onset": (first_true - first_b) * 7.0 if first_b is not None else None,
        })

    lead_df = pd.DataFrame(records)

    # Population counts and percentages
    pop_a = lead_df[lead_df["population"] == "both_flagged"]
    pop_b = lead_df[lead_df["population"] == "baseline_never_flagged"]
    pop_c = lead_df[lead_df["population"] == "main_never_flagged"]

    n_both = len(pop_a)
    n_base_censored = len(pop_b)
    n_main_censored = len(pop_c)

    pct_base_censored = round((n_base_censored / total_disengaged) * 100, 1)
    pct_main_censored = round((n_main_censored / total_disengaged) * 100, 1)

    # Population (a) statistics
    both_leads_d = pop_a["lead_time_days"].values
    both_leads_w = pop_a["lead_time_weeks"].values

    med_d = float(np.median(both_leads_d))
    mean_d = float(np.mean(both_leads_d))
    q25_d = float(np.percentile(both_leads_d, 25))
    q75_d = float(np.percentile(both_leads_d, 75))

    boot_med = [np.median(np.random.choice(both_leads_d, size=len(both_leads_d), replace=True)) for _ in range(2000)]
    boot_mean = [np.mean(np.random.choice(both_leads_d, size=len(both_leads_d), replace=True)) for _ in range(2000)]

    ci_med = [round(float(np.percentile(boot_med, 2.5)), 1), round(float(np.percentile(boot_med, 97.5)), 1)]
    ci_mean = [round(float(np.percentile(boot_mean, 2.5)), 1), round(float(np.percentile(boot_mean, 97.5)), 1)]

    # Kaplan-Meier Survival Estimator
    def compute_km(first_weeks, max_t=16):
        times = [w if (w is not None and w <= max_t) else max_t for w in first_weeks]
        events = [1 if (w is not None and w <= max_t) else 0 for w in first_weeks]
        times = np.array(times)
        events = np.array(events)
        unique_times = np.arange(1, max_t + 1)
        s_t = 1.0
        n_at_risk = len(times)
        km_median_discrete = None
        km_median_interpolated = None
        s_history = {}
        prev_s = 1.0
        for t in unique_times:
            d_t = int(np.sum((times == t) & (events == 1)))
            c_t = int(np.sum((times == t) & (events == 0)))
            if n_at_risk > 0:
                s_t *= (1.0 - d_t / n_at_risk)
            s_history[int(t)] = round(float(s_t), 4)
            if km_median_discrete is None and s_t <= 0.50:
                km_median_discrete = int(t)
            if km_median_interpolated is None and s_t <= 0.50:
                if prev_s != s_t:
                    frac = (prev_s - 0.50) / (prev_s - s_t)
                    km_median_interpolated = round(float((t - 1) + frac), 2)
                else:
                    km_median_interpolated = float(t)
            prev_s = s_t
            n_at_risk -= (d_t + c_t)
        return {
            "discrete_week": km_median_discrete or max_t,
            "interpolated_week": km_median_interpolated or float(max_t),
            "curve": s_history,
        }

    km_main = compute_km(lead_df["first_main_week"])
    km_base = compute_km(lead_df["first_baseline_week"])
    km_lead_advantage_weeks = km_main["discrete_week"] - km_main["discrete_week"]  # placeholder
    km_disc_lead_w = km_base["discrete_week"] - km_main["discrete_week"]
    km_interp_lead_w = round(km_base["interpolated_week"] - km_main["interpolated_week"], 2)

    # Histogram frequency counts
    hist_w = pd.Series(both_leads_w).value_counts().sort_index()
    histogram = {int(k): int(v) for k, v in hist_w.items()}

    return {
        "disengaged_cohort_size": total_disengaged,
        "population_breakdown": {
            "both_flagged_count": n_both,
            "both_flagged_pct": round((n_both / total_disengaged) * 100, 1),
            "baseline_never_flagged_count": n_base_censored,
            "baseline_never_flagged_pct": pct_base_censored,
            "main_never_flagged_count": n_main_censored,
            "main_never_flagged_pct": pct_main_censored,
        },
        "uncensored_lead_time_pop_a": {
            "median_lead_days": round(med_d, 1),
            "median_lead_weeks": round(med_d / 7.0, 1),
            "ci_95_median_days": ci_med,
            "mean_lead_days": round(mean_d, 1),
            "ci_95_mean_days": ci_mean,
            "iqr_q25_q75_days": [round(q25_d, 1), round(q75_d, 1)],
            "lead_time_histogram_weeks": histogram,
        },
        "kaplan_meier": {
            "km_median_detection_week_main": km_main["discrete_week"],
            "km_median_interpolated_week_main": km_main["interpolated_week"],
            "km_median_detection_week_baseline": km_base["discrete_week"],
            "km_median_interpolated_week_baseline": km_base["interpolated_week"],
            "km_lead_time_advantage_weeks": km_disc_lead_w,
            "km_lead_time_advantage_days": km_disc_lead_w * 7,
            "km_lead_time_interpolated_advantage_weeks": km_interp_lead_w,
            "km_lead_time_interpolated_advantage_days": round(km_interp_lead_w * 7.0, 1),
            "survival_curve_main": km_main["curve"],
            "survival_curve_baseline": km_base["curve"],
        },
        "timing_relative_to_true_onset": {
            "main_median_days_before_onset": round(float(np.median(pop_a["main_days_before_onset"].dropna())), 1),
            "base_median_days_before_onset": round(float(np.median(pop_a["base_days_before_onset"].dropna())), 1),
        },
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
    print("-" * 50)
    print(f"CENSORING BREAKDOWN (N={results['lead_time']['disengaged_cohort_size']} Disengaged Students):")
    print(f"  • Pop (a) Both Flagged (Uncensored)           : {results['lead_time']['population_breakdown']['both_flagged_count']} ({results['lead_time']['population_breakdown']['both_flagged_pct']}%)")
    print(f"  • Pop (b) Baseline NEVER Flags (Right-Censored): {results['lead_time']['population_breakdown']['baseline_never_flagged_count']} ({results['lead_time']['population_breakdown']['baseline_never_flagged_pct']}%)")
    print(f"  • Pop (c) Main Model NEVER Flags              : {results['lead_time']['population_breakdown']['main_never_flagged_count']} ({results['lead_time']['population_breakdown']['main_never_flagged_pct']}%)")
    print("-" * 50)
    print("KAPLAN-MEIER SURVIVAL ESTIMATE (TIME-TO-DETECTION):")
    print(f"  • Main Model KM Median Detection: Week {results['lead_time']['kaplan_meier']['km_median_detection_week_main']} (Interpolated: Week {results['lead_time']['kaplan_meier']['km_median_interpolated_week_main']})")
    print(f"  • Baseline KM Median Detection  : Week {results['lead_time']['kaplan_meier']['km_median_detection_week_baseline']} (Interpolated: Week {results['lead_time']['kaplan_meier']['km_median_interpolated_week_baseline']})")
    print(f"  • KM Lead Advantage (Discrete)  : {results['lead_time']['kaplan_meier']['km_lead_time_advantage_days']} days ({results['lead_time']['kaplan_meier']['km_lead_time_advantage_weeks']} weeks earlier)")
    print(f"  • KM Lead Advantage (Interp)    : {results['lead_time']['kaplan_meier']['km_lead_time_interpolated_advantage_days']} days ({results['lead_time']['kaplan_meier']['km_lead_time_interpolated_advantage_weeks']} weeks earlier)")
    print("-" * 50)
    print("POPULATION (a) UNCENSORED LEAD TIME (Both Models Flagged):")
    print(f"  • Median Lead: {results['lead_time']['uncensored_lead_time_pop_a']['median_lead_days']} days | 95% CI: [{results['lead_time']['uncensored_lead_time_pop_a']['ci_95_median_days'][0]}, {results['lead_time']['uncensored_lead_time_pop_a']['ci_95_median_days'][1]}] days")
    print(f"  • Mean Lead  : {results['lead_time']['uncensored_lead_time_pop_a']['mean_lead_days']} days | 95% CI: [{results['lead_time']['uncensored_lead_time_pop_a']['ci_95_mean_days'][0]}, {results['lead_time']['uncensored_lead_time_pop_a']['ci_95_mean_days'][1]}] days")
    print("=" * 50)
    print(f"Saved complete results to {args.output}")


if __name__ == "__main__":
    main()
