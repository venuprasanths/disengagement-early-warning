"""
Sensitivity & Stress Testing Across Cohort Compositions.

Stress tests the Transparent Multi-Signal Model against the Lagging Baseline across
at least 3 realistic skewed cohort distributions:
1. Reference Cohort: Standard balanced composition (N=500, 20% quietly struggling, 1.4% acute shock).
2. High Acute-Shock Cohort: Post-crisis / seasonal health surge (20% acute shock, 16% quiet struggle).
3. High Chronic Quiet-Struggle Cohort: Rigorous magnet school setting (44% quiet struggle, 10% checked out).
4. Mixed Worst-Case Cohort: Under-resourced, highly disrupted school (30% quiet struggle, 24% checked out, 12% acute shock).

Evaluates:
- F1 Score shifts
- Recall (Counselor Sensitivity) shifts
- Precision (Student/Parent Protection) shifts
- False Alarm Rate (False Positive Rate: FP / (FP + TN)) shifts
- Archetype-specific failure mode shifts (e.g. Acute Shock False Alarm Rate, Quietly Struggling Recall)
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

from data.synthetic_data_generator import generate_cohort
from src.features import prepare_feature_matrix
from src.baseline_model import LaggingAttendanceMarksBaseline
from src.main_model import TransparentMultiSignalModel


SCENARIO_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "reference_balanced": {
        "name": "Standard Reference (Balanced)",
        "description": "Standard population distribution (50% engaged, 20% quiet struggle, 15% checked out, 10% improving, 5% edge cases).",
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
    "high_acute_shock": {
        "name": "High Acute-Shock (Crisis Surge)",
        "description": "Simulates a community crisis or health surge where 20% of the cohort experiences sudden temporary distress.",
        "archetype_counts": {
            "consistently_engaged": 200,
            "quietly_struggling": 80,
            "checked_out": 60,
            "genuinely_improving": 40,
            "transfer_student": 10,
            "signal_gamer": 10,
            "acute_shock": 100,  # 20.0% acute shock (14x baseline rate)
        },
    },
    "high_chronic_quiet_struggle": {
        "name": "High Chronic Quiet-Struggle (STEM/Magnet)",
        "description": "Simulates a high-pressure magnet school where 44% of students quietly struggle while maintaining near-perfect attendance.",
        "archetype_counts": {
            "consistently_engaged": 150,
            "quietly_struggling": 220,  # 44.0% quiet struggle (2.2x baseline rate)
            "checked_out": 50,
            "genuinely_improving": 40,
            "transfer_student": 15,
            "signal_gamer": 15,
            "acute_shock": 10,
        },
    },
    "mixed_worst_case": {
        "name": "Mixed Worst-Case (Stressed/Under-Resourced)",
        "description": "Compound disruption: high quiet struggle (30%), checked out (24%), and acute shocks (12%) in an under-resourced environment.",
        "archetype_counts": {
            "consistently_engaged": 100,  # Only 20% engaged
            "quietly_struggling": 150,
            "checked_out": 120,
            "genuinely_improving": 30,
            "transfer_student": 20,
            "signal_gamer": 20,
            "acute_shock": 60,
        },
    },
}


def evaluate_cohort_scenario(
    scenario_key: str,
    scenario_cfg: Dict[str, Any],
    seed: int = 42,
    train_end_week: int = 10,
    test_start_week: int = 11,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Generates a cohort under specified archetype distribution, trains both models
    on Weeks 1–10, and evaluates holdout performance on Weeks 11–16.
    """
    archetype_counts = scenario_cfg["archetype_counts"]
    df = generate_cohort(seed=seed, archetype_counts=archetype_counts)

    # Temporal split
    df_train = df[df["week"] <= train_end_week].copy()
    df_test = df[df["week"] >= test_start_week].copy()

    X_train, y_train, _ = prepare_feature_matrix(df_train)
    X_test, y_test, audit_test = prepare_feature_matrix(df_test)

    # Train Baseline & Main Model
    baseline = LaggingAttendanceMarksBaseline()
    main_model = TransparentMultiSignalModel(random_state=seed)
    main_model.fit(X_train, y_train)

    # Predictions on holdout test set
    base_preds = baseline.predict(df_test)
    base_risk = baseline.predict_risk_score(df_test)

    main_risk = main_model.predict_risk_score(X_test)
    main_preds = main_risk >= threshold

    y_test_arr = y_test.to_numpy()

    # Overall Metrics: Main Model
    main_rec = float(recall_score(y_test_arr, main_preds, zero_division=0))
    main_prec = float(precision_score(y_test_arr, main_preds, zero_division=0))
    main_f1 = float(f1_score(y_test_arr, main_preds, zero_division=0))
    main_auc = float(roc_auc_score(y_test_arr, main_risk))
    main_brier = float(brier_score_loss(y_test_arr, main_risk))

    # Overall Metrics: Baseline
    base_rec = float(recall_score(y_test_arr, base_preds, zero_division=0))
    base_prec = float(precision_score(y_test_arr, base_preds, zero_division=0))
    base_f1 = float(f1_score(y_test_arr, base_preds, zero_division=0))
    base_auc = float(roc_auc_score(y_test_arr, base_risk))
    base_brier = float(brier_score_loss(y_test_arr, base_risk))

    # False-Alarm Rate on Negatives: FP / (FP + TN)
    neg_mask = y_test_arr == 0
    main_fpr = float(np.mean(main_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
    base_fpr = float(np.mean(base_preds[neg_mask])) if np.sum(neg_mask) > 0 else 0.0

    # Archetype breakdown
    eval_df = audit_test.copy()
    eval_df["y_true"] = y_test_arr
    eval_df["main_pred"] = main_preds
    eval_df["base_pred"] = base_preds

    arch_breakdown = {}
    for arch, grp in eval_df.groupby("archetype"):
        y_arch = grp["y_true"].to_numpy()
        m_pred = grp["main_pred"].to_numpy()
        b_pred = grp["base_pred"].to_numpy()

        n_pos = int(np.sum(y_arch == 1))
        n_neg = int(np.sum(y_arch == 0))

        # Recall (if positives exist)
        m_rec = float(np.sum(m_pred[y_arch == 1]) / n_pos) if n_pos > 0 else None
        b_rec = float(np.sum(b_pred[y_arch == 1]) / n_pos) if n_pos > 0 else None

        # False alarm rate (if negatives exist)
        m_fa = float(np.sum(m_pred[y_arch == 0]) / n_neg) if n_neg > 0 else None
        b_fa = float(np.sum(b_pred[y_arch == 0]) / n_neg) if n_neg > 0 else None

        arch_breakdown[arch] = {
            "count_observations": len(grp),
            "disengaged_count": n_pos,
            "main_recall": round(m_rec, 3) if m_rec is not None else "N/A",
            "baseline_recall": round(b_rec, 3) if b_rec is not None else "N/A",
            "main_false_alarm_rate": round(m_fa, 3) if m_fa is not None else "N/A",
            "baseline_false_alarm_rate": round(b_fa, 3) if b_fa is not None else "N/A",
        }

    return {
        "scenario_key": scenario_key,
        "name": scenario_cfg["name"],
        "description": scenario_cfg["description"],
        "cohort_size": len(df["student_id"].unique()),
        "test_observations": len(df_test),
        "disengagement_base_rate": round(float(np.mean(y_test_arr)), 3),
        "main_model": {
            "f1": round(main_f1, 4),
            "recall": round(main_rec, 4),
            "precision": round(main_prec, 4),
            "false_alarm_rate": round(main_fpr, 4),
            "roc_auc": round(main_auc, 4),
            "brier_score": round(main_brier, 4),
        },
        "baseline_model": {
            "f1": round(base_f1, 4),
            "recall": round(base_rec, 4),
            "precision": round(base_prec, 4),
            "false_alarm_rate": round(base_fpr, 4),
            "roc_auc": round(base_auc, 4),
            "brier_score": round(base_brier, 4),
        },
        "relative_f1_gain": round((main_f1 - base_f1) / max(1e-5, base_f1) * 100, 1),
        "archetype_breakdown": arch_breakdown,
    }


def run_all_sensitivity_tests(seed: int = 42) -> Dict[str, Any]:
    """Runs all 4 cohort stress test scenarios and aggregates results."""
    results = {}
    for key, cfg in SCENARIO_CONFIGURATIONS.items():
        print(f"--> Executing stress test: {cfg['name']}...")
        results[key] = evaluate_cohort_scenario(key, cfg, seed=seed)
    return results


def format_markdown_summary(results: Dict[str, Any]) -> str:
    """Formats sensitivity test results into an auditable Markdown report table."""
    lines = [
        "| Cohort Scenario | Model | F1 Score | Recall (Sens.) | Precision | False-Alarm Rate (FPR) | Brier Calibration | ROC-AUC |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for key, data in results.items():
        scen_name = data["name"]
        m = data["main_model"]
        b = data["baseline_model"]

        lines.append(
            f"| **{scen_name}** | **Main 5-Signal** | **{m['f1']:.3f}** | **{m['recall']*100:.1f}%** | **{m['precision']*100:.1f}%** | **{m['false_alarm_rate']*100:.1f}%** | **{m['brier_score']:.3f}** | **{m['roc_auc']:.3f}** |"
        )
        lines.append(
            f"| *(Base Rate: {data['disengagement_base_rate']*100:.1f}%)* | Lagging Baseline | {b['f1']:.3f} | {b['recall']*100:.1f}% | {b['precision']*100:.1f}% | {b['false_alarm_rate']*100:.1f}% | {b['brier_score']:.3f} | {b['roc_auc']:.3f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity/stress testing across cohort compositions.")
    parser.add_argument("--output", type=str, default="data/sensitivity_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("      SENSITIVITY & STRESS TESTING ACROSS COHORT COMPOSITIONS")
    print("=" * 70 + "\n")

    results = run_all_sensitivity_tests(seed=args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("                  STRESS TESTING RESULTS SUMMARY")
    print("=" * 70 + "\n")
    print(format_markdown_summary(results))
    print("\n" + "=" * 70)
    print(f"Saved complete sensitivity analysis results to {args.output}")


if __name__ == "__main__":
    main()
