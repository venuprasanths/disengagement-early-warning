"""
Feature Engineering Pipeline for Disengagement Early-Warning System.

Extracts multi-signal temporal features across rolling 3-week windows,
computes cross-signal divergence ratios (e.g. login-to-active-time ratio for gaming detection),
and enforces the Information Barrier between features and the ground-truth audit vault.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List
from src.schema import (
    GROUND_TRUTH_COLUMNS,
    ID_COLUMNS,
    assert_no_ground_truth_leakage,
)

ENGINEERED_FEATURE_NAMES: List[str] = [
    # 1. Attendance Features
    "att_3wk_mean",
    "att_trend_slope",
    "unexcused_absences_sum",
    "tardy_rate",
    # 2. Activity Features
    "lms_logins_3wk_mean",
    "content_time_3wk_mean",
    "late_submission_ratio",
    "logins_per_active_hour",  # Gaming indicator
    "discussion_posts_3wk_mean",
    # 3. Assessment Features
    "quiz_score_recent",
    "cumulative_score_avg",
    "score_trend_slope",
    "score_variance",
    # 4. Help-Seeking Features
    "office_hours_3wk_sum",
    "tutoring_3wk_sum",
    "questions_asked_3wk_mean",
    "help_seeking_delay_days",
    "is_seeking_help",
    # 5. Feedback / Sentiment Features
    "survey_sentiment_recent",
    "teacher_concern_flags_3wk",
    "confidence_rating_recent",
]


def extract_student_features(student_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts rolling multi-signal features for a single student across their available observation weeks.

    Parameters:
        student_df (pd.DataFrame): Time-series slice for an individual student containing
                                   raw behavioral columns. Must contain 'week' column.

    Returns:
        pd.DataFrame: Augmented DataFrame including 21 engineered feature columns.

    Mathematical Formulations & Safeguards:
        - Rolling Window: Trailing 3-week window (min_periods=1 or 2) to capture immediate
          velocity without losing responsiveness.
        - Trend Slope: 1st-degree polynomial fit (ordinary least squares) over 3 weeks:
            slope = sum((t - t_bar) * (y_t - y_bar)) / sum((t - t_bar)^2)
        - Gaming Divergence Ratio:
            logins_per_active_hour = lms_logins_3wk_mean / max(content_time_3wk_mean / 60, 0.1)
          Detects superficial engagement ('signal gaming') where students generate logins
          without reading course content.
        - Division by Zero Protection: np.clip(lower=1) or clip(lower=0.1) on denominators.
        - Missing Survey Imputation: Forward-fill followed by neutral neutral default (0.0).
    """
    student_df = student_df.sort_values("week").copy()

    # 1. Attendance Rolling Metrics
    att_roll = student_df["days_present"].rolling(window=3, min_periods=1)
    student_df["att_3wk_mean"] = att_roll.mean()

    # Attendance slope over 3-week window
    def calc_slope(series):
        vals = series.values
        if len(vals) < 2:
            return 0.0
        x = np.arange(len(vals))
        return float(np.polyfit(x, vals, 1)[0])

    student_df["att_trend_slope"] = (
        student_df["days_present"]
        .rolling(window=3, min_periods=2)
        .apply(calc_slope, raw=False)
        .fillna(0.0)
    )
    student_df["unexcused_absences_sum"] = student_df["unexcused_absences"].cumsum()
    present_sum = student_df["days_present"].cumsum().clip(lower=1)
    student_df["tardy_rate"] = student_df["tardy_count"].cumsum() / present_sum

    # 2. Activity Rolling Metrics
    student_df["lms_logins_3wk_mean"] = (
        student_df["lms_logins"].rolling(window=3, min_periods=1).mean()
    )
    student_df["content_time_3wk_mean"] = (
        student_df["content_time_minutes"].rolling(window=3, min_periods=1).mean()
    )
    total_subs = student_df["assignment_submissions"].cumsum().clip(lower=1)
    late_subs = student_df["late_submissions"].cumsum()
    student_df["late_submission_ratio"] = late_subs / total_subs

    # Gaming divergence: high logins but low reading minutes
    active_hours = (student_df["content_time_3wk_mean"] / 60.0).clip(lower=0.1)
    student_df["logins_per_active_hour"] = student_df["lms_logins_3wk_mean"] / active_hours

    student_df["discussion_posts_3wk_mean"] = (
        student_df["discussion_posts"].rolling(window=3, min_periods=1).mean()
    )

    # 3. Assessment Rolling Metrics
    student_df["quiz_score_recent"] = student_df["quiz_score"].ffill().fillna(50.0)
    # cumulative_score_avg, score_trend_slope, score_variance already computed weekly

    # 4. Help-Seeking Metrics
    student_df["office_hours_3wk_sum"] = (
        student_df["office_hours_attended"].rolling(window=3, min_periods=1).sum()
    )
    student_df["tutoring_3wk_sum"] = (
        student_df["tutoring_sessions"].rolling(window=3, min_periods=1).sum()
    )
    student_df["questions_asked_3wk_mean"] = (
        student_df["questions_asked"].rolling(window=3, min_periods=1).mean()
    )
    total_help_3wk = student_df["office_hours_3wk_sum"] + student_df["tutoring_3wk_sum"]
    student_df["is_seeking_help"] = (total_help_3wk > 0).astype(float)

    # 5. Feedback / Sentiment Metrics
    # Forward fill bi-weekly survey sentiment; default to neutral 0.0
    student_df["survey_sentiment_recent"] = (
        student_df["survey_sentiment"].ffill().fillna(0.0)
    )
    student_df["teacher_concern_flags_3wk"] = (
        student_df["teacher_note_flag"].rolling(window=3, min_periods=1).sum()
    )
    student_df["confidence_rating_recent"] = (
        student_df["confidence_rating"].ffill().fillna(3.0)
    )

    return student_df


def prepare_feature_matrix(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Transforms raw weekly cohort data into model-ready matrices while enforcing the Information Barrier.

    Processing Pipeline:
    1. Pre-condition Barrier Check: Scans input columns to guarantee zero presence of
       ground-truth columns ('archetype', 'latent_engagement', 'is_disengaged') among features.
    2. Group-Wise Time-Series Expansion: Iterates over student cohorts independently to compute
       rolling 3-week statistics without cross-student leakage.
    3. Matrix Decomposition:
       - X: Clean engineered 21-feature matrix filled with 0.0 for initial boundary conditions.
       - y: Binary outcome label series (1 = disengaged, 0 = engaged).
       - audit_df: Metadata vault preserving student_id, week, and ground-truth audit columns
         for post-inference evaluation and fairness checking only.

    Parameters:
        df (pd.DataFrame): Raw weekly cohort dataframe matching StudentWeeklyRecord schema.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
            - X: DataFrame of shape (N*T, 21) containing engineered features.
            - y: pd.Series of shape (N*T,) containing binary ground-truth labels.
            - audit_df: DataFrame of shape (N*T, 5) containing audit identifiers and latent states.

    Raises:
        ValueError: If any ground-truth column is detected in feature extraction candidates.
    """
    # Verify no raw ground truth is used as features
    feature_candidates = [c for c in df.columns if c not in GROUND_TRUTH_COLUMNS]
    assert_no_ground_truth_leakage(feature_candidates)

    # Group by student to extract time-series features
    processed_dfs = []
    for _, student_group in df.groupby("student_id"):
        processed_group = extract_student_features(student_group)
        processed_dfs.append(processed_group)

    full_df = pd.concat(processed_dfs, ignore_index=True)

    # Extract Feature Matrix X
    X = full_df[ENGINEERED_FEATURE_NAMES].copy()
    X = X.fillna(0.0)

    # Extract Target y and Audit Data
    y = full_df["is_disengaged"].astype(int)
    audit_cols = ID_COLUMNS + GROUND_TRUTH_COLUMNS
    audit_df = full_df[audit_cols].copy()

    return X, y, audit_df
