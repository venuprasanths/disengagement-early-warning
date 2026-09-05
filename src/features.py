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
    Extracts rolling features for a single student across their available weeks.
    Assumes student_df is sorted by 'week'.
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
    Transforms raw weekly data into:
    1. X: Feature matrix containing only engineered non-leaked behavioral features
    2. y: Binary ground-truth label series (is_disengaged)
    3. audit_df: Metadata and latent state for backtest evaluation and error analysis
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
