"""
Current-Practice Lagging Baseline Model.

Represents what schools conventionally do today:
A rigid, punitive rule-based system relying ONLY on attendance and marks.
Flags students only after their attendance drops below 80% or their marks fail below 60%.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from src.schema import BASELINE_FEATURE_COLUMNS, assert_no_ground_truth_leakage


class LaggingAttendanceMarksBaseline:
    """
    Conventional punitive baseline model representing standard school early-warning heuristics.

    Architectural Blind Spots:
        1. Attendance Blindness: Students in the 'Quietly Struggling' archetype attend
           class faithfully (days_present = 5/5) while suffering emotional withdrawal.
           The baseline achieves only 8.3% recall on these students.
        2. Signal Gaming Blindness: Students logging in repeatedly without studying are
           unseen by this model.
        3. Lagging Response: Marks are only updated after major cumulative quizzes or exams,
           causing detection to lag by 6 to 7 weeks compared to continuous multi-signal fusion.
    """

    def __init__(
        self,
        attendance_threshold: float = 0.80,
        marks_threshold: float = 60.0,
    ):
        """
        Initializes the baseline with standard educational district thresholds.

        Parameters:
            attendance_threshold (float): Minimum acceptable attendance percentage (default 0.80 = 80%).
            marks_threshold (float): Passing academic mark percentage (default 60.0 = 60%).
        """
        self.attendance_threshold = attendance_threshold
        self.marks_threshold = marks_threshold
        self.feature_columns = BASELINE_FEATURE_COLUMNS
        self.name = "Lagging Attendance+Marks Baseline (Current Practice)"

    def predict_risk_score(self, df: pd.DataFrame) -> np.ndarray:
        """
        Computes a continuous proxy risk score based purely on attendance and marks deficits.

        Mathematical Formulation:
            att_risk = clip((attendance_threshold - attendance_rate) / attendance_threshold, 0, 1)
            marks_risk = clip((marks_threshold - marks) / marks_threshold, 0, 1)
            combined_risk = max(att_risk, marks_risk)

        Parameters:
            df (pd.DataFrame): Input DataFrame containing 'days_present' and 'cumulative_score_avg'.

        Returns:
            np.ndarray: Continuous risk score in range [0.0, 1.0].
        """
        assert_no_ground_truth_leakage(self.feature_columns)

        # Check only permitted baseline columns
        attendance_rate = df["days_present"] / 5.0
        marks = df["cumulative_score_avg"]

        # Attendance risk: 0 when >= 80%, linearly scaling to 1 as attendance drops to 0
        att_risk = np.clip((self.attendance_threshold - attendance_rate) / self.attendance_threshold, 0.0, 1.0)

        # Marks risk: 0 when >= 60%, linearly scaling to 1 as marks drop to 0
        marks_risk = np.clip((self.marks_threshold - marks) / self.marks_threshold, 0.0, 1.0)

        # Naive max risk across the two punitive sensors
        combined_risk = np.maximum(att_risk, marks_risk)
        return combined_risk

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Binary flag matching standard district intervention policy:
        Flags True if attendance < 80% OR cumulative marks < 60%.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            threshold (float): Unused compatibility argument (baseline uses hard policy thresholds).

        Returns:
            np.ndarray: Boolean array of flags.
        """
        assert_no_ground_truth_leakage(self.feature_columns)
        attendance_flag = (df["days_present"] / 5.0) < self.attendance_threshold
        marks_flag = df["cumulative_score_avg"] < self.marks_threshold
        return (attendance_flag | marks_flag).to_numpy()

    def get_signal_contributions(self, row: pd.Series) -> Dict[str, Any]:
        """Explains why the baseline flagged or did not flag a student."""
        att_rate = row["days_present"] / 5.0
        marks = row["cumulative_score_avg"]
        return {
            "attendance_rate": round(float(att_rate), 2),
            "attendance_flagged": bool(att_rate < self.attendance_threshold),
            "cumulative_marks": round(float(marks), 1),
            "marks_flagged": bool(marks < self.marks_threshold),
            "rule": f"Flagged if attendance < {int(self.attendance_threshold*100)}% OR marks < {int(self.marks_threshold)}%",
            "status": "FLAGGED" if (att_rate < self.attendance_threshold or marks < self.marks_threshold) else "NORMAL",
        }
