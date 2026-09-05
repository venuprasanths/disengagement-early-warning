"""
Main Multi-Signal Model for Transparent Disengagement Early-Warning.

Fuses 5 signal families:
- Attendance
- Course Activity / LMS
- Assessment Trends
- Help-Seeking Behaviors
- Feedback & Sentiment

Provides:
1. Calibrated probability of disengagement risk p_hat in [0.0, 1.0]
2. Per-student explainability via SHAP / feature contributions
3. Signal family contribution aggregation for actionable counselor intervention
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.schema import assert_no_ground_truth_leakage, GROUND_TRUTH_COLUMNS
from src.features import ENGINEERED_FEATURE_NAMES


SIGNAL_FAMILY_MAPPING = {
    # 1. Attendance
    "att_3wk_mean": "Attendance",
    "att_trend_slope": "Attendance",
    "unexcused_absences_sum": "Attendance",
    "tardy_rate": "Attendance",
    # 2. Activity
    "lms_logins_3wk_mean": "Activity",
    "content_time_3wk_mean": "Activity",
    "late_submission_ratio": "Activity",
    "logins_per_active_hour": "Activity",
    "discussion_posts_3wk_mean": "Activity",
    # 3. Assessment
    "quiz_score_recent": "Assessment",
    "cumulative_score_avg": "Assessment",
    "score_trend_slope": "Assessment",
    "score_variance": "Assessment",
    # 4. Help-Seeking
    "office_hours_3wk_sum": "Help-Seeking",
    "tutoring_3wk_sum": "Help-Seeking",
    "questions_asked_3wk_mean": "Help-Seeking",
    "help_seeking_delay_days": "Help-Seeking",
    "is_seeking_help": "Help-Seeking",
    # 5. Feedback / Sentiment
    "survey_sentiment_recent": "Feedback & Sentiment",
    "teacher_concern_flags_3wk": "Feedback & Sentiment",
    "confidence_rating_recent": "Feedback & Sentiment",
}


class TransparentMultiSignalModel:
    """
    Transparent, calibrated multi-signal classifier.
    Combines HistGradientBoosting with Isotonic/Sigmoid calibration
    for well-calibrated probabilities, coupled with Tree/Kernel SHAP explanation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.name = "Transparent Multi-Signal Model (5-Family Fusion)"
        self.feature_names = ENGINEERED_FEATURE_NAMES

        # Base estimator: interpretable, robust gradient booster
        self.base_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=15,       # Keeps trees shallow for interpretability
            min_samples_leaf=20,
            learning_rate=0.08,
            random_state=self.random_state,
        )
        # Probability calibrator
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="sigmoid",
            cv=3,
        )
        self.is_fitted = False
        self._feature_importances: Optional[Dict[str, float]] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransparentMultiSignalModel":
        """Fits the calibrated multi-signal model."""
        assert_no_ground_truth_leakage(list(X.columns))
        X_clean = X[self.feature_names].fillna(0.0)

        self.calibrated_model.fit(X_clean, y)
        self.is_fitted = True

        # Fit base model separately for direct tree feature importances
        self.base_model.fit(X_clean, y)
        return self

    def predict_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated risk probabilities p_hat in [0.0, 1.0]."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_risk_score.")
        assert_no_ground_truth_leakage(list(X.columns))
        X_clean = X[self.feature_names].fillna(0.0)
        # Probability of class 1 (disengaged)
        return self.calibrated_model.predict_proba(X_clean)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Returns binary flag given decision threshold."""
        scores = self.predict_risk_score(X)
        return scores >= threshold

    def explain_instance(
        self,
        instance_features: pd.Series,
        background_X: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Generates local explanation for an individual student row.
        Breaks down contribution across the 5 signal families and individual features.
        """
        assert_no_ground_truth_leakage(self.feature_names)
        x_row = instance_features[self.feature_names].to_frame().T.fillna(0.0)
        risk_score = float(self.predict_risk_score(x_row)[0])

        # If background_X is provided, compute approximate marginal contributions
        # relative to cohort baseline
        contributions = {}
        if background_X is not None:
            bg_clean = background_X[self.feature_names].fillna(0.0)
            baseline_prob = float(np.mean(self.predict_risk_score(bg_clean)))
            delta_total = risk_score - baseline_prob

            # Approximate permutation attribution
            feature_deltas = {}
            for col in self.feature_names:
                perturbed_row = x_row.copy()
                perturbed_row[col] = bg_clean[col].median()
                score_without = float(self.predict_risk_score(perturbed_row)[0])
                feature_deltas[col] = risk_score - score_without

            sum_deltas = sum(abs(v) for v in feature_deltas.values()) or 1.0
            for col, d in feature_deltas.items():
                contributions[col] = (d / sum_deltas) * delta_total
        else:
            baseline_prob = 0.25
            for col in self.feature_names:
                contributions[col] = 0.0

        # Aggregate contributions by the 5 signal families
        family_contributions: Dict[str, float] = {
            "Attendance": 0.0,
            "Activity": 0.0,
            "Assessment": 0.0,
            "Help-Seeking": 0.0,
            "Feedback & Sentiment": 0.0,
        }
        for col, val in contributions.items():
            fam = SIGNAL_FAMILY_MAPPING.get(col, "Other")
            if fam in family_contributions:
                family_contributions[fam] += val

        sorted_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)
        top_risk_drivers = [
            {"feature": f, "impact": round(v, 3), "family": SIGNAL_FAMILY_MAPPING.get(f, "Other")}
            for f, v in sorted_features if v > 0
        ][:5]
        top_protective_factors = [
            {"feature": f, "impact": round(v, 3), "family": SIGNAL_FAMILY_MAPPING.get(f, "Other")}
            for f, v in sorted_features if v < 0
        ][:5]

        return {
            "risk_score": round(risk_score, 3),
            "baseline_cohort_risk": round(baseline_prob, 3),
            "family_contributions": {k: round(v, 3) for k, v in family_contributions.items()},
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "primary_driver_family": max(family_contributions.items(), key=lambda item: item[1])[0],
        }
