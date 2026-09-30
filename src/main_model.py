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
2. Per-student explainability via SHAP / feature contributions with plain-language labels
3. Signal family contribution aggregation for actionable counselor intervention
4. Compassionate Restorative Triage (Stakeholder Persona 4 safeguard)
5. FERPA-compliant audit record generation (Stakeholder Persona 5 safeguard)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
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

# Plain-language labels for non-technical educators & counselors (Review 2 Requirement 1)
FEATURE_PLAIN_LANGUAGE_MAPPING = {
    # 1. Attendance
    "att_3wk_mean": "In-Seat Attendance Rate (3-Wk Avg)",
    "att_trend_slope": "Attendance Trajectory / Slope",
    "unexcused_absences_sum": "Unexcused Absences (Total)",
    "tardy_rate": "Class Tardiness Frequency",
    # 2. Activity
    "lms_logins_3wk_mean": "LMS Portal Login Frequency",
    "content_time_3wk_mean": "Active Digital Reading & Study Time",
    "late_submission_ratio": "Late Assignment Submission Rate",
    "logins_per_active_hour": "Superficial Activity Ratio (Clicks vs Depth)",
    "discussion_posts_3wk_mean": "Discussion Forum Participation",
    # 3. Assessment
    "quiz_score_recent": "Most Recent Assessment Score",
    "cumulative_score_avg": "Cumulative Gradebook Average",
    "score_trend_slope": "Grade Momentum / Velocity",
    "score_variance": "Assessment Score Volatility",
    # 4. Help-Seeking
    "office_hours_3wk_sum": "Teacher Office Hours Attended",
    "tutoring_3wk_sum": "Peer Tutoring Sessions Attended",
    "questions_asked_3wk_mean": "In-Class Questions Asked",
    "help_seeking_delay_days": "Help-Seeking Latency (Days Delayed)",
    "is_seeking_help": "Active Academic Help-Seeking Status",
    # 5. Feedback / Sentiment
    "survey_sentiment_recent": "Bi-Weekly Pulse Survey Morale",
    "teacher_concern_flags_3wk": "Teacher Qualitative Concern Notes",
    "confidence_rating_recent": "Self-Reported Academic Confidence",
}


def format_plain_interpretation(feature: str, val: float, impact: float) -> str:
    """Provides an educator-friendly context string for an observed feature value."""
    if feature == "content_time_3wk_mean":
        if impact > 0:
            return f"Low active reading time ({val:.1f} mins/wk vs 45 min class expectation)"
        return f"Healthy active reading time ({val:.1f} mins/wk)"
    elif feature == "cumulative_score_avg":
        if impact > 0:
            return f"Low cumulative grade average ({val:.1f}%)"
        return f"Solid cumulative grade average ({val:.1f}%)"
    elif feature == "quiz_score_recent":
        if impact > 0:
            return f"Recent quiz score dropped to {val:.1f}%"
        return f"Recent quiz score is healthy ({val:.1f}%)"
    elif feature == "score_trend_slope":
        if val < 0:
            return f"Declining grade velocity ({val:.1f} pts/wk)"
        return f"Positive grade velocity (+{val:.1f} pts/wk)"
    elif feature == "survey_sentiment_recent":
        if val < 0:
            return f"Negative pulse morale ({val:+.2f} on -1 to +1 scale)"
        return f"Positive pulse morale ({val:+.2f})"
    elif feature == "confidence_rating_recent":
        if val <= 2.5:
            return f"Low self-reported academic confidence ({val:.1f}/5.0)"
        return f"Strong self-reported academic confidence ({val:.1f}/5.0)"
    elif feature == "logins_per_active_hour":
        if impact > 0:
            return f"High login count with brief reading ({val:.0f} logins/hr — superficial activity)"
        return "Balanced login-to-reading ratio"
    elif feature == "att_3wk_mean":
        if impact > 0:
            return f"Attendance dip ({val:.1f} days/wk)"
        return f"Consistent in-person attendance ({val:.1f}/5 days/wk)"
    elif feature == "office_hours_3wk_sum":
        if val == 0:
            return "No office hours visits in trailing 3 weeks"
        return f"Attended {int(val)} office hours sessions"
    elif feature == "tutoring_3wk_sum":
        if val == 0:
            return "No peer tutoring sessions attended"
        return f"Attended {int(val)} peer tutoring sessions"
    elif feature == "late_submission_ratio":
        if val > 0.2:
            return f"{val*100:.0f}% of assignments submitted late"
        return "Timely assignment submissions"
    elif feature == "unexcused_absences_sum":
        if val > 0:
            return f"{int(val)} unexcused absences recorded"
        return "Zero unexcused absences"
    elif feature == "teacher_concern_flags_3wk":
        if val > 0:
            return f"{int(val)} teacher concern note(s) logged"
        return "No teacher concern flags"
    else:
        return f"Observed value: {val:.2f}"


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
        self.shap_explainer = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TransparentMultiSignalModel":
        """Fits the calibrated multi-signal model and initializes SHAP explainer."""
        assert_no_ground_truth_leakage(list(X.columns))
        X_clean = X[self.feature_names].fillna(0.0)

        self.calibrated_model.fit(X_clean, y)
        self.is_fitted = True

        # Fit base model separately for direct tree feature importances & SHAP
        self.base_model.fit(X_clean, y)

        # Initialize SHAP Explainer
        try:
            import shap
            bg_sample = X_clean.iloc[:min(60, len(X_clean))]
            self.shap_explainer = shap.Explainer(self.base_model, bg_sample)
        except Exception:
            self.shap_explainer = None

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
        Breaks down contribution across the 5 signal families and individual features
        using SHAP attributions and plain-language educator terminology.
        """
        assert_no_ground_truth_leakage(self.feature_names)
        x_row = instance_features[self.feature_names].to_frame().T.fillna(0.0)
        risk_score = float(self.predict_risk_score(x_row)[0])

        if background_X is not None:
            bg_clean = background_X[self.feature_names].fillna(0.0)
            baseline_prob = float(np.mean(self.predict_risk_score(bg_clean)))
        else:
            baseline_prob = 0.25

        delta_total = risk_score - baseline_prob
        contributions: Dict[str, float] = {}

        # 1. Compute local feature attributions via SHAP if available
        shap_computed = False
        if self.shap_explainer is not None:
            try:
                sv = self.shap_explainer(x_row)
                vals = sv.values[0]
                if len(vals.shape) == 2:
                    vals = vals[:, 1]
                sum_shap = float(np.sum(np.abs(vals))) or 1.0
                for idx, col in enumerate(self.feature_names):
                    # Scale SHAP attributions so they sum to delta_total in risk probability space
                    contributions[col] = (float(vals[idx]) / sum_shap) * delta_total
                shap_computed = True
            except Exception:
                shap_computed = False

        # Fallback to marginal feature perturbation if SHAP was unavailable
        if not shap_computed:
            if background_X is not None:
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

        # Sorted feature drivers
        sorted_features = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)

        top_risk_drivers = [
            {
                "feature": f,
                "plain_feature": FEATURE_PLAIN_LANGUAGE_MAPPING.get(f, f),
                "family": SIGNAL_FAMILY_MAPPING.get(f, "Other"),
                "impact": round(v, 3),
                "observed_value": round(float(x_row[f].iloc[0]), 2),
                "interpretation": format_plain_interpretation(f, float(x_row[f].iloc[0]), v),
            }
            for f, v in sorted_features if v > 0
        ][:5]

        top_protective_factors = [
            {
                "feature": f,
                "plain_feature": FEATURE_PLAIN_LANGUAGE_MAPPING.get(f, f),
                "family": SIGNAL_FAMILY_MAPPING.get(f, "Other"),
                "impact": round(v, 3),
                "observed_value": round(float(x_row[f].iloc[0]), 2),
                "interpretation": format_plain_interpretation(f, float(x_row[f].iloc[0]), v),
            }
            for f, v in sorted_features if v < 0
        ][:5]

        # Full per-feature list for detailed inspection / table
        all_feature_contributions = [
            {
                "feature": f,
                "plain_feature": FEATURE_PLAIN_LANGUAGE_MAPPING.get(f, f),
                "family": SIGNAL_FAMILY_MAPPING.get(f, "Other"),
                "impact": round(v, 3),
                "direction": "Elevates Risk" if v > 0 else "Protective Factor",
                "observed_value": round(float(x_row[f].iloc[0]), 2),
                "interpretation": format_plain_interpretation(f, float(x_row[f].iloc[0]), v),
            }
            for f, v in sorted_features
        ]

        # Stakeholder Persona 4 Safeguard: Compassionate Restorative Triage
        sentiment_val = float(x_row.get("survey_sentiment_recent", pd.Series([0.0])).iloc[0])
        unexcused = float(x_row.get("unexcused_absences_sum", pd.Series([0.0])).iloc[0])
        att_val = float(x_row.get("att_3wk_mean", pd.Series([5.0])).iloc[0])
        confidence_val = float(x_row.get("confidence_rating_recent", pd.Series([3.0])).iloc[0])

        is_wellness_triage = (
            (sentiment_val <= -0.35 or confidence_val <= 2.0)
            and unexcused == 0
            and att_val >= 3.8
        )

        if is_wellness_triage:
            intervention_pathway = "RESTO_WELLNESS_CHECK"
            pathway_label = "Compassionate Wellness Check-In (Non-Academic)"
            outreach_guidance = (
                "Student exhibits low morale or confidence despite faithful in-person attendance and zero "
                "infractions. Recommend a supportive, non-evaluative check-in focused on emotional well-being "
                "rather than academic pressure."
            )
        elif risk_score >= 0.50:
            intervention_pathway = "ACADEMIC_INTERVENTION"
            pathway_label = "Academic Support & Course Scaffolding"
            outreach_guidance = (
                "Focus on courseware reading habits, assignment pacing, and connecting with peer tutoring "
                "or teacher office hours."
            )
        else:
            intervention_pathway = "STANDARD_MONITORING"
            pathway_label = "Standard Healthy Monitoring"
            outreach_guidance = "Student trajectory is stable. Continue normal positive reinforcement."

        return {
            "risk_score": round(risk_score, 3),
            "baseline_cohort_risk": round(baseline_prob, 3),
            "family_contributions": {k: round(v, 3) for k, v in family_contributions.items()},
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "all_feature_contributions": all_feature_contributions,
            "primary_driver_family": max(family_contributions.items(), key=lambda item: item[1])[0],
            "intervention_pathway": intervention_pathway,
            "pathway_label": pathway_label,
            "outreach_guidance": outreach_guidance,
        }

    def export_counselor_audit_record(
        self,
        student_id: str,
        instance_features: pd.Series,
        explanation: Dict[str, Any],
    ) -> str:
        """
        Generates a FERPA-compliant plain-language advisory note for counselor records.
        Contains zero raw machine learning jargon, avoids permanent deficit labels,
        and includes required student privacy disclaimers (Stakeholder Persona 5 safeguard).
        """
        top_risks = explanation.get("top_risk_drivers", [])
        top_protect = explanation.get("top_protective_factors", [])

        risk_bullets = "\n".join(
            [f"  - {r['plain_feature']}: {r['interpretation']} (+{r['impact']*100:.1f}% risk impact)" for r in top_risks[:3]]
        ) or "  - None identified"

        protect_bullets = "\n".join(
            [f"  - {p['plain_feature']}: {p['interpretation']} ({p['impact']*100:.1f}% risk mitigation)" for p in top_protect[:3]]
        ) or "  - None identified"

        record = f"""================================================================================
CONFIDENTIAL STUDENT SUPPORT CONSULTATION MEMORANDUM
Student ID: {student_id} | Advisory Status: {explanation.get('pathway_label', 'Standard Review')}
Classification: Human-in-the-Loop Early Intervention (Non-Disciplinary)
================================================================================

1. PRIMARY OBSERVATIONAL DRIVERS (Why Support is Recommended):
{risk_bullets}

2. DEMONSTRATED PROTECTIVE STRENGTHS (Student Assets):
{protect_bullets}

3. RECOMMENDED COUNSELOR ACTION:
  Guidance: {explanation.get('outreach_guidance', 'Check in with student.')}
  Pathway: {explanation.get('pathway_label', 'General Check-In')}

FERPA & PRIVACY NOTICE:
This record is an ephemeral decision-support aid designed strictly for internal
counselor guidance. It does NOT constitute an academic evaluation, disciplinary
record, or permanent educational transcript entry.
================================================================================"""
        return record
