"""
Data Schema and Information Barrier Enforcer for Disengagement Early-Warning System.
"""
from typing import Optional, List
from pydantic import BaseModel, Field

# Columns that MUST NEVER be used as features in baseline or main models
GROUND_TRUTH_COLUMNS: List[str] = [
    "archetype",
    "latent_engagement",
    "is_disengaged",
]

ID_COLUMNS: List[str] = [
    "student_id",
    "week",
]

ATTENDANCE_COLUMNS: List[str] = [
    "days_present",
    "days_absent",
    "tardy_count",
    "unexcused_absences",
]

ACTIVITY_COLUMNS: List[str] = [
    "lms_logins",
    "content_time_minutes",
    "assignment_submissions",
    "late_submissions",
    "discussion_posts",
]

ASSESSMENT_COLUMNS: List[str] = [
    "quiz_score",
    "cumulative_score_avg",
    "score_trend_slope",
    "score_variance",
]

HELP_SEEKING_COLUMNS: List[str] = [
    "questions_asked",
    "office_hours_attended",
    "tutoring_sessions",
    "help_seeking_delay_days",
]

FEEDBACK_COLUMNS: List[str] = [
    "survey_sentiment",
    "teacher_note_flag",
    "confidence_rating",
]

RAW_FEATURE_COLUMNS: List[str] = (
    ATTENDANCE_COLUMNS
    + ACTIVITY_COLUMNS
    + ASSESSMENT_COLUMNS
    + HELP_SEEKING_COLUMNS
    + FEEDBACK_COLUMNS
)

BASELINE_FEATURE_COLUMNS: List[str] = [
    "days_present",
    "days_absent",
    "cumulative_score_avg",
]


class StudentWeeklyRecord(BaseModel):
    """Observable weekly record for a single student."""
    # Identifiers
    student_id: str = Field(..., description="Unique student identifier")
    week: int = Field(..., ge=1, le=16, description="Semester week index (1-16)")

    # 1. Attendance
    days_present: int = Field(..., ge=0, le=5)
    days_absent: int = Field(..., ge=0, le=5)
    tardy_count: int = Field(..., ge=0, le=5)
    unexcused_absences: int = Field(..., ge=0, le=5)

    # 2. Activity
    lms_logins: int = Field(..., ge=0)
    content_time_minutes: float = Field(..., ge=0.0)
    assignment_submissions: int = Field(..., ge=0)
    late_submissions: int = Field(..., ge=0)
    discussion_posts: int = Field(..., ge=0)

    # 3. Assessment
    quiz_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    cumulative_score_avg: float = Field(..., ge=0.0, le=100.0)
    score_trend_slope: float = Field(0.0)
    score_variance: float = Field(0.0, ge=0.0)

    # 4. Help-Seeking
    questions_asked: int = Field(..., ge=0)
    office_hours_attended: int = Field(..., ge=0)
    tutoring_sessions: int = Field(..., ge=0)
    help_seeking_delay_days: float = Field(..., ge=0.0)

    # 5. Feedback / Sentiment (Weak, noisy signals)
    survey_sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0)
    teacher_note_flag: int = Field(..., ge=0, le=2)
    confidence_rating: Optional[int] = Field(None, ge=1, le=5)


class GroundTruthAuditRecord(BaseModel):
    """
    STRICT AUDIT VAULT:
    Contains latent ground-truth state and labels.
    Never permitted into feature pipelines, baseline models, or main models.
    Used ONLY in backtesting, calibration checks, and error analysis.
    """
    student_id: str
    week: int
    archetype: str
    latent_engagement: float = Field(..., ge=0.0, le=1.0)
    is_disengaged: bool


def assert_no_ground_truth_leakage(columns: List[str]) -> None:
    """Verifies that no ground-truth/audit column is included in a feature set."""
    leakages = [col for col in columns if col in GROUND_TRUTH_COLUMNS]
    if leakages:
        raise ValueError(
            f"CRITICAL INFORMATION BARRIER VIOLATION! Ground-truth columns {leakages} "
            f"found in feature set. These must only be used in evaluation/auditing."
        )
