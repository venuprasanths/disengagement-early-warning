"""
Data Schema and Information Barrier Enforcer for Disengagement Early-Warning System.
"""
from typing import Optional, List
from pydantic import BaseModel, Field

# ==============================================================================
# INFORMATION BARRIER VAULT & COLUMN GROUPS
# ==============================================================================
# The system enforces a strict information barrier between operational features
# and ground-truth evaluation data. Under no circumstances may any column in
# GROUND_TRUTH_COLUMNS be accessible during feature extraction, model training,
# or real-time inference.

# Columns that MUST NEVER be used as features in baseline or main models.
# These represent unobservable latent engagement states or post-hoc labels.
GROUND_TRUTH_COLUMNS: List[str] = [
    "archetype",          # Latent synthetic archetype (e.g., 'quietly_struggling', 'signal_gamer')
    "latent_engagement",  # Continuous latent state E_{i,t} in [0.0, 1.0]
    "is_disengaged",      # Binary true outcome label (E_{i,t} < 0.40)
]

# Time and identity coordinates for each observation row
ID_COLUMNS: List[str] = [
    "student_id",         # Unique student identifier (e.g., 'STU_0042')
    "week",               # Discrete academic semester week index (1 to 16)
]

# Signal Family 1: Physical / In-Person Attendance & Punctuality
# Represents physical presence; note that 'quietly struggling' students attend 5/5 days.
ATTENDANCE_COLUMNS: List[str] = [
    "days_present",       # Integer days in classroom seat (0 to 5)
    "days_absent",        # Integer days absent (0 to 5, excused + unexcused)
    "tardy_count",        # Punctuality infractions per week (0 to 5)
    "unexcused_absences", # Absences without verified parental/medical excuse (0 to 5)
]

# Signal Family 2: Learning Management System (LMS) Digital Activity
# Captures digital engagement, time-on-task, and submission pacing.
ACTIVITY_COLUMNS: List[str] = [
    "lms_logins",              # Count of authentication sessions logged in LMS portal
    "content_time_minutes",    # Time actively reading/interacting with course materials (minutes)
    "assignment_submissions",  # Total weekly deliverables submitted
    "late_submissions",        # Count of deliverables submitted past published deadlines
    "discussion_posts",        # Peer/instructor forum discussion contributions
]

# Signal Family 3: Assessment Trends & Academic Mastery Trajectory
# Focuses on velocity, variance, and rolling momentum rather than static snapshots.
ASSESSMENT_COLUMNS: List[str] = [
    "quiz_score",          # Weekly formative evaluation percentage (0.0 to 100.0, nullable)
    "cumulative_score_avg",# Running semester-to-date weighted gradebook average (0.0 to 100.0)
    "score_trend_slope",   # 3-week linear regression slope of performance (points/week)
    "score_variance",      # 3-week trailing grade variance (stability indicator)
]

# Signal Family 4: Proactive Academic Help-Seeking & Support Utilization
# Differentiates struggling students who seek help from those who withdraw silently.
HELP_SEEKING_COLUMNS: List[str] = [
    "questions_asked",         # In-class or asynchronous questions directed to instructor
    "office_hours_attended",   # 1-on-1 instructor/TA consultation appointments attended
    "tutoring_sessions",       # Peer tutoring center sessions completed
    "help_seeking_delay_days", # Latency between struggling grade and first help inquiry
]

# Signal Family 5: Qualitative Student Voice, Morale & Teacher Observations
# Weak Bayesian priors reflecting emotional state and qualitative faculty concern.
FEEDBACK_COLUMNS: List[str] = [
    "survey_sentiment",    # Bi-weekly pulse survey sentiment polarity (-1.0 to +1.0, nullable)
    "teacher_note_flag",   # Faculty concern level: 0=None, 1=Mild Concern, 2=Acute Flag
    "confidence_rating",   # Self-reported academic self-efficacy rating (1 to 5, nullable)
]

# Complete set of raw behavioral features admissible for feature engineering
RAW_FEATURE_COLUMNS: List[str] = (
    ATTENDANCE_COLUMNS
    + ACTIVITY_COLUMNS
    + ASSESSMENT_COLUMNS
    + HELP_SEEKING_COLUMNS
    + FEEDBACK_COLUMNS
)

# Constrained subset utilized strictly by the conventional lagging baseline
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
