"""
Synthetic Data Generator for Transparent Disengagement Early-Warning System.

Generates a realistic student cohort with:
1. Dynamic latent engagement trajectories E_{i,t} across 16 weeks.
2. 4 core student archetypes + 3 edge cases (N=500 total).
3. Non-trivial ground-truth label: is_disengaged is driven by latent engagement,
   NOT by simple attendance or marks thresholds.
4. Five distinct signal families with realistic observational lag, missingness, and noise.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


def _compute_rolling_slope_and_var(scores: List[float]) -> Tuple[float, float]:
    """Computes 3-week rolling slope and variance from recent quiz scores."""
    valid_scores = [s for s in scores[-3:] if s is not None and not np.isnan(s)]
    if len(valid_scores) < 2:
        return 0.0, 0.0
    var = float(np.var(valid_scores))
    x = np.arange(len(valid_scores))
    slope = float(np.polyfit(x, valid_scores, 1)[0])
    return slope, var


def generate_cohort(
    n_students: int = 500,
    n_weeks: int = 16,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a full synthetic cohort dataframe.

    Archetype Distribution (N=500):
    - Consistently Engaged: 250 (50.0%)
    - Quietly Struggling but Attending: 100 (20.0%)
    - Attending but Checked Out: 75 (15.0%)
    - Genuinely Improving / Recovering: 50 (10.0%)
    - Edge Case 1 - Transfer Student: 10 (2.0%)
    - Edge Case 2 - Signal Gamer: 8 (1.6%)
    - Edge Case 3 - Acute Shock (Family Emergency): 7 (1.4%)
    """
    np.random.seed(seed)

    # Assign archetypes
    archetypes = (
        ["consistently_engaged"] * 250
        + ["quietly_struggling"] * 100
        + ["checked_out"] * 75
        + ["genuinely_improving"] * 50
        + ["transfer_student"] * 10
        + ["signal_gamer"] * 8
        + ["acute_shock"] * 7
    )

    records: List[Dict] = []

    for idx, archetype in enumerate(archetypes):
        student_id = f"STU_{idx+1:04d}"

        # Initialize student latent baseline E_0
        if archetype == "consistently_engaged":
            latent_e = float(np.clip(np.random.normal(0.85, 0.04), 0.70, 0.98))
        elif archetype == "quietly_struggling":
            latent_e = float(np.clip(np.random.normal(0.80, 0.04), 0.70, 0.90))
        elif archetype == "checked_out":
            latent_e = float(np.clip(np.random.normal(0.74, 0.04), 0.65, 0.85))
        elif archetype == "genuinely_improving":
            latent_e = float(np.clip(np.random.normal(0.38, 0.05), 0.25, 0.50))
        elif archetype == "transfer_student":
            latent_e = float(np.clip(np.random.normal(0.68, 0.08), 0.50, 0.85))
        elif archetype == "signal_gamer":
            latent_e = float(np.clip(np.random.normal(0.62, 0.05), 0.50, 0.75))
        elif archetype == "acute_shock":
            latent_e = float(np.clip(np.random.normal(0.86, 0.04), 0.75, 0.95))
        else:
            latent_e = 0.75

        # Track history for temporal calculations
        history_latent_e: List[float] = []
        quiz_history: List[float] = []

        start_week = 7 if archetype == "transfer_student" else 1

        for w in range(1, n_weeks + 1):
            if w < start_week:
                continue

            # 1. Update Latent Engagement Trajectory with parametrized mu(t)
            if archetype == "consistently_engaged":
                # Mean-reverting around 0.85
                drift = -0.05 * (latent_e - 0.85)
            elif archetype == "quietly_struggling":
                # Gradual decay accelerating over time
                drift = -0.038 - 0.002 * w
            elif archetype == "checked_out":
                # Rapid cliff in weeks 4-8
                if w <= 3:
                    drift = -0.02
                elif w <= 8:
                    drift = -0.065
                else:
                    drift = -0.015
            elif archetype == "genuinely_improving":
                # Strong early recovery through week 8
                if w <= 8:
                    drift = 0.065
                elif latent_e < 0.85:
                    drift = 0.02
                else:
                    drift = 0.0
            elif archetype == "transfer_student":
                drift = 0.01
            elif archetype == "signal_gamer":
                drift = -0.042
            elif archetype == "acute_shock":
                if w == 7:
                    drift = -0.48  # Acute external crisis
                elif w in (8, 9):
                    drift = 0.24   # Rapid post-crisis rebound
                else:
                    drift = -0.05 * (latent_e - 0.85)

            # Stochastic perturbation
            noise_e = np.random.normal(0, 0.03)
            latent_e = float(np.clip(latent_e + drift + noise_e, 0.02, 0.98))
            history_latent_e.append(latent_e)

            # Ground-Truth Disengagement: < 0.40 for >= 2 consecutive weeks
            if len(history_latent_e) >= 2:
                is_disengaged = bool(history_latent_e[-1] < 0.40 and history_latent_e[-2] < 0.40)
            else:
                is_disengaged = bool(history_latent_e[-1] < 0.35)

            # 2. Emit Signal Family 1: Attendance
            if archetype == "quietly_struggling":
                # Compulsively high physical attendance despite disengaging!
                days_present = int(np.random.choice([4, 5], p=[0.15, 0.85]))
                tardy_count = int(np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10]))
                unexcused = 0
            elif archetype == "checked_out":
                # Present in seat most of the time
                days_present = int(np.random.choice([3, 4, 5], p=[0.20, 0.45, 0.35]))
                tardy_count = int(np.random.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10]))
                unexcused = int(np.random.choice([0, 1], p=[0.75, 0.25]))
            elif archetype == "genuinely_improving":
                if w <= 3:
                    days_present = int(np.random.choice([2, 3, 4], p=[0.30, 0.45, 0.25]))
                    tardy_count = int(np.random.choice([1, 2, 3], p=[0.40, 0.40, 0.20]))
                    unexcused = int(np.random.choice([1, 2], p=[0.70, 0.30]))
                else:
                    days_present = int(np.random.choice([4, 5], p=[0.20, 0.80]))
                    tardy_count = int(np.random.choice([0, 1], p=[0.85, 0.15]))
                    unexcused = 0
            elif archetype == "acute_shock" and w == 7:
                # Acute dip during family crisis
                days_present = int(np.random.choice([0, 1], p=[0.7, 0.3]))
                tardy_count = 0
                unexcused = 0  # Excused emergency
            else:
                # Correlated with latent engagement
                prob_present = 0.50 + 0.48 * latent_e
                days_present = int(np.random.binomial(5, np.clip(prob_present, 0.1, 0.99)))
                tardy_count = int(np.random.binomial(days_present, max(0.02, 0.35 * (1 - latent_e))))
                unexcused = int(np.random.binomial(5 - days_present, 0.4))

            days_absent = 5 - days_present
            unexcused = min(unexcused, days_absent)

            # 3. Emit Signal Family 2: Activity (LMS / Coursework)
            if archetype == "signal_gamer":
                # Excessive logins, but negligible active reading time
                lms_logins = int(np.random.normal(65, 10))
                content_time_minutes = float(max(5.0, np.random.normal(12.0, 4.0)))
                assignment_subs = int(np.random.choice([1, 2], p=[0.6, 0.4]))
                late_submissions = int(np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
                discussion_posts = int(np.random.poisson(0.3))
            elif archetype == "quietly_struggling":
                # High logins out of anxiety, but collapsing reading time and late submissions
                lms_logins = int(np.random.normal(18, 4))
                content_time_minutes = float(max(15.0, np.random.normal(25.0 + 80.0 * latent_e, 15.0)))
                assignment_subs = int(np.random.choice([1, 2], p=[0.4, 0.6]))
                late_submissions = int(np.random.poisson(1.4 * (1.0 - latent_e)))
                discussion_posts = int(np.random.poisson(0.4 * latent_e))
            elif archetype == "checked_out":
                lms_logins = int(max(1, np.random.normal(4, 2)))
                content_time_minutes = float(max(10.0, np.random.normal(20.0 + 40.0 * latent_e, 10.0)))
                assignment_subs = int(np.random.choice([0, 1, 2], p=[0.35, 0.50, 0.15]))
                late_submissions = int(np.random.poisson(0.8))
                discussion_posts = 0
            else:
                lms_logins = int(max(2, np.random.normal(6 + 14 * latent_e, 3)))
                content_time_minutes = float(max(15.0, np.random.normal(30.0 + 160.0 * latent_e, 25.0)))
                assignment_subs = int(np.random.choice([1, 2, 3], p=[0.15, 0.70, 0.15]))
                late_submissions = int(np.random.poisson(max(0.05, 1.2 * (1.0 - latent_e))))
                discussion_posts = int(np.random.poisson(max(0.1, 2.5 * latent_e)))

            # 4. Emit Signal Family 3: Assessment
            # Assessment scores lag behind latent engagement
            if archetype == "quietly_struggling":
                # Scores hold up initially via brute-force memorization, then crater
                if w <= 8:
                    base_score = 78.0 - 1.2 * w + np.random.normal(0, 4)
                else:
                    base_score = 68.0 - 4.5 * (w - 8) + np.random.normal(0, 6)
            elif archetype == "genuinely_improving":
                # Early low marks (failing), steadily recovering
                if w <= 3:
                    base_score = 48.0 + 3.0 * w + np.random.normal(0, 5)
                else:
                    base_score = 60.0 + 3.2 * (w - 3) + np.random.normal(0, 4)
            elif archetype == "acute_shock" and w == 7:
                base_score = 38.0 + np.random.normal(0, 5)
            else:
                # Correlated with latent engagement plus noise
                base_score = 35.0 + 60.0 * latent_e + np.random.normal(0, 6.0)

            quiz_score = float(np.clip(base_score, 10.0, 100.0))
            quiz_history.append(quiz_score)

            cumulative_score_avg = float(np.mean(quiz_history))
            score_trend_slope, score_variance = _compute_rolling_slope_and_var(quiz_history)

            # 5. Emit Signal Family 4: Help-Seeking Behaviors
            if archetype == "quietly_struggling":
                # Complete absence of help-seeking due to anxiety / shame
                questions_asked = int(np.random.poisson(0.2))
                office_hours = 0
                tutoring = 0
                help_delay = float(np.random.uniform(9.0, 14.0))
            elif archetype == "genuinely_improving":
                # Active help-seeking turning performance around
                questions_asked = int(np.random.poisson(3.8))
                office_hours = int(np.random.choice([1, 2], p=[0.6, 0.4]))
                tutoring = int(np.random.choice([1, 2], p=[0.7, 0.3]))
                help_delay = float(np.random.uniform(0.5, 2.5))
            elif archetype == "checked_out":
                questions_asked = 0
                office_hours = 0
                tutoring = 0
                help_delay = 14.0
            else:
                questions_asked = int(np.random.poisson(max(0.1, 2.5 * latent_e)))
                office_hours = int(np.random.binomial(2, np.clip(0.4 * latent_e, 0.0, 0.8)))
                tutoring = int(np.random.binomial(1, np.clip(0.3 * latent_e, 0.0, 0.6)))
                help_delay = float(np.clip(np.random.exponential(max(1.0, 7.0 * (1.0 - latent_e))), 0.0, 14.0))

            # 6. Emit Signal Family 5: Feedback / Sentiment (Lagged & Noisy)
            # Pulse survey happens bi-weekly (even weeks)
            if w % 2 == 0:
                lagged_e = history_latent_e[-2] if len(history_latent_e) >= 2 else latent_e
                sentiment_target = 2.0 * lagged_e - 1.0
                survey_sentiment = float(np.clip(sentiment_target + np.random.normal(0, 0.35), -1.0, 1.0))
                confidence_val = int(np.clip(np.round(1.0 + 4.0 * lagged_e + np.random.normal(0, 0.7)), 1, 5))
            else:
                survey_sentiment = None
                confidence_val = None

            # Teacher qualitative note with 2-week observation lag
            lagged_note_e = history_latent_e[-3] if len(history_latent_e) >= 3 else latent_e
            if archetype == "acute_shock" and w in (7, 8):
                teacher_note_flag = 1  # Note mentions temporary family issue
            elif lagged_note_e < 0.40:
                # Even for disengaged students, teachers only file a note ~55% of the time
                teacher_note_flag = int(np.random.choice([0, 1, 2], p=[0.45, 0.35, 0.20]))
            elif lagged_note_e < 0.70:
                teacher_note_flag = int(np.random.choice([0, 1], p=[0.80, 0.20]))
            else:
                teacher_note_flag = int(np.random.choice([0, 1], p=[0.96, 0.04]))

            record = {
                # Identifiers
                "student_id": student_id,
                "week": w,
                # Signal Family 1: Attendance
                "days_present": days_present,
                "days_absent": days_absent,
                "tardy_count": tardy_count,
                "unexcused_absences": unexcused,
                # Signal Family 2: Activity
                "lms_logins": lms_logins,
                "content_time_minutes": round(content_time_minutes, 1),
                "assignment_submissions": assignment_subs,
                "late_submissions": late_submissions,
                "discussion_posts": discussion_posts,
                # Signal Family 3: Assessment
                "quiz_score": round(quiz_score, 1),
                "cumulative_score_avg": round(cumulative_score_avg, 1),
                "score_trend_slope": round(score_trend_slope, 2),
                "score_variance": round(score_variance, 2),
                # Signal Family 4: Help-Seeking
                "questions_asked": questions_asked,
                "office_hours_attended": office_hours,
                "tutoring_sessions": tutoring,
                "help_seeking_delay_days": round(help_delay, 1),
                # Signal Family 5: Feedback / Sentiment (Weak & Noisy)
                "survey_sentiment": round(survey_sentiment, 2) if survey_sentiment is not None else np.nan,
                "teacher_note_flag": teacher_note_flag,
                "confidence_rating": confidence_val if confidence_val is not None else np.nan,
                # Ground-Truth / Audit Vault (NEVER TO BE USED AS FEATURES)
                "archetype": archetype,
                "latent_engagement": round(latent_e, 3),
                "is_disengaged": is_disengaged,
            }
            records.append(record)

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cohort dataset.")
    parser.add_argument("--output", type=str, default="data/synthetic_cohort.csv", help="Output CSV path")
    parser.add_argument("--students", type=int, default=500, help="Number of students")
    parser.add_argument("--weeks", type=int, default=16, help="Number of semester weeks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    print(f"Generating synthetic cohort: {args.students} students, {args.weeks} weeks (seed={args.seed})...")
    df = generate_cohort(n_students=args.students, n_weeks=args.weeks, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Successfully wrote {len(df)} rows to {args.output}")
    print("\nArchetype distribution:")
    print(df[df["week"] == 16]["archetype"].value_counts())
    print("\nDisengagement rate at Week 16:")
    print(df[df["week"] == 16]["is_disengaged"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
