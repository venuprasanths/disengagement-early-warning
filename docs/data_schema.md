# Data Schema & Latent Engagement Trajectory Specification

## 1. Ground-Truth & Audit Wall Notice

> [!CRITICAL]
> **INFORMATION BARRIER REQUIREMENT**:
> The columns `archetype`, `latent_engagement`, and `is_disengaged` are **strictly ground-truth / audit columns**.
> They represent the unobserved latent reality and ground-truth labels used **only** for backtest benchmarking, calibration verification, and error analysis.
> Under no circumstances may `archetype`, `latent_engagement`, or `is_disengaged` be passed as features or inputs into either the baseline model or the main multi-signal model.

---

## 2. Cohort Population Size & Distribution

The synthetic benchmark cohort models a realistic high school / introductory college semester consisting of **$N = 500$ students** observed weekly across **$T = 16$ weeks** (totaling up to 8,000 student-week observation rows).

| Archetype / Cohort Segment | Count ($N_i$) | % of Cohort | Description & Behavioral Pattern |
| :--- | :--- | :--- | :--- |
| **Archetype 1: Quietly Struggling but Attending** | 100 | 20.0% | Diligent physical attendance ($days\_present \approx 5$) and frequent logins, but failing silently. Avoids help-seeking due to anxiety; sentiment deteriorates; marks hold early then crash late. |
| **Archetype 2: Attending but Checked Out** | 75 | 15.0% | In-seat presence without cognitive engagement. Minimal discussion posts, 0 questions asked, declining submissions. |
| **Archetype 3: Genuinely Improving / Recovering** | 50 | 10.0% | Starts semester with poor marks (<60%) and shaky attendance, but engages heavily with tutoring/office hours; scores trending upwards. |
| **Archetype 4: Consistently Engaged** | 250 | 50.0% | Stable high engagement ($E_{i,t} \approx 0.85$), reliable submission history, healthy help-seeking when needed. |
| **Edge Case 1: Transfer Student (Missing Data)** | 10 | 2.0% | Enrolls at Week 7. Weeks 1–6 records are `NaN`/absent. Tests model's ability to handle sparse history and widen uncertainty intervals. |
| **Edge Case 2: Signal Gamer (Surface Activity)** | 8 | 1.6% | Generates excessive LMS logins (50+/wk) but active content reading time is near zero; does not seek help; quiz scores drop. |
| **Edge Case 3: Acute Shock (Family Emergency)** | 7 | 1.4% | Previously high-performing student suffers acute 1-week attendance/quiz collapse at Week 7, accompanied by mitigating teacher note, rebounding by Week 8–9. Tests non-overreaction. |
| **Total Cohort** | **500** | **100.0%** | |

---

## 3. Parametrized Latent Trajectory Functions $\mu_{\text{archetype}}(t)$

The latent engagement state $E_{i,t} \in [0.0, 1.0]$ evolves according to:
$$E_{i,t} = \text{clip}\left( E_{i,t-1} + \mu_{\text{archetype}}(t) + \epsilon_{i,t}, 0.0, 1.0 \right)$$
where $\epsilon_{i,t} \sim \mathcal{N}(0, 0.035^2)$.

### Parametrized Drift per Archetype:

1. **Archetype 1 (Quietly Struggling)**:
   - Initial state: $E_{i,0} \sim \mathcal{N}(0.78, 0.04^2)$
   - Drift function:
     $$\mu_{\text{quietly\_struggling}}(t) = -0.040 - 0.002 \cdot t$$
     *(Gradual decay accelerating as concepts compound and student feels increasingly overwhelmed).*

2. **Archetype 2 (Checked Out)**:
   - Initial state: $E_{i,0} \sim \mathcal{N}(0.72, 0.05^2)$
   - Drift function:
     $$\mu_{\text{checked\_out}}(t) = \begin{cases} -0.025 & \text{if } t \le 3 \\ -0.065 & \text{if } 4 \le t \le 8 \\ -0.015 & \text{if } t > 8 \end{cases}$$
     *(Rapid mid-semester cliff as student mentally disconnects).*

3. **Archetype 3 (Genuinely Improving)**:
   - Initial state: $E_{i,0} \sim \mathcal{N}(0.38, 0.05^2)$
   - Drift function:
     $$\mu_{\text{improving}}(t) = \begin{cases} +0.065 & \text{if } t \le 8 \\ +0.020 & \text{if } t > 8 \text{ and } E_{i,t-1} < 0.85 \\ 0.0 & \text{otherwise} \end{cases}$$
     *(Strong positive recovery slope from early tutoring intervention).*

4. **Archetype 4 (Consistently Engaged)**:
   - Initial state: $E_{i,0} \sim \mathcal{N}(0.85, 0.04^2)$
   - Drift function:
     $$\mu_{\text{engaged}}(t) = -0.05 \cdot (E_{i,t-1} - 0.85)$$
     *(Mean-reverting Ornstein-Uhlenbeck style stability around 0.85).*

5. **Edge Cases**:
   - **Transfer Student**: Initialized at $t=7$ with $E_{i,7} \sim \mathcal{N}(0.65, 0.10^2)$ and mild positive drift $+0.01$.
   - **Signal Gamer**: $E_{i,0} \sim \mathcal{N}(0.60, 0.05^2)$ with steady negative drift $\mu(t) = -0.045$.
   - **Acute Shock**: Starts at $E_{i,0} \approx 0.85$. At $t=7$, an external shock shock $\Delta_{i,7} = -0.45$ occurs; followed by recovery drift $\mu(t) = +0.22$ in weeks 8 and 9.

### Ground-Truth Disengagement Condition:
A student is labeled `is_disengaged = True` at week $t$ if:
$$E_{i,t} < 0.40 \quad \text{AND} \quad E_{i,t-1} < 0.40$$
*(Requires at least 2 consecutive weeks below the critical 0.40 threshold to avoid transient noise).*

---

## 4. Signal Emission Formulas & Realistic Lag/Noise

To prevent `teacher_note_flag` and `survey_sentiment` from acting as trivial cheat readouts of ground truth, they are generated with substantial observational lag, low frequency, and random noise:

### 4.1 Survey Sentiment (`survey_sentiment` $\in [-1.0, 1.0]$)
- Students only complete pulse surveys every $\sim 2$ weeks (simulated missingness on odd weeks).
- Survey responses have an emotional filter lag of 1 week:
  $$\text{sentiment\_target}_{i,t} = 2.0 \cdot E_{i, t-1} - 1.0$$
- Substantial response noise $\eta_{\text{survey}} \sim \mathcal{N}(0, 0.35^2)$:
  $$\text{survey\_sentiment}_{i,t} = \text{clip}\left(\text{sentiment\_target}_{i,t} + \eta_{\text{survey}}, -1.0, 1.0\right)$$
- *Result*: A student with $E=0.30$ will sometimes report neutral or mildly positive sentiment ($+0.1$) due to polite response bias or social desirability bias.

### 4.2 Teacher Qualitative Note Flag (`teacher_note_flag` $\in \{0, 1, 2\}$)
- Teachers manage 120+ students and do not notice subtle disengagement immediately.
- Observation probability is lagged by 2 weeks and probabilistic:
  $$P(\text{Note} \mid E_{i,t-2}) = \begin{cases} 0.05 & \text{if } E_{i,t-2} \ge 0.70 \text{ (baseline random note)} \\ 0.25 & \text{if } 0.40 \le E_{i,t-2} < 0.70 \\ 0.60 & \text{if } E_{i,t-2} < 0.40 \end{cases}$$
- Severity:
  - 0: None / positive observation
  - 1: Mild concern ("Seems quiet lately", "missed one reading")
  - 2: Serious concern ("Unresponsive in group work", "sleeping in class")
- *Result*: In over 40% of weeks, even severely disengaged students have no teacher note filed (`flag = 0`).

---

## 5. Feature Schema Table

| Column Name | Type | Signal Family | Feature for Models? | Description |
| :--- | :--- | :--- | :--- | :--- |
| `student_id` | `str` | Identifier | No (ID) | Unique student token |
| `week` | `int` | Temporal | Yes (ordering) | Semester week ($1..16$) |
| `days_present` | `int` | Attendance | **Yes** | In-person attendance ($0..5$) |
| `days_absent` | `int` | Attendance | **Yes** | In-person absences ($0..5$) |
| `tardy_count` | `int` | Attendance | **Yes** | Late arrivals ($0..5$) |
| `unexcused_absences`| `int` | Attendance | **Yes** | Unexcused absences ($0..5$) |
| `lms_logins` | `int` | Activity | **Yes** | Total LMS sessions |
| `content_time_minutes`| `float` | Activity | **Yes** | Active reading/video minutes |
| `assignment_submissions`| `int` | Activity | **Yes** | Completed assignments |
| `late_submissions` | `int` | Activity | **Yes** | Assignments submitted late |
| `discussion_posts` | `int` | Activity | **Yes** | Forum contributions |
| `quiz_score` | `float` | Assessment | **Yes** | Percentage ($0..100$) or `NaN` |
| `cumulative_score_avg`| `float` | Assessment | **Yes** | Running average score |
| `score_trend_slope` | `float` | Assessment | **Yes** | 3-week linear regression slope |
| `score_variance` | `float` | Assessment | **Yes** | 3-week score variance |
| `questions_asked` | `int` | Help-Seeking | **Yes** | Questions in forum/class |
| `office_hours_attended`| `int` | Help-Seeking | **Yes** | 1-on-1 instructor sessions |
| `tutoring_sessions` | `int` | Help-Seeking | **Yes** | Peer/tutor sessions attended |
| `help_seeking_delay_days`| `float`| Help-Seeking | **Yes** | Days between bad quiz & help |
| `survey_sentiment` | `float` | Feedback | **Yes** | Pulse sentiment ($-1.0..+1.0$) |
| `teacher_note_flag` | `int` | Feedback | **Yes** | $0=$ none, $1=$ mild, $2=$ acute |
| `confidence_rating` | `int` | Feedback | **Yes** | Self-reported rating ($1..5$) |
| `archetype` | `str` | Ground Truth | ❌ **AUDIT ONLY** | Archetype identifier |
| `latent_engagement`| `float` | Ground Truth | ❌ **AUDIT ONLY** | True continuous engagement |
| `is_disengaged` | `bool` | Ground Truth | ❌ **AUDIT ONLY** | True ground-truth label |

---

## 6. Pydantic Code Schema (`src/schema.py`)

```python
from pydantic import BaseModel, Field
from typing import Optional

class StudentWeeklyRecord(BaseModel):
    # Identifiers
    student_id: str = Field(..., description="Unique student ID")
    week: int = Field(..., ge=1, le=16, description="Week index 1-16")
    
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
    
    # 5. Feedback / Sentiment
    survey_sentiment: Optional[float] = Field(None, ge=-1.0, le=1.0)
    teacher_note_flag: int = Field(..., ge=0, le=2)
    confidence_rating: Optional[int] = Field(None, ge=1, le=5)

class GroundTruthAuditRecord(BaseModel):
    """
    STRICT AUDIT VAULT: Never passed into model training or inference!
    """
    student_id: str
    week: int
    archetype: str
    latent_engagement: float = Field(..., ge=0.0, le=1.0)
    is_disengaged: bool
```
