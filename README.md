# Transparent Disengagement Early-Warning System for Schools

An uncertainty-aware, interpretable early warning platform for schools supporting diverse learning paces. The system fuses five weak signal families into a calibrated risk score, quantifies epistemic and aleatoric uncertainty, exposes the trade-off between conflicting stakeholder objectives (Academic Counselor Recall vs. Student/Parent Precision), and provides plain-language local SHAP explanations for counselors.

---

## 1. Problem & Core Distinction

Schools conventionally rely on **attendance thresholds** (e.g., `< 80%`) and **failing test scores** (e.g., `< 60%`) to trigger academic interventions. These are **lagging, punitive indicators**:
- By the time attendance or marks crater, disengagement is already deeply entrenched.
- Remediation is delayed, costly, and demoralizing.
- Students silently disengage while technically "passing" on paper ("Quietly Struggling").

### Our Solution: Multi-Signal Behavioral Fusion
This system fuses 5 distinct signal families:
1. **Attendance**: In-seat days present, tardiness, unexcused absence patterns.
2. **Activity**: LMS logins, active reading/content time, submission timeliness, discussion participation.
3. **Assessment Trends**: 3-week score trajectory slope, performance variance, running averages.
4. **Help-Seeking Behaviors**: Office hours visits, peer tutoring attendance, questions asked, help-seeking latency.
5. **Qualitative Feedback / Morale**: Bi-weekly student pulse sentiment, self-reported confidence, teacher observation flags.

---

## 2. Information Barrier & Ethical Principles

To eliminate data leakage and ensure realistic modeling:
- **Strict Ground-Truth Vault**: The latent engagement state $E_{i,t}$, synthetic archetype, and true disengagement label `is_disengaged` are strictly quarantined for evaluation/auditing. Neither the baseline nor the main model ever has access to these fields.
- **Weak, Noisy Qualitative Signals**: Teacher notes and pulse surveys reflect realistic observation lags and noise, ensuring they act as weak Bayesian priors rather than ground-truth proxies.
- **Human-in-the-Loop & FERPA Alignment**: Risk scores are ephemeral resource-allocation recommendations for counselors, never automated penalties or permanent disciplinary marks.

---

## 3. Project Deliverables Directory

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **1. Stakeholder Assumptions** | [`docs/stakeholder_assumptions.md`](docs/stakeholder_assumptions.md) | Formulates Counselor Recall vs Student Precision conflict and ethical bounds. |
| **2. Architecture Diagram** | [`docs/architecture.md`](docs/architecture.md) | Mermaid pipeline diagram, data flow, and temporal holdout specification. |
| **3. Data Schema & Formulas** | [`docs/data_schema.md`](docs/data_schema.md) | 5 signal families schema, cohort sizes ($N=500$), parametrized drift functions $\mu(t)$. |
| **4. Synthetic Data Generator** | [`data/synthetic_data_generator.py`](data/synthetic_data_generator.py) | Dynamic state-space latent engagement generator ($N=500$, $T=16$, $7,940$ rows); supports custom cohort distributions. |
| **5. Lagging Baseline Model** | [`src/baseline_model.py`](src/baseline_model.py) | Current-practice attendance+marks threshold benchmark. |
| **6. Feature Engineering** | [`src/features.py`](src/features.py) | Rolling 3-week trailing features, score slopes, and gaming divergence ratios. |
| **7. Main Multi-Signal Model** | [`src/main_model.py`](src/main_model.py) | Calibrated model with TreeSHAP plain-language local explainability, restorative triage, and FERPA memo export. |
| **8. Uncertainty Estimator** | [`src/uncertainty.py`](src/uncertainty.py) | Bootstrapped confidence intervals + sparse data epistemic penalty ($ECE = 0.0381$). |
| **9. Backtesting Framework** | [`src/backtest.py`](src/backtest.py) | Temporal holdout (Weeks 1–10 train / 11–16 test) + Kaplan-Meier survival estimator. |
| **10. Sensitivity & Stress Testing** | [`src/sensitivity_analysis.py`](src/sensitivity_analysis.py) | Multi-cohort composition stress testing across 4 scenarios (balanced, acute-shock, quiet-struggle, worst-case). |
| **11. Error Analysis & Synthesis** | [`docs/error_analysis.md`](docs/error_analysis.md) | Before-and-after synthesis table, stress test audit, decile reliability table, and lookahead audit. |
| **12. Automated Test Suite** | [`tests/`](tests/) | 11 automated test suites covering edge cases, SHAP explainability, sensitivity, and calibration (100% passing). |
| **13. Stakeholder Dashboard** | [`dashboard/app.py`](dashboard/app.py) | Streamlit UI with policy slider, local SHAP waterfall, reliability curve, and non-deprecated `width="stretch"` layout. |
| **14. Risk Register** | [`docs/risk_register.md`](docs/risk_register.md) | 7 ethical, operational, and algorithmic risks with 3-tier mitigations. |
| **15. Educator User Guide** | [`docs/user_guide.md`](docs/user_guide.md) | Non-technical handbook with "Curiosity, Not Surveillance" conversational playbooks and workflows. |
| **16. Stakeholder Validation** | [`docs/stakeholder_validation.md`](docs/stakeholder_validation.md) | 5 stakeholder personas (Counselor, Parent, Teacher, Mental Health, Privacy) + 4 implemented code changes. |

---

## 4. Data Schema Reference

The system processes weekly student records partitioned across an Information Barrier:

### 4.1 Raw Observable Input Schema (`StudentWeeklyRecord`)
| Family | Field Name | Type | Valid Range | Update Cadence | Educational Interpretation |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Identity** | `student_id` | `str` | `STU_0001`+ | Static | Unique student identifier pseudonymized for FERPA. |
| **Temporal** | `week` | `int` | `1 .. 16` | Weekly | Discrete semester week index. |
| **Attendance** | `days_present` | `int` | `0 .. 5` | Daily / Weekly | In-seat days physically present in classroom. |
| **Attendance** | `days_absent` | `int` | `0 .. 5` | Daily / Weekly | Total absent days (excused + unexcused). |
| **Attendance** | `tardy_count` | `int` | `0 .. 5` | Daily / Weekly | Class punctuality infractions. |
| **Attendance** | `unexcused_absences` | `int` | `0 .. 5` | Daily / Weekly | Unexcused truancies without parent verification. |
| **Activity** | `lms_logins` | `int` | $\ge 0$ | Real-time / Daily | Total authentication sessions logged on LMS. |
| **Activity** | `content_time_minutes` | `float` | $\ge 0.0$ | Daily / Weekly | Active time interacting with course reading materials. |
| **Activity** | `assignment_submissions` | `int` | $\ge 0$ | Weekly | Completed deliverables submitted. |
| **Activity** | `late_submissions` | `int` | $\ge 0$ | Weekly | Deliverables submitted past deadline. |
| **Activity** | `discussion_posts` | `int` | $\ge 0$ | Weekly | Discussion forum questions or peer replies. |
| **Assessment** | `quiz_score` | `float?` | `0.0 .. 100.0` | Weekly | Formative quiz percentage (nullable if no quiz). |
| **Assessment** | `cumulative_score_avg` | `float` | `0.0 .. 100.0` | Weekly | Running semester gradebook weighted average. |
| **Assessment** | `score_trend_slope` | `float` | $(-\infty, +\infty)$ | Weekly | 3-week linear grade trajectory slope (pts/week). |
| **Assessment** | `score_variance` | `float` | $\ge 0.0$ | Weekly | Trailing score variance across recent assessments. |
| **Help-Seeking** | `questions_asked` | `int` | $\ge 0$ | Weekly | In-class or asynchronous questions to instructor. |
| **Help-Seeking** | `office_hours_attended` | `int` | $\ge 0$ | Weekly | 1-on-1 office hours consultations attended. |
| **Help-Seeking** | `tutoring_sessions` | `int` | $\ge 0$ | Weekly | Peer tutoring sessions completed. |
| **Help-Seeking** | `help_seeking_delay_days` | `float` | $\ge 0.0$ | Event-driven | Days between low grade and first help inquiry. |
| **Feedback** | `survey_sentiment` | `float?` | `-1.0 .. +1.0` | Bi-weekly | Student pulse sentiment polarity (nullable). |
| **Feedback** | `teacher_note_flag` | `int` | `0, 1, 2` | Weekly | Faculty concern note: 0=None, 1=Mild, 2=Acute. |
| **Feedback** | `confidence_rating` | `int?` | `1 .. 5` | Bi-weekly | Self-reported academic self-efficacy rating (nullable). |

*Strict Audit Vault (Ground Truth - Never exposed to models)*: `archetype` (`str`), `latent_engagement` (`float`), `is_disengaged` (`bool`).

### 4.2 Engineered Features (`ENGINEERED_FEATURE_NAMES`)
The model uses 21 engineered behavioral features derived over trailing 3-week windows:
- **Attendance (4)**: `att_3wk_mean`, `att_trend_slope`, `unexcused_absences_sum`, `tardy_rate`.
- **Activity (5)**: `lms_logins_3wk_mean`, `content_time_3wk_mean`, `late_submission_ratio`, `logins_per_active_hour` (gaming detection: logins divided by active reading hours), `discussion_posts_3wk_mean`.
- **Assessment (4)**: `quiz_score_recent`, `cumulative_score_avg`, `score_trend_slope`, `score_variance`.
- **Help-Seeking (5)**: `office_hours_3wk_sum`, `tutoring_3wk_sum`, `questions_asked_3wk_mean`, `help_seeking_delay_days`, `is_seeking_help` (binary help engagement flag).
- **Feedback & Sentiment (3)**: `survey_sentiment_recent`, `teacher_concern_flags_3wk`, `confidence_rating_recent`.

---

## 5. Developer Core API Reference

Internal Python module APIs for developers integrating or extending the system:

### 5.1 Feature Extraction Pipeline (`src/features.py`)
```python
from src.features import prepare_feature_matrix

X, y, audit_df = prepare_feature_matrix(df: pd.DataFrame)
```
- **Inputs**: `df` (`pd.DataFrame`) containing raw student weekly records matching `StudentWeeklyRecord`.
- **Outputs**:
  - `X` (`pd.DataFrame`, shape `[N*T, 21]`): Clean, normalized feature matrix with zero NaNs.
  - `y` (`pd.Series`, shape `[N*T]`): Binary ground truth labels (0 = engaged, 1 = disengaged).
  - `audit_df` (`pd.DataFrame`, shape `[N*T, 5]`): Protected audit coordinates (`student_id`, `week`, `archetype`, `latent_engagement`, `is_disengaged`).
- **Exceptions**: Raises `ValueError` if any ground truth column leaks into candidate feature columns.

### 5.2 Calibrated Multi-Signal Model (`src/main_model.py`)
```python
from src.main_model import TransparentMultiSignalModel

model = TransparentMultiSignalModel(random_state: int = 42)
model.fit(X: pd.DataFrame, y: pd.Series) -> TransparentMultiSignalModel
risk_scores = model.predict_risk_score(X: pd.DataFrame) -> np.ndarray  # p_hat in [0.0, 1.0]
flags = model.predict(X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray  # bool array
explanation = model.explain_instance(instance_features: pd.Series, background_X: Optional[pd.DataFrame] = None) -> Dict[str, Any]
memo_text = model.export_counselor_audit_record(student_id: str, instance_features: pd.Series, explanation: Dict[str, Any]) -> str
```
- **Explanation Payload Structure**:
  - `risk_score`: Calibrated probability $\hat{p} \in [0.0, 1.0]$.
  - `family_contributions`: `Dict[str, float]` mapping each of the 5 signal families to its risk impact.
  - `top_risk_drivers`: Top 5 risk-elevating features with plain-language labels and context strings.
  - `top_protective_factors`: Top 5 protective assets mitigating risk.
  - `intervention_pathway`: Triage pathway (`RESTO_WELLNESS_CHECK`, `ACADEMIC_INTERVENTION`, `STANDARD_MONITORING`).

### 5.3 Uncertainty Estimation (`src/uncertainty.py`)
```python
from src.uncertainty import BootstrappedUncertaintyEstimator

unc = BootstrappedUncertaintyEstimator(n_bootstraps: int = 10, random_state: int = 42)
unc.fit(X: pd.DataFrame, y: pd.Series) -> BootstrappedUncertaintyEstimator
mean_r, low_b, up_b, width = unc.predict_with_intervals(X: pd.DataFrame, confidence_level: float = 0.80)
instance_unc = unc.compute_instance_uncertainty(instance_features: pd.Series, weeks_available: int = 16) -> Dict[str, Any]
```
- **Outputs**: Granular confidence interval `[interval_lower, interval_upper]`, `interval_width`, Shannon entropy (aleatoric uncertainty), and automatic transfer-student interval expansion when `weeks_available < 6`.

### 5.4 Benchmark Baseline Model (`src/baseline_model.py`)
```python
from src.baseline_model import LaggingAttendanceMarksBaseline

baseline = LaggingAttendanceMarksBaseline(attendance_threshold: float = 0.80, marks_threshold: float = 60.0)
baseline_risk = baseline.predict_risk_score(df: pd.DataFrame) -> np.ndarray
baseline_flags = baseline.predict(df: pd.DataFrame) -> np.ndarray  # True if att < 80% OR marks < 60%
breakdown = baseline.get_signal_contributions(row: pd.Series) -> Dict[str, Any]
```

### 5.5 Multi-Cohort Sensitivity Engine (`src/sensitivity_analysis.py`)
```python
from src.sensitivity_analysis import evaluate_cohort_scenario, SCENARIO_CONFIGURATIONS

results = evaluate_cohort_scenario(
    scenario_cfg: Dict[str, Any],
    seed: int = 42,
    train_end_week: int = 10,
    test_start_week: int = 11,
    threshold: float = 0.50,
    fixed_model: Optional[TransparentMultiSignalModel] = None,
) -> Dict[str, Any]
```

---

## 6. Before vs. Target vs. Measured Synthesis Table

Evaluated strictly on the temporal holdout test set (Weeks 11–16, $N=3,000$ student-week observations):

| Evaluation Dimension / Metric | Current-Practice Baseline | Review 2 Target Milestone | Measured Result (Main Model) | Error Analysis & Operational Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **F1 Score** | `0.374` | $\ge 0.850$ | **`0.902`** | **+141% relative improvement**. Balances comprehensive counselor sensitivity with student protection against false-alarm stigma. |
| **Counselor Recall (Sensitivity)** | `27.2%` | $\ge 80.0%$ | **`85.9%`** | Baseline misses **72.8%** of struggling student-weeks. Multi-signal fusion catches 6 out of 7 disengaging students. |
| **Student/Parent Precision** | `59.5%` | $\ge 90.0%$ | **`94.9%`** | Eliminates the 40.5% baseline false-positive error rate, preventing unneeded interventions and parental distrust. |
| **False-Alarm Rate (FPR on Negatives)** | `8.7%` | $\le 4.0%$ | **`2.1%`** | **4x lower false-alarm burden** on counselors, preventing alert fatigue and caseload thrashing. |
| **ROC-AUC** | `0.585` | $\ge 0.950$ | **`0.992`** | Demonstrates near-perfect ranking discrimination across all possible decision thresholds $\tau \in [0.10, 0.90]$. |
| **Brier Calibration Score** | `0.231` | $\le 0.080$ | **`0.038`** | **6x better probabilistic calibration**. Probabilities reflect empirical disengagement frequencies ($ECE = 0.0381$). |
| **Kaplan-Meier Lead-Time Advantage** | Week 15.18 / 16 | $\ge 30\text{ Days Earlier}$ | **Week 8.95 / 9** | **43.6 to 49.0 days (6.2 to 7.0 weeks) earlier detection**, providing counselors a 1-month intervention runway before midterms. |
| **Right-Censored False Negatives** | **32.8% (58 students)** | $\le 5.0%$ | **0.0% (0 students)** | **Completely eliminates the baseline's 1/3 blind spot**. Every disengaged student is caught before the semester ends. |
| **Quietly Struggling Recall** | `8.3%` | $\ge 75.0%$ | **`85.6%`** | Attendance baseline fails because students attend class faithfully ($5/5$ days). Model detects drop in active reading & morale. |
| **Signal Gamer Recall** | `41.7%` (ablation) | $\ge 80.0%$ | **`85.4%`** | Single-metric models are fooled by login frequency. Caught via `logins_per_active_hour` ratio divergence ($> 100$). |
| **False Alarms on Recovering Students** | `> 40.0%` | $\le 5.0%$ | **`0.0%`** | Baseline penalizes historical low GPA. Model recognizes positive 3-week velocity ($> +2.5\%$/wk) and tutoring attendance. |
| **Sparse History Data Protection** | `0.0%` (No guardrail) | 100% Policy Protection | **100% Guardrail** | Transfer students with $< 4$ weeks automatically trigger `MONITOR_ONLY_SPARSE_DATA` governance shield and wide intervals. |
| **Explanatory Resolution** | Binary Rule ("Marks low") | Plain-Language Attribution | **Plain-Language SHAP** | Translates 21 technical features into plain-English educator cards with actionable consultation memoranda. |

---

## 7. Sensitivity & Stress Testing Across Cohort Compositions

Benchmarked across 4 distinct synthetic cohort environments ($N=500$ students each, evaluated on temporal holdout test set):

| Cohort Scenario | Model | F1 Score | Recall (Sens.) | Precision | False-Alarm Rate (FPR) | Brier Calibration | ROC-AUC | Relative F1 Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Reference (Balanced)** | **Main 5-Signal** | **`0.902`** | **`85.8%`** | **`95.2%`** | **`2.1%`** | **`0.058`** | **`0.992`** | **+141.4%** |
| *(Base Rate: 32.1%)* | Lagging Baseline | 0.374 | 27.2% | 59.6% | 8.7% | 0.304 | 0.586 | Benchmark |
| **High Acute-Shock (Crisis Surge)** | **Main 5-Signal** | **`0.926`** | **`89.4%`** | **`96.0%`** | **`1.4%`** | **`0.042`** | **`0.996`** | **+140.5%** |
| *(Base Rate: 26.7%)* | Lagging Baseline | 0.385 | 29.4% | 55.8% | 8.5% | 0.250 | 0.599 | Benchmark |
| **High Chronic Quiet-Struggle** | **Main 5-Signal** | **`0.942`** | **`92.1%`** | **`96.3%`** | **`3.8%`** | **`0.059`** | **`0.992`** | **+205.6%** |
| *(Base Rate: 52.3%)* | Lagging Baseline | 0.308 | 19.4% | 74.0% | 7.5% | 0.500 | 0.556 | Benchmark |
| **Mixed Worst-Case (Stressed)** | **Main 5-Signal** | **`0.928`** | **`89.9%`** | **`96.0%`** | **`3.9%`** | **`0.052`** | **`0.989`** | **+111.8%** |
| *(Base Rate: 50.6%)* | Lagging Baseline | 0.438 | 30.3% | 78.8% | 8.4% | 0.467 | 0.604 | Benchmark |

* **Key Takeaway**: Under extreme Quiet-Struggle skew (44% of students), the baseline recall collapses to **19.4%**, while the Main Model sustains **92.1% recall** (**+205.6% relative F1 gain**).

---

## 8. Quickstart Guide

### Windows (PowerShell)
```powershell
# 1. Setup dependencies
.\run.ps1 setup

# 2. Generate synthetic cohort data
.\run.ps1 data

# 3. Run comprehensive automated test suite (21 tests)
.\run.ps1 test

# 4. Run sensitivity & stress testing across cohort compositions
.\run.ps1 stress

# 5. Run multi-year longitudinal drift analysis
.\run.ps1 drift

# 6. Execute temporal holdout backtesting
.\run.ps1 backtest

# 7. Launch interactive stakeholder dashboard
.\run.ps1 run
```

### Linux / macOS (Makefile)
```bash
make setup
make data
make test
make stress
make drift
make backtest
make run
```
