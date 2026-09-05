# Transparent Disengagement Early-Warning System for Schools

An uncertainty-aware, interpretable early warning prototype for schools supporting diverse learning paces. The system fuses five weak signal families into a calibrated risk score, quantifies epistemic and aleatoric uncertainty, and exposes the trade-off between conflicting stakeholder objectives (Academic Counselor Recall vs. Student/Parent Precision).

---

## 1. Problem & Core Distinction

Schools conventionally rely on **attendance thresholds** (e.g., `< 80%`) and **failing test scores** (e.g., `< 60%`) to trigger academic interventions. These are **lagging, punitive indicators**:
- By the time attendance or marks crater, disengagement is already deeply entrenched.
- Remediation is delayed, costly, and demoralizing.
- Students silently disengage while technically "passing" on paper.

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
- **Human-in-the-Loop**: Risk scores are resource-allocation recommendations for counselors, never automated penalties or permanent disciplinary marks.

---

## 3. Project Deliverables Directory (All 14 Deliverables)

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **1. Stakeholder Assumptions** | [`docs/stakeholder_assumptions.md`](docs/stakeholder_assumptions.md) | Formulates Counselor Recall vs Student Precision conflict and ethical bounds. |
| **2. Architecture Diagram** | [`docs/architecture.md`](docs/architecture.md) | Mermaid pipeline diagram, data flow, and temporal holdout specification. |
| **3. Data Schema & Formulas** | [`docs/data_schema.md`](docs/data_schema.md) | 5 signal families schema, cohort sizes ($N=500$), parametrized drift functions $\mu(t)$. |
| **4. Synthetic Data Generator** | [`data/synthetic_data_generator.py`](data/synthetic_data_generator.py) | Dynamic state-space latent engagement generator ($N=500$, $T=16$, $7,940$ rows). |
| **5. Lagging Baseline Model** | [`src/baseline_model.py`](src/baseline_model.py) | Current-practice attendance+marks threshold benchmark. |
| **6. Feature Engineering** | [`src/features.py`](src/features.py) | Rolling 3-week trailing features, score slopes, and gaming divergence ratios. |
| **7. Main Multi-Signal Model** | [`src/main_model.py`](src/main_model.py) | Calibrated, interpretable model with 5-domain explanation breakdown. |
| **8. Uncertainty Estimator** | [`src/uncertainty.py`](src/uncertainty.py) | Bootstrapped confidence intervals + sparse data epistemic penalty. |
| **9. Backtesting Framework** | [`src/backtest.py`](src/backtest.py) | Temporal holdout (Weeks 1–10 train / 11–16 test) + Kaplan-Meier survival estimator. |
| **10. Error Analysis & Audit** | [`src/error_analysis.py`](src/error_analysis.py), [`docs/error_analysis.md`](docs/error_analysis.md) | Failure mode audits, ablation study, censoring analysis, and lookahead audit. |
| **11. Edge Case Tests** | [`tests/test_edge_cases.py`](tests/test_edge_cases.py) | Transfer students, signal gamers, acute temporary shocks, and information barrier. |
| **12. Stakeholder Dashboard** | [`dashboard/app.py`](dashboard/app.py) | Streamlit interactive trade-off UI with dynamic threshold slider. |
| **13. Risk Register** | [`docs/risk_register.md`](docs/risk_register.md) | 7 ethical, operational, and algorithmic risks with 3-tier mitigations. |
| **14. User Guide & Validation** | [`docs/user_guide.md`](docs/user_guide.md), [`docs/stakeholder_validation.md`](docs/stakeholder_validation.md) | Day-to-day workflow, ethical protocols, 3 stakeholder audits, and implemented changes. |

---

## 4. Head-to-Head Evaluation: Before vs. After

Evaluated strictly on the temporal holdout test set (Weeks 11–16, $N=3,000$ student-week observations):

| Evaluation Metric | Current-Practice Baseline | Transparent Multi-Signal Model | Improvement / Benefit |
| :--- | :---: | :---: | :---: |
| **F1 Score** | `0.374` | **`0.902`** | **+0.528 (+141% relative)** |
| **Recall (Counselor Sensitivity)** | `27.2%` | **`85.9%`** | **+58.7% absolute gain** |
| **Precision (Student Protection)** | `59.5%` | **`94.9%`** | **+35.4% (avoids false-alarm stigma)** |
| **ROC-AUC** | `0.585` | **`0.992`** | **+0.407** |
| **Brier Calibration Score** | `0.231` | **`0.038`** | **6x better probabilistic calibration** |
| **Kaplan-Meier Median Detection** | Week 15.18 / 16 | **Week 8.95 / 9** | **43.6 to 49.0 days earlier detection** |
| **Right-Censored (Never Detected)**| **32.8% (58 students)** | **0.0% (0 students)** | **Eliminated the 1/3 baseline blindspot** |
| **Quietly Struggling Recall** | `8.3%` | **`85.6%`** | Catches students who attend but disengage |
| **Signal Gamer Recall** | `41.7%` (ablation) | **`85.4%`** | Detected via activity-depth divergence |
| **False Alarms on Recovering Students** | `> 40%` | **`0.0%`** | Upward velocity discounts past low marks |

---

## 4. Quickstart Guide

### Windows (PowerShell)
```powershell
# 1. Setup dependencies
.\run.ps1 setup

# 2. Generate synthetic cohort data
.\run.ps1 data

# 3. Run edge case tests
.\run.ps1 test

# 4. Execute temporal backtesting
.\run.ps1 backtest

# 5. Launch interactive stakeholder dashboard
.\run.ps1 run
```

### Linux / macOS (Makefile)
```bash
make setup
make data
make test
make backtest
make run
```
