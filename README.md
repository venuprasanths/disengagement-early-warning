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

## 3. Project Deliverables Directory

| Deliverable | Location | Description |
| :--- | :--- | :--- |
| **Stakeholder Assumptions** | [`docs/stakeholder_assumptions.md`](docs/stakeholder_assumptions.md) | Formulates Counselor Recall vs Student Precision conflict and ethical bounds. |
| **Architecture Diagram** | [`docs/architecture.md`](docs/architecture.md) | Mermaid pipeline diagram, data flow, and temporal holdout specification. |
| **Data Schema & Formulas** | [`docs/data_schema.md`](docs/data_schema.md) | 5 signal families schema, exact cohort sizes ($N=500$), parametrized drift functions $\mu(t)$. |
| **Synthetic Data Generator** | [`data/synthetic_data_generator.py`](data/synthetic_data_generator.py) | Dynamic state-space latent engagement generator ($N=500$, $T=16$). |
| **Lagging Baseline Model** | [`src/baseline_model.py`](src/baseline_model.py) | Current-practice attendance+marks threshold benchmark. |
| **Feature Engineering** | [`src/features.py`](src/features.py) | Rolling 3-week aggregations, score trend slopes, gaming divergence ratios. |
| **Main Multi-Signal Model** | [`src/main_model.py`](src/main_model.py) | Calibrated, interpretable model with SHAP local attribution. |
| **Uncertainty Estimator** | [`src/uncertainty.py`](src/uncertainty.py) | Bootstrapped confidence intervals quantifying prediction confidence. |
| **Backtesting Framework** | [`src/backtest.py`](src/backtest.py) | Temporal holdout (Weeks 1–10 train / 11–16 test) evaluating lead-time advantage. |
| **Error Analysis** | [`src/error_analysis.py`](src/error_analysis.py) | Systematic failure mode breakdown across archetypes. |
| **Edge Case Tests** | [`tests/test_edge_cases.py`](tests/test_edge_cases.py) | Transfer students, signal gamers, and acute temporary shocks. |
| **Stakeholder Dashboard** | [`dashboard/app.py`](dashboard/app.py) | Streamlit interactive trade-off UI with dynamic threshold slider. |
| **Risk Register** | [`docs/risk_register.md`](docs/risk_register.md) | Ethical, operational, and algorithmic risk mitigation matrix. |
| **User Guide** | [`docs/user_guide.md`](docs/user_guide.md) | Day-to-day workflow instructions for educators and intervention teams. |
| **Stakeholder Validation** | [`docs/stakeholder_validation.md`](docs/stakeholder_validation.md) | Simulated counselor, parent, teacher feedback and resulting code change. |
| **Milestone Changelog** | [`PROGRESS.md`](PROGRESS.md) | Living progress log and Phase 2 checkpoint review. |

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
