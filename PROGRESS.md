# Project Progress Log: Transparent Disengagement Early-Warning System

Living status and milestone tracking for the Transparent Disengagement Early-Warning System. Updated at every phase, milestone, and formal evaluation checkpoint.

---

## Overall Status Summary

| Phase | Description | Status | Target Completion |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Foundations (Docs + Synthetic Data + Baseline Model)** | **COMPLETE (100%)** | Done |
| **Phase 2** | **Core MVP (~35% Graded Review 1 Checkpoint — Scored 100%)** | **COMPLETE (100%)** | Done |
| **Phase 3** | **Review 2 Improvements (~70% Graded Checkpoint)** | **COMPLETE (100%)** | Done |
| **Phase 4** | **Pilot Deployment, LTI Integrations & Final Submission** | **IN PROGRESS (~70% Overall)** | Next Milestone |

---

## Detailed Deliverable Checklist

- [x] **Deliverable 1: Stakeholder Assumptions Document** (`docs/stakeholder_assumptions.md`)
  - Explicitly defines Counselor (Recall/Sensitivity) vs. Student/Parent (Precision/Privacy) vs. Teacher (Actionability).
  - Multi-dimensional operational definition of disengagement (behavioral, cognitive, affective).
  - Consent models, ethical bounds, FERPA/GDPR alignment, and human-in-the-loop mandate.

- [x] **Deliverable 2: Architecture Diagram** (`docs/architecture.md`)
  - Complete Mermaid pipeline diagram checked into repo.
  - Strict Information Barrier wall quarantining ground truth (`archetype`, `latent_engagement`, `is_disengaged`).
  - Fixed temporal holdout partition: Weeks 1–10 train / Weeks 11–16 test.

- [x] **Deliverable 3: Data Schema & Pydantic Definitions** (`docs/data_schema.md`, `src/schema.py`)
  - 5 signal families (Attendance, Activity, Assessment, Help-Seeking, Feedback/Sentiment).
  - Exact population breakdown documented: $N=500$ students (250 engaged, 100 quietly struggling, 75 checked out, 50 improving, 10 transfer, 8 gamers, 7 acute shock).
  - Parametrized $\mu_{\text{archetype}}(t)$ drift functions logged.
  - Noisy, lagged observation equations for pulse sentiment and teacher notes.

- [x] **Deliverable 4: Synthetic Data Generator** (`data/synthetic_data_generator.py`)
  - Dynamic state-space latent engagement trajectory $E_{i,t} \in [0.0, 1.0]$.
  - Generated $7,940$ student-week observation rows in `data/synthetic_cohort.csv`.
  - Non-trivial ground-truth label: requires $E_{i,t} < 0.40$ for $\ge 2$ consecutive weeks.
  - **Review 2 Enhancement**: Supports parameterized `archetype_counts` for arbitrary cohort stress testing.

- [x] **Deliverable 5: Naive Lagging Baseline Model** (`src/baseline_model.py`)
  - Current-practice threshold rule: Attendance $< 80\%$ OR Cumulative Marks $< 60\%$.
  - Clearly documented as the punitive, lagging status quo benchmark.

- [x] **Deliverable 6: Main Multi-Signal Model & Local Explainability** (`src/main_model.py`, `src/features.py`)
  - 5-signal fusion with 3-week rolling features and `logins_per_active_hour` gaming ratio.
  - Calibrated probability output $\hat{p}_{i,t} \in [0.0, 1.0]$ via Platt/Sigmoid calibration.
  - **Review 2 Enhancement**: TreeSHAP / Explainer local feature attributions with educator-friendly plain-language terminology (`FEATURE_PLAIN_LANGUAGE_MAPPING`).
  - **Review 2 Enhancement**: Plain-English contextual interpretations, top risk drivers, and protective student strengths.
  - **Review 2 Enhancement**: Compassionate Restorative Wellness Triage (`RESTO_WELLNESS_CHECK`).
  - **Review 2 Enhancement**: FERPA-compliant Consultation Memorandum generator (`export_counselor_audit_record`).

- [x] **Deliverable 7: Uncertainty Estimation & Calibration Engine** (`src/uncertainty.py`)
  - Bootstrapped ensemble (8 models) for epistemic uncertainty.
  - Shannon entropy for aleatoric uncertainty.
  - Additive epistemic margin penalty for sparse historical data (transfer students).
  - **Review 2 Enhancement**: Probabilistic calibration verification ($ECE = 0.0381$, Brier $= 0.038$, $84.7\%$ interval coverage).

- [x] **Deliverable 15: Granular Testing & Error Handling Guide** (`docs/testing_and_error_handling.md`)
  - Detailed technical documentation on all 11 automated unit test suites, failure modes, and error boundaries.
  - Documents missing/malformed input policies, zero-division safeguards, and defensive schema fallbacks.

- [x] **Deliverable 16: Code Comments & Developer Schema / API Reference** (`README.md`, `src/`)
  - Expanded docstrings, mathematical formulations, and edge-case comments across all core `src/` modules.
  - Direct 5-signal raw input schema table and 21 engineered feature definitions embedded in `README.md`.
  - Comprehensive Developer Core API Reference documenting function signatures, inputs, outputs, exceptions, and usage examples.

- [x] **Deliverable 17: Production LMS & SIS Integration Connectors** (`src/connectors/`, `tests/test_connectors.py`)
  - Modular integration adapters for standard school feeds: OneRoster v1.2 REST API, Canvas LMS REST API, and PowerSchool SIS REST API.
  - Defined `BaseConnector` abstract contract with OAuth2/API key authentication, health checks, rate-limiting, and exponential backoff retry.
  - Implemented `UnifiedIngestionPipeline` that orchestrates parallel extraction, merges multi-source telemetry on `(student_id, week)`, enforces the Information Barrier, and validates against Pydantic schema.
  - Added 4 automated unit tests verifying authentication, data extraction, mapping fidelity, and downstream ML compatibility (15/15 tests passing).

- [x] **Deliverable 18: Student Self-Advocacy Reflection Portal** (`dashboard/app.py`, `tests/test_student_portal.py`)
  - Built a separate, privacy-conscious role view in the dashboard (`render_student_portal`) for students themselves.
  - Strengths-based architecture: showcases positive learning assets (in-seat attendance consistency, active study reading time, classroom discourse, proactive help-seeking).
  - Strict privacy shield: **Zero administrative risk scores**, **zero disengagement flags**, and **zero deficit labels** exposed to students.
  - Interactive self-reflection pulse with confidence rating, study focus areas, study habits trend chart (with `width="stretch"` layout), and direct 1-click booking links for teacher office hours and peer tutoring.
  - Added automated unit tests confirming strict privacy boundaries and zero deficit exposure (17/17 tests passing).

- [x] **Deliverable 19: Multi-Year Longitudinal Drift Simulator & Production MLOps Monitoring** (`src/drift_monitor.py`, `docs/error_analysis.md`, `tests/test_drift_monitor.py`)
  - Extended synthetic cohort framework to model 3 consecutive academic years (Years 1–3), simulating curriculum rigor shifts and Grade 9 freshman transition shock.
  - Computed formal Population Stability Index (PSI) and 2-sample Kolmogorov-Smirnov statistics across all 21 engineered behavioral features.
  - Tracked performance degradation of Unretrained Legacy Model vs. Annual Retrained Model: demonstrated that annual retraining preserves sharp probability calibration (Brier recovers from 0.072 to 0.043) and maintains $97.2\%$ recall.
  - Authored Section 8 in `docs/error_analysis.md` outlining the 4-tier district MLOps retraining and human-in-the-loop review protocol.
  - Added 4 automated unit tests for PSI, KS stats, and simulation runs (21/21 tests passing).

- [x] **Deliverable 8: Backtesting Framework** (`src/backtest.py`)
  - Temporal holdout split: Weeks 1–10 train / Weeks 11–16 test.
  - Kaplan-Meier Survival Analysis: **KM Median Detection Week 8.95 / 9 (Main) vs Week 15.18 / 16 (Baseline) — 43.6 to 49.0 days (6.2 to 7.0 weeks earlier)** across full cohort.
  - Censoring Audit: Baseline suffers **32.8% right-censoring** (never flags 58 of 177 disengaged students).
  - Lookahead Leakage Audit: Verified all rolling features strictly trailing ($week \le t$) with default `center=False`.
  - Verified F1 Improvement: **Baseline F1 = 0.374 vs Main Model F1 = 0.902** (ROC-AUC 0.585 vs 0.992).

- [x] **Deliverable 9: Sensitivity & Stress Testing Engine** (`src/sensitivity_analysis.py`) — **NEW IN REVIEW 2**
  - Tests 4 distinct cohort distributions ($N=500$ each, $N=3,000$ holdout test observations):
    1. Standard Reference (Balanced)
    2. High Acute-Shock (20.0% acute distress)
    3. High Chronic Quiet-Struggle (44.0% quiet struggle)
    4. Mixed Worst-Case (Stressed/Under-resourced, 50.6% base disengagement rate)
  - Reports shifts in F1, Recall, Precision, False-Alarm Rate (FPR), Brier score, and ROC-AUC.
  - Audits honest degradation under cohort skew (e.g. FPR increase from 2.1% to 3.8% on transitional weeks).

- [x] **Deliverable 10: Stakeholder Trade-Off Dashboard** (`dashboard/app.py`) — **ENHANCED IN REVIEW 2**
  - Interactive Streamlit UI with decision threshold slider $\tau \in [0.10, 0.90]$.
  - Maintenance cleanup: Replaced deprecated `use_container_width=True` with `width="stretch"` across all charts and tables.
  - Per-student local explainability: Plain-language summary, local SHAP waterfall chart, 5-family attribution, and longitudinal trend history.
  - Reliability Diagram: Interactive Plotly calibration curve with decile bins, ECE metric, and perfect calibration reference line.
  - FERPA Consultation Memorandum preview card.

- [x] **Deliverable 11: Error Analysis, Ablation & Synthesis** (`docs/error_analysis.md`) — **ENHANCED IN REVIEW 2**
  - Archetype failure mode audits and assessment-only ablation study.
  - **Review 2 Enhancement**: Before-and-After Synthesis Table (Baseline vs Target vs Measured vs Operational Rationale).
  - **Review 2 Enhancement**: Full Sensitivity & Stress Testing Report with honest degradation audit.
  - **Review 2 Enhancement**: Uncertainty Calibration Decile Table ($N=3,000$ holdout test samples).

- [x] **Deliverable 12: Educator & Counselor User Guide** (`docs/user_guide.md`) — **REWRITTEN IN REVIEW 2**
  - Completely rewritten for a non-technical educator audience.
  - Empathetic tone, "Curiosity, Not Surveillance" operating rule, and step-by-step weekly counselor workflows.
  - Conversational Playbook with exact word-for-word scripts for Quietly Struggling, Improving, Gamer, Transfer, and Crisis students.
  - Teacher checklists and non-technical FAQs.

- [x] **Deliverable 13: Stakeholder Validation & Algorithmic Iteration** (`docs/stakeholder_validation.md`) — **EXPANDED IN REVIEW 2**
  - 5 representative stakeholder personas:
    1. Dr. Elena Rostova (Academic Counselor)
    2. Marcus Chen (Student/Parent Advocate)
    3. Sarah Jenkins (Classroom Teacher)
    4. **David Vance, LCSW (School Social Worker & District Mental Health Coordinator)** — *New*
    5. **Rachel Torres, CISSP (District IT & Student Privacy Compliance Officer)** — *New*
  - 4 concrete implemented code changes directly resulting from feedback:
    1. Sparse History Data Guardrail
    2. Recovery Velocity Discount
    3. Compassionate Restorative Wellness Triage Safeguard
    4. FERPA-Compliant Consultation Memorandum Generator

- [x] **Deliverable 14: Comprehensive Automated Test Suite** (`tests/`) — **EXPANDED IN REVIEW 2**
  - `tests/test_edge_cases.py` (4 edge-case & barrier tests).
  - `tests/test_review2_features.py` (7 tests: plain-language mapping, SHAP explainability, restorative triage, FERPA memorandum, sensitivity scenarios, `use_container_width` cleanup, calibration computation).
  - **11/11 automated tests passing via pytest in 20.08s!**

- [x] **Deliverable 15: Technical Testing & Error Handling Specification** (`docs/testing_and_error_handling.md`) — **NEW IN PHASE 4**
  - Granular documentation of all 11 unit & edge-case tests, failure modes, and system defenses.
  - Detailed malformed/missing input policies (forward-fill with uninformative priors, clamped ratios, NaN safety).
  - Explicit tiered error boundary hierarchy (Data, Model, Uncertainty, Policy guardrails).

---

## Review 2 Milestone Checkpoint Review (~70% Target Reached)

### 1. What You Have Completed Since Review 1
1. **Per-Student Local Plain-Language Explainability**:
   - Replaced opaque machine learning column names (`logins_per_active_hour`, `content_time_3wk_mean`, `survey_sentiment_recent`) with 21 certified educator definitions in `FEATURE_PLAIN_LANGUAGE_MAPPING`.
   - Integrated TreeSHAP / Explainer in `TransparentMultiSignalModel.explain_instance()` to calculate exact additive contributions to risk probability.
   - Built a 5-tab student inspection suite in `dashboard/app.py` featuring a plain-English summary, a local SHAP waterfall chart, 5-family domain attribution, longitudinal trajectory graphs, and FERPA memo export.
2. **Sensitivity & Stress Testing Across Cohort Skews**:
   - Implemented `src/sensitivity_analysis.py` benchmarking both models across 4 realistic compositions:
     - Standard Reference (Balanced, 32.1% disengaged)
     - High Acute-Shock (Crisis Surge, 20% acute shock, 14x baseline rate)
     - High Chronic Quiet-Struggle (STEM/Magnet school, 44% quiet struggle)
     - Mixed Worst-Case (Stressed/Under-resourced, 50.6% disengagement rate)
   - Audited honest degradation modes: documented that Main Model F1 remains $\ge 0.902$ across all skews (vs. Baseline collapsing to $0.308$), while honestly reporting an increase in transitional false alarms (from 2.1% to 3.8%) on borderline early-disengagement weeks.
3. **Maintenance Cleanup & Educator User Guide**:
   - Eliminated all deprecated Streamlit `use_container_width=True` calls, updating them to `width="stretch"`.
   - Rewrote `docs/user_guide.md` from the ground up for non-technical educators, complete with an empathetic "Curiosity, Not Surveillance" protocol and conversational scripts.
4. **Uncertainty Calibration & Synthesis**:
   - Implemented an interactive Probabilistic Calibration / Reliability Diagram in `dashboard/app.py` with 10 confidence decile bins, ECE ($0.0381$), and Brier score ($0.038$).
   - Authored the Before-and-After Synthesis Table in `docs/error_analysis.md` comparing baseline vs target vs measured metrics across 13 dimensions.
5. **Stakeholder Validation Expansion & Code Changes**:
   - Expanded `docs/stakeholder_validation.md` with David Vance, LCSW (Mental Health Coordinator) and Rachel Torres, CISSP (Privacy Officer).
   - Implemented two resulting code changes: **Compassionate Restorative Triage** (`RESTO_WELLNESS_CHECK`) and the **FERPA Consultation Memorandum Generator**.
6. **Automated Verification**:
   - Added `tests/test_review2_features.py`; confirmed 11/11 tests passing in 20.08s.

---

### 2. Key Features and Modules Currently Working End-to-End
* **`src/main_model.py`**: Calibrated multi-signal model with TreeSHAP local attributions, plain-language translations, restorative triage routing, and FERPA memo export.
* **`dashboard/app.py`**: Streamlit application with policy threshold slider, 5-tab student profile inspector, local SHAP waterfall charts, and reliability curve visualization.
* **`src/sensitivity_analysis.py`**: Automated cohort composition stress tester; saves full JSON benchmarks to `data/sensitivity_results.json`.
* **`tests/test_review2_features.py` & `tests/test_edge_cases.py`**: 11 automated pytest suites passing 100%.

---

### 3. Empirical Verification Summary (Real Execution Results)

#### Head-to-Head Performance (Holdout Test Set: Weeks 11–16, $N=3,000$):
* **F1 Score**: Baseline `0.374` vs. Main Model **`0.902`** (**+141% gain**)
* **Counselor Recall**: Baseline `27.2%` vs. Main Model **`85.9%`** (**+58.7% absolute gain**)
* **Student/Parent Precision**: Baseline `59.5%` vs. Main Model **`94.9%`** (**avoids false-alarm stigma**)
* **False-Alarm Rate (FPR)**: Baseline `8.7%` vs. Main Model **`2.1%`** (**4x lower counselor alert fatigue**)
* **ROC-AUC**: Baseline `0.585` vs. Main Model **`0.992`**
* **Probabilistic Calibration**: Expected Calibration Error **`ECE = 0.0381`** | Brier Score **`0.038`**
* **Kaplan-Meier Lead-Time**: **43.6 to 49.0 Days Earlier Detection** (Week 8.95 / 9 vs Week 15.18 / 16)
* **Right-Censored False Negatives**: Baseline **32.8% (58 students)** vs. Main Model **0.0% (0 students)**

#### Sensitivity Stress Testing Summary:
* **High Acute-Shock (20% Shock)**: Main Model F1 **`0.926`** (Recall 89.4%, Precision 96.0%, False Alarm Rate 0.0% on recovered students) vs. Baseline F1 `0.385` (Precision degrades to 55.8%).
* **High Quiet-Struggle (44% Quiet Struggle)**: Main Model F1 **`0.942`** (Recall 92.1%, Quietly Struggling Recall 93.3%) vs. Baseline F1 `0.308` (Recall collapses to 19.4%).
* **Mixed Worst-Case (50.6% Base Rate)**: Main Model F1 **`0.928`** vs. Baseline F1 `0.438`.

---

### 4. What is Pending for Final Review (~30% Remaining)
1. **LTI / LMS Connector Prototype**: Standards-compliant LTI 1.3 tool definition for direct embedding in Canvas, Schoology, or Google Classroom.
2. **Student Self-Advocacy Reflection Portal**: A private student-facing interface providing strengths-based feedback and self-paced study habit suggestions without exposing administrative risk scores.
3. **Multi-Semester Longitudinal Drift Simulator**: Synthetic data extension modeling multi-year grade-level transitions (e.g. 9th to 10th grade transition shock).
4. **Final Production Polish**: Comprehensive Docker packaging, deployment runbooks, and final project presentation slide deck.
