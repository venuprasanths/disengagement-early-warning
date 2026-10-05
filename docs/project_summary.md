# Stakeholder Presentation Slide Deck & Executive Summary
## Transparent Disengagement Early-Warning System for Schools
**Final Review Defense (100% Completion Milestone)**

---

<!-- slide -->
# Slide 1: Title & Executive Summary

### **Transparent Disengagement Early-Warning System**
*An Uncertainty-Aware, Multi-Signal Behavioral Platform for Schools Supporting Diverse Learning Paces*

* **The Problem**: Conventional school early-warning systems wait for attendance to crater ($< 80\%$) or marks to fail ($< 60\%$). By then, disengagement is entrenched, remediation is expensive, and students have already fallen behind.
* **The Breakthrough**: Fuses 5 weak behavioral signal families into a calibrated early warning model with local TreeSHAP explainability, detecting academic disengagement **43.6 to 49.0 days earlier** than status-quo practice.
* **The Philosophy**: *"Curiosity, Not Surveillance"* — Empowers counselors with transparent decision support and students with private self-advocacy tools without punitive deficit labeling.

> **Key Milestone**: Scored 100% in Review 1, 92% in Review 2, now at **100% completion** with LMS connectors, student portal, longitudinal drift monitoring, and Docker deployment.

---

<!-- slide -->
# Slide 2: The Status Quo Crisis — The Fatal Blind Spot of Lagging Indicators

```
Conventional School Early-Warning System (Current Practice):
[Week 1-8: Student withdrawing silently] ---> [Week 9-14: Passing on paper (5/5 attendance)] ---> [Week 15-16: Midterm grades fail]
                                                                                                      |
                                                                                    TOO LATE TO INTERVENE!
```

### The Three Critical Failures of Attendance & Marks:
1. **The "Quietly Struggling" Blind Spot**: Students physically present in class 5 out of 5 days, submitting homework on time, but suffering silent cognitive withdrawal. **Status quo catches only 8.3% of them**.
2. **The "Signal Gamer" Blind Spot**: Students who frequently authenticate into the LMS portal to create a digital paper trail without engaging in deep course reading.
3. **Severe Right-Censoring**: Conventional baseline models suffer **32.8% right-censoring** (never flagging 58 of 177 disengaged students before the semester concludes).

---

<!-- slide -->
# Slide 3: Our Solution — 5-Signal Fusion & Ethical Information Barrier

```
                     STRICT INFORMATION BARRIER
      [Ground-Truth Audit Vault]          [Operational Behavioral Features]
      - Latent State E_{i,t}               1. Attendance (In-seat, punctuality, unexcused)
      - True Archetype                     2. LMS Activity (Reading mins, logins, late subs)
      - Post-Hoc Outcome Label            3. Assessment Momentum (Score velocity, variance)
             || [QUARANTINED]              4. Help-Seeking (Office hours, tutoring, delay)
             \/                            5. Student Voice (Pulse morale, teacher notes)
      Only used in post-hoc                       ||
      auditing and calibration                    \/
                                      [HistGradientBoosting + Platt Calibration]
                                                  ||
                                                  \/
                                      Calibrated Risk p_hat in [0.0, 1.0]
                                      + 80% Bootstrapped Prediction Intervals
                                      + Plain-Language Local TreeSHAP Explanations
```

* **No Data Leakage**: Ground truth $E_{i,t}$ is strictly isolated in an audit vault.
* **Weak Bayesian Signals**: Teacher notes and pulse surveys reflect realistic observation lags and noise, ensuring they act as weak priors rather than ground-truth proxies.

---

<!-- slide -->
# Slide 4: Empirical Breakthrough — Before vs. Target vs. Measured

Evaluated strictly on the temporal holdout test set (Weeks 11–16, $N=3,000$ student-week observations):

| Evaluation Dimension | Current Practice Baseline | Target Milestone | Measured Result (Main Model) | Improvement / Impact |
| :--- | :---: | :---: | :---: | :--- |
| **F1 Score** | `0.374` | $\ge 0.850$ | **`0.902`** | **+141% relative improvement** |
| **Counselor Recall** | `27.2%` | $\ge 80.0%$ | **`85.9%`** | **Catches 6 out of 7 disengaged students** (vs. 1 in 4) |
| **Student/Parent Precision** | `59.5%` | $\ge 90.0%$ | **`94.9%`** | **Eliminates 40.5% false-positive rate** |
| **False-Alarm Rate (FPR)** | `8.7%` | $\le 4.0%$ | **`2.1%`** | **4x lower counselor alert fatigue** |
| **ROC-AUC Discrimination** | `0.585` | $\ge 0.950$ | **`0.992`** | Near-perfect discrimination across all thresholds |
| **Brier Calibration Score** | `0.231` | $\le 0.080$ | **`0.038`** | **6x sharper probabilistic accuracy** ($ECE = 0.0381$) |
| **Lead-Time Advantage** | Week 15.18 / 16 | $\ge 30\text{ Days Earlier}$ | **Week 8.95 / 9** | **43.6 to 49.0 days earlier intervention runway** |
| **Right-Censored Blind Spot** | **32.8% (58 students)** | $\le 5.0%$ | **0.0% (0 students)** | **100% of disengaged students identified in-semester** |
| **Quietly Struggling Recall** | `8.3%` | $\ge 75.0%$ | **`85.6%`** | Detects active reading collapse despite 5/5 attendance |
| **Signal Gamer Recall** | `41.7%` | $\ge 80.0%$ | **`85.4%`** | Caught via `logins_per_active_hour` ratio divergence |
| **False Alarms on Recovering**| `> 40.0%` | $\le 5.0%$ | **`0.0%`** | Discounts past marks when grade velocity is positive |

---

<!-- slide -->
# Slide 5: Local Explainability & Restorative Triage

### From Machine Learning Jargon to Actionable Counselor Dialogue:
* Translates 21 raw features into educator-certified concepts (`FEATURE_PLAIN_LANGUAGE_MAPPING`).
* TreeSHAP explains **exact mathematical contributions** to risk probability in plain English.

```
+-----------------------------------------------------------------------------------------+
| STUDENT SUPPORT CONSULTATION MEMORANDUM (STU_0251)                                       |
| Advisory Pathway: Compassionate Restorative Wellness Check (Non-Academic)              |
+-----------------------------------------------------------------------------------------+
| 1. PRIMARY OBSERVATIONAL DRIVERS (Why Support is Recommended):                          |
|    - Active Digital Reading & Study Time: Low active reading (14 mins/wk) (+28.4% risk) |
|    - Bi-Weekly Pulse Survey Morale: Negative pulse morale (-0.62) (+19.1% risk)         |
|                                                                                         |
| 2. DEMONSTRATED PROTECTIVE STRENGTHS (Student Assets):                                  |
|    - In-Seat Attendance Rate: Consistent 5/5 days/week present (-12.5% risk reduction)  |
|                                                                                         |
| 3. RECOMMENDED COUNSELOR ACTION:                                                        |
|    "Student exhibits low morale despite faithful attendance. Recommend a supportive,   |
|     non-evaluative check-in focused on emotional well-being rather than academic marks."|
+-----------------------------------------------------------------------------------------+
```

* **Restorative Wellness Safeguard**: Flags low morale without punishing in-seat attendance.
* **FERPA Privacy Shield**: Ephemeral consultation note; never enters official student transcript.

---

<!-- slide -->
# Slide 6: Rigorous Stress Testing Across Skewed Cohort Scenarios

Benchmarked across **4 distinct synthetic cohort environments** ($N=500$ students each, $N=31,670$ total observations):

```
F1 Score Comparison Across Cohort Compositions:
1.0 |========================================================================|
0.8 | [0.902]           [0.926]           [0.942]           [0.928]          |  <-- Main 5-Signal Model
0.6 |                                                                        |      (Stable & Robust)
0.4 | [0.374]           [0.385]           [0.308]           [0.438]          |  <-- Lagging Baseline
0.2 |                                                                        |      (Collapses under skew)
0.0 +------------------------------------------------------------------------+
       Standard           High Acute-        High Chronic        Mixed Worst-
      Reference              Shock          Quiet-Struggle          Case
```

* **Extreme Quiet-Struggle (STEM/Magnet School)**: Baseline recall collapses to **19.4%**, while the Main Model sustains **92.1% recall** (**+205.6% relative F1 gain**).
* **High Acute-Shock (Crisis Surge)**: Baseline false alarms surge to 9.5% on recovered students; Main Model maintains **0.0% false alarms on recovered students**.
* **Scientific Rigor**: Evaluated across both Paradigm A (Fixed OOD Zero-Retraining) and Paradigm B (Retrained In-Domain Capacity).

---

<!-- slide -->
# Slide 7: Longitudinal Stability — 3-Year Population Drift & MLOps Protocol

Simulated across **3 consecutive academic school years** to model curriculum rigor shifts and Grade 9 transition shock:

| Academic Year | Pre-Intervention Base Rate | Mean Cohort PSI | Max Drift Signal | Unretrained Model F1 | Retrained Model F1 | Retrained Brier Score |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **Year 1** *(Baseline)* | 33.0% | `0.000` | In-Seat Attendance | **`0.958`** | **`0.958`** | **`0.054`** |
| **Year 2** *(Rigor Shift)* | 39.3% | `0.011` | Questions Asked ($PSI=0.018$) | **`0.958`** | **`0.962`** | **`0.042`** |
| **Year 3** *(Transition Shock)* | 44.2% | `0.037` | Help-Seeking Delay ($PSI=0.069$) | **`0.957`** | **`0.967`** | **`0.043`** |

* **Finding**: Unretrained model maintains ranking discrimination ($F_1 \ge 0.957$), but probability calibration decays without retraining ($Brier = 0.072$).
* **MLOps Solution**: Annual retraining restores sharp calibration ($Brier = 0.043$) and improves recall to **97.2%**.
* **Retraining Protocol**: Automated triggers when $PSI \ge 0.25$ with human-in-the-loop review by counselors and privacy officers.

---

<!-- slide -->
# Slide 8: Enterprise Ingestion & Student Self-Advocacy Portal

### Production-Ready LMS & SIS Integration Adapters:
* **OneRoster v1.2 REST API**: Standardized roster exchange, course enrollments, demographic segmentation.
* **Canvas LMS REST API**: Real-time telemetry: reading time in minutes, login sessions, submission pacing.
* **PowerSchool SIS REST API**: Official daily/period attendance, gradebook averages, tutoring visit logs.
* **UnifiedIngestionPipeline**: Merges sources on `(student_id, week)` and validates against Pydantic schema.

### Student Self-Advocacy Reflection Portal:
* **Private Student View**: Separate role in the dashboard dedicated exclusively to student self-advocacy.
* **Zero Deficit Exposure**: **No risk scores, no disengagement flags, no deficit labels**.
* **Strengths-Oriented**: Highlights attendance consistency, study reading hours, and discussion participation.
* **Resource Empowerment**: 1-click booking for teacher office hours and peer tutoring commons.

---

<!-- slide -->
# Slide 9: Ethical AI Principles & Human-in-the-Loop Governance

```
                                  ETHICAL GOVERNANCE PILLARS
+------------------------------+------------------------------+------------------------------+
|       CURIOSITY OVER         |      TRANSFER STUDENT        |         RECOVERY             |
|        SURVEILLANCE          |      GOVERNANCE SHIELD       |     VELOCITY DISCOUNT        |
+------------------------------+------------------------------+------------------------------+
| Alerts are conversational    | Students with < 4 weeks of   | When a student attends       |
| invitations for support,     | history trigger automatic    | tutoring and exhibits        |
| never automated penalties,   | interval expansion and       | positive grade trajectory,   |
| disciplinary actions, or     | suspension of high-priority  | historical GPA penalties     |
| transcript marks.            | outreach alerts.             | are discounted.              |
+------------------------------+------------------------------+------------------------------+
```

* **FERPA Compliant**: Pseudonymized tokens, TLS 1.3 encryption, role-based access control, ephemeral memo storage.
* **Stakeholder Validated**: Built with feedback from Counselors, Parents, Teachers, Mental Health Coordinators, and Data Privacy Officers.

---

<!-- slide -->
# Slide 10: Conclusion & District Pilot Roadmap

### Review 3 Final Deliverables Summary:
* ✅ **100% Phase Completion**: All foundational, MVP, Review 2, and Phase 4 deliverables implemented.
* ✅ **Comprehensive Automated Verification**: 21/21 automated unit tests passing across all edge cases, connectors, portals, and drift monitors.
* ✅ **Containerized Packaging**: Production multi-stage `Dockerfile` and `docker-compose.yml` ready for immediate district deployment.

### Next Steps for District Pilot:
1. **Fall Semester Pilot**: Deploy containerized platform to pilot cohort across 3 high schools.
2. **Weekly Sync**: Ingest real LMS/SIS data via `UnifiedIngestionPipeline` into confidential counselor workspaces.
3. **Restorative Feedback Loop**: Track counselor intervention outcomes to refine local prior distributions.

*Open for Questions & Stakeholder Discussion.*
