# Stakeholder Assumptions & Ethical Requirements

## 1. Executive Summary & Problem Framing

Current school intervention systems rely heavily on lagging indicators—specifically attendance thresholds (e.g., `< 85%`) and failing grades (e.g., `< 60%`). In practice, these indicators act as autopsies rather than early warnings: by the time a student consistently skips classes or fails midterms, their academic, social, and emotional disengagement is deeply entrenched.

Remediating entrenched disengagement is expensive, stressful, and frequently too late. Furthermore, these lagging metrics feel punitive: students receive letters or detention notices when what they needed weeks earlier was scaffolded academic support, peer tutoring, or mental health check-ins.

This system seeks to fuse five weak signal families (Attendance, LMS/Course Activity, Assessment Trends, Help-Seeking Behaviors, and Qualitative Sentiment/Notes) into a calibrated, uncertainty-aware disengagement risk score. However, deploying an early warning system inherently introduces conflicting stakeholder values that cannot be resolved by machine learning alone. This document explicitly formalizes those tensions.

---

## 2. Conflicting Stakeholder Groups & Objective Functions

### Stakeholder A: The Academic Counselor / Early Intervention Specialist
* **Primary Objective**: **High Recall (Sensitivity)**.
* **Core Motivation**: "Catch every single student who is silently slipping before they reach academic crisis."
* **Cost Asymmetry**:
  - **False Negative (FN)**: Catastrophic cost. A student silently drops out or fails a course without any intervention. The school incurs retention loss, remediation expenses, and fails its educational duty.
  - **False Positive (FP)**: Low-to-moderate cost. A counselor conducts an exploratory 10-minute 1-on-1 check-in with a student who was actually doing fine. The counselor views this as an opportunity to build rapport anyway.
* **Preferred Operating Point**: Low decision threshold (e.g., flag top 25–30% of cohort or risk score $\ge 0.35$). Wants early flags even if confidence has wide error bands.

### Stakeholder B: The Student & Parent Advocate Group
* **Primary Objective**: **High Precision & Privacy Preservation**.
* **Core Motivation**: "Do not stigmatize students, do not create self-fulfilling deficit labels, and do not subject students to invasive algorithmic surveillance."
* **Cost Asymmetry**:
  - **False Negative (FN)**: Moderate cost. The family prefers autonomy and organic problem-solving over unwarranted institutional intrusion.
  - **False Positive (FP)**: Severe harm. Being tagged as "at-risk" triggers confirmation bias among educators, causes anxiety for parents, damages student self-esteem, and can lead to lower teacher expectations (the Pygmalion effect).
* **Preferred Operating Point**: High decision threshold (e.g., risk score $\ge 0.70$) with narrow uncertainty intervals and clear explainability showing *why* a student was flagged.

### Stakeholder C: The Classroom Teacher
* **Primary Objective**: **Actionability and Low Cognitive Load**.
* **Core Motivation**: "Give me specific, non-punitive actions I can take this week without requiring me to spend 5 hours reviewing dashboards or inputting data."
* **Requirements**: Needs explanations grounded in modifiable behaviors (e.g., "Student has not asked questions despite quiz dip" $\to$ invite to small study group), not fatalistic predictions.

---

## 3. Operational Definition of "Disengagement"

Disengagement is frequently misconstrued as simple absenteeism or low intellectual capability. In this system, disengagement is operationalized as a **multi-dimensional progressive withdrawal from the learning community**:

1. **Behavioral Withdrawal**: Reduction in effort, non-completion of formative tasks, passive physical presence without mental presence (e.g., sitting in the back, opening LMS tabs without active reading).
2. **Cognitive Withdrawal**: Ceasing to seek help when struggling, avoiding office hours, failing to review feedback, or turning in unattempted work.
3. **Affective / Emotional Withdrawal**: Deteriorating self-reported confidence, feelings of isolation, negative sentiment in feedback surveys, and visible detachment noted by educators.

### What Disengagement is NOT:
* It is **not** low academic ability. A struggling student who asks questions, attends tutoring, and engages actively is *engaged*, not disengaged.
* It is **not** a fixed student trait or moral failing. It is a temporary, reversible behavioral state influenced by environmental, emotional, or curriculum factors.

---

## 4. Assumptions Regarding Data Availability, Consent, and Privacy

| Area | Realistic Assumption in Practice | System Boundary / Guardrails |
| :--- | :--- | :--- |
| **Data Collection** | All 5 signal families exist in typical modern school LMS (Canvas/Moodle/Blackboard), SIS (PowerSchool), and weekly pulse tools. | No biometric, keystroke tracking, eye-tracking, or ambient audio data is ever collected or permitted. |
| **Consent & Transparency** | Students and parents have a right to algorithmic transparency under FERPA and GDPR principles. | Every student flagged by the system has access to their own transparent signal scorecard and explanation. Black-box models are disallowed. |
| **Qualitative Sentiment** | Teacher notes and student pulse surveys are noisy, subjective, and prone to interpersonal bias. | Sentiment scores are intentionally treated as weak, lagged Bayesian priors—never permitted to single-handedly trigger an intervention alert. |
| **Data Retention** | Predictive risk scores are ephemeral operational artifacts for the current semester only. | Historical risk scores must NOT become permanent entries in a student's permanent disciplinary or academic transcript. |

---

## 5. Decision Threshold Governance & The "Human-in-the-Loop" Mandate

1. **No Automated Sanctions**: The system is strictly an *Early-Warning / Resource-Allocation Recommender*. Under no circumstances may an automated notification, grade penalty, or disciplinary record be generated directly by model output.
2. **Dynamic Thresholding**: Rather than hiding behind an arbitrary fixed 0.50 threshold, the school's leadership, counselors, and student advocates must collectively calibrate the operating threshold based on available counselor capacity and risk tolerance, exposed live through the Stakeholder Trade-off Dashboard.
