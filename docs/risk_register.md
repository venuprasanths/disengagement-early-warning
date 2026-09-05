# Risk Register & Algorithmic Governance

A systematic identification and mitigation framework for ethical, operational, and algorithmic risks associated with the Disengagement Early-Warning System.

---

## 1. Risk Register Matrix

| ID | Risk Description | Likelihood | Impact | Severity | Concrete Mitigation Strategy |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **R-01** | **Deficit Mindset & Stigmatization**<br/>Flagging students as "at-risk" triggers confirmation bias in educators (Pygmalion effect) and harms student self-esteem. | High | High | **CRITICAL** | **1. Transparent Scorecard**: Students and parents have full access to view their own metrics.<br/>**2. Growth Framing**: Scores framed as "support opportunities," not moral deficits.<br/>**3. Affirmative Intervention**: Counselors lead with inquiry ("How can we help?"), never punishment. |
| **R-02** | **Surveillance Capitalism & Creep**<br/>Gradual expansion into invasive monitoring (e.g., keystroke logging, webcam/eye-tracking, sentiment analysis of private messages). | Medium | High | **HIGH** | **1. Hard Boundary**: Strict ban on biometric, ambient audio, and keystroke surveillance in system code and charter.<br/>**2. Ephemeral Storage**: Behavioral risk scores are expunged at the end of each academic semester. |
| **R-03** | **Over-Reliance / Automation Bias**<br/>Counselors treat the algorithmic score as an infallible diagnostic verdict rather than an early heuristic. | High | High | **HIGH** | **1. Human-in-the-Loop Mandate**: No automated sanctions, emails, or disciplinary flags permitted.<br/>**2. Uncertainty Bands**: UI displays wide confidence intervals when data is noisy or sparse to deliberately discourage overconfidence. |
| **R-04** | **Gaming & Goodhart's Law**<br/>Students discover that LMS logins are tracked and script automated tab openers to inflate activity metrics. | High | Medium | **MEDIUM** | **1. Divergence Features**: Feature pipeline computes `logins_per_active_hour` ratio.<br/>**2. Multi-Signal Fusion**: High logins without corresponding quiz engagement or help-seeking still elevates risk score. |
| **R-05** | **Algorithmic Bias / Neurodiversity Blindspots**<br/>Quiet, introverted, or neurodiverse students who rarely post on forums are misclassified as disengaged. | Medium | High | **HIGH** | **1. Multi-Dimensional Decoupling**: Solitary learners who master concepts through reading and independent quizzes maintain low risk despite 0 forum posts.<br/>**2. Archetype Error Audits**: Continuous stratified auditing across learning archetypes. |
| **R-06** | **Subjectivity in Qualitative Notes**<br/>Teacher observation notes reflect interpersonal bias or cultural misunderstandings. | High | Medium | **MEDIUM** | **1. Weak Prior Weighting**: Sentiment and teacher notes are constrained as low-weight, lagged Bayesian signals.<br/>**2. Multi-Signal Threshold**: Qualitative notes cannot trigger an intervention without corroborating objective signals. |
| **R-07** | **Model Drift Across Academic Terms**<br/>Changes in curriculum, grading curves, or LMS tooling alter feature distributions over time. | Medium | Medium | **MEDIUM** | **1. Temporal Holdout Validation**: Continuous backtesting across semester transitions.<br/>**2. Recalibration Protocol**: Periodic re-estimation of Platt scaling parameters before each term. |

---

## 2. Governance Protocol & Escalation Hierarchy

```
[ Algorithmic Signal Detected ]
               │
               ▼
[ Tier 1: Automated Sanity Checks ]
  • Is interval width > 0.40? (Sparse history / transfer student)
  • Is there an active medical/bereavement exception filed?
               │
               ▼
[ Tier 2: Counselor Review (Human-in-the-Loop) ]
  • Counselor examines 5-signal waterfall breakdown
  • Checks recent qualitative context
               │
               ▼
[ Tier 3: Non-Punitive Collaborative Outreach ]
  • Informal 10-minute check-in with student
  • Offer peer tutoring, academic coaching, or schedule adjustments
```
