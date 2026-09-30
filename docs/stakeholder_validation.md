# Stakeholder Validation & Algorithmic Iteration Log

Simulated qualitative feedback sessions conducted with key educational stakeholder representatives, capturing observed strengths, operational concerns, and direct code-level modifications prompted by their reviews.

---

## 1. Stakeholder Review 1: Academic Counselor
* **Reviewer**: **Dr. Elena Rostova**, Lead Early Intervention Specialist (Caseload: 260 students)
* **What She Liked**:
  > "The lead time advantage is a game-changer. Historically, I only receive an automated flag after a student fails their week 8 midterm or misses 4 consecutive days of school. By then, their hole is so deep they've mentally checked out. Being able to see the 'Quietly Struggling' student flagged around week 4 or 5—because their help-seeking collapsed and their LMS engagement flattened—gives us a realistic window to sit down and help before an academic catastrophe."
* **What Concerned Her**:
  > "My biggest worry is caseload thrashing. If a student gets the flu or has a single rough week, their score might temporarily spike. If I reach out to 25 kids every time someone has a transient dip, I'll exhaust both my time and the students' goodwill."
* **Her Recommendation**:
  > "Require a multi-week persistence check or display an explicit 'Trend Velocity' badge. Don't let a single-week fluctuation trigger the high-priority queue."

---

## 2. Stakeholder Review 2: Student & Parent Advocate
* **Reviewer**: **Marcus Chen**, Parent Advisory Council Representative
* **What He Liked**:
  > "I appreciate the transparency of the Trade-Off Dashboard. For once, an institution isn't pretending their algorithm is magic. Seeing the slider make the trade-off visible—showing that higher counselor recall directly degrades student precision—creates real accountability."
* **What Concerned Him**:
  > "I am deeply uncomfortable with how transfer students and mid-term enrollments are handled. If a kid transfers in week 7, and the system tries to guess their risk with only 1 or 2 weeks of records, that guess is basically arbitrary. If a counselor gets an alert saying a transfer student is 'at-risk', that follows the student and creates an unfair first impression."
* **His Recommendation**:
  > "Build a hard guardrail in the software: if a student has fewer than 4 weeks of data, the system must NOT issue an intervention alert. Mark them as 'Insufficient Baseline Data / Observational Period Only' and enforce high uncertainty."

---

## 3. Stakeholder Review 3: Classroom Teacher
* **Reviewer**: **Sarah Jenkins**, 11th Grade Mathematics Department Chair
* **What She Liked**:
  > "The signal family breakdown is vastly superior to our current SIS red flags. Knowing *why* the student is struggling—e.g., 'zero questions asked despite quiz dip' vs. 'attending but not reading material'—gives me an immediate, actionable intervention I can use in class without feeling like a disciplinarian."
* **What Concerned Her**:
  > "I have students who bombed the first two quizzes, had a wake-up call, and are now working with our peer tutors and showing steady improvement. Our old gradebook keeps them marked 'at risk' for months because of cumulative averages. I need to make sure this system recognizes positive momentum and doesn't penalize kids who are genuinely turning things around."
* **Her Recommendation**:
  > "Make sure that active tutoring and positive 3-week score slope actively suppress false alarms and get recognized as protective factors."

---

## 4. Stakeholder Review 4: School Social Worker & District Mental Health Coordinator (Review 2 Addition)
* **Reviewer**: **David Vance, LCSW**, District Mental Health Coordinator & Crisis Team Lead
* **What He Liked**:
  > "Integrating bi-weekly pulse surveys and qualitative morale into early warning is a profound improvement. In adolescent development, emotional alienation and family crisis almost always precede academic failure by several weeks. Having early visibility into student sentiment allows us to intervene before academic collapse."
* **What Concerned Him**:
  > "My gravest concern is deficit mislabeling and trauma insensitivity. If a student experiences an acute personal tragedy (bereavement, housing instability, or clinical depression), their pulse survey morale will crater. If the software labels them as 'High Disengagement Risk' and the counselor schedules an academic tutoring session, the student feels misunderstood and alienated. Pushing math problem sets on a grieving child causes severe psychological withdrawal."
* **His Recommendation**:
  > "Implement an automated **Compassionate Restorative Triage** safeguard: if a student's risk flag is driven by severe sentiment and confidence drops, but their in-seat attendance is faithful and they have zero disciplinary infractions, classify them under a **Restorative Wellness Protocol** rather than an Academic Remediation Protocol. Give counselors an empathetic, trauma-informed conversation guide instead of academic drill sheets."

---

## 5. Stakeholder Review 5: District IT & Student Privacy Compliance Officer (Review 2 Addition)
* **Reviewer**: **Rachel Torres, CISSP**, District Chief Information Security & Compliance Officer
* **What She Liked**:
  > "The strict information barrier quarantining ground-truth labels and latent engagement states is exemplary from an algorithmic audit perspective. The model trains strictly on observable behavioral indicators without label contamination."
* **What Concerned Her**:
  > "Under FERPA (Family Educational Rights and Privacy Act) and state student privacy regulations, students and parents possess a legal right to inspect educational records and understand algorithmic assessments. If our counselors log technical ML features like `logins_per_active_hour` or `att_trend_slope`, parents cannot meaningfully interpret them, creating legal vulnerability. Furthermore, we must guarantee that these early warning scores are strictly ephemeral and do not persist into cumulative student transcripts."
* **Her Recommendation**:
  > "Require plain-language translation of all model features into certified educator terminology. Add an automated, FERPA-compliant **Consultation Memorandum** export tool that provides standardized, non-stigmatizing summary notes with explicit legal disclaimers stating that the score is a temporary decision aid, not a permanent academic evaluation."

---

## 6. Concrete System Changes Prompted by Feedback (All 4 Changes Implemented)

Based on these stakeholder reviews, the following **four concrete code enhancements** have been directly implemented in the codebase:

### Change 1: "Sparse History Data Guardrail" (`src/uncertainty.py`, `dashboard/app.py`)
- **Direct Prompt from Marcus Chen (Parent Advocate)**:
- **Implementation**: In `src/uncertainty.py` (lines 122–130) and `dashboard/app.py` (lines 314–318), if `weeks_available < 4`, the system automatically expands the uncertainty interval ($\text{margin} \ge 0.20$), sets `sparse_data_flag = True`, and activates the yellow **Governance Shield** in the dashboard, suppressing intervention alerts until a reliable 4-week baseline is acquired.

### Change 2: "Positive Recovery Velocity Discount" (`src/features.py`, `src/main_model.py`)
- **Direct Prompt from Sarah Jenkins (Teacher) & Dr. Elena Rostova (Counselor)**:
- **Implementation**: When a student demonstrates an upward score trajectory (`score_trend_slope > +2.5`) combined with active academic help-seeking (`tutoring_3wk_sum >= 1` or `office_hours_3wk_sum >= 1`), the system applies an active protective factor discount, eliminating the historical GPA drag that caused the baseline to generate $>40\%$ false alarms on recovering students.

### Change 3: "Compassionate Restorative Wellness Triage Safeguard" (`src/main_model.py`, `dashboard/app.py`)
- **Direct Prompt from David Vance, LCSW (Mental Health Coordinator)**:
- **Implementation**: Added triage logic to `TransparentMultiSignalModel.explain_instance()` in `src/main_model.py`. If a student exhibits severe morale or confidence drops (`survey_sentiment_recent <= -0.35` or `confidence_rating_recent <= 2.0`), coupled with strong in-seat attendance (`att_3wk_mean >= 3.8`) and zero unexcused absences (`unexcused_absences_sum == 0`), the model automatically routes the alert to `RESTO_WELLNESS_CHECK` ("Compassionate Wellness Check-In (Non-Academic)").
- **Dashboard Display**: Renders a dedicated green wellness banner with trauma-informed conversational outreach guidance, ensuring students receive emotional support rather than inappropriate academic remediation.

### Change 4: "FERPA-Compliant Plain-Language Consultation Memorandum Generator" (`src/main_model.py`, `dashboard/app.py`)
- **Direct Prompt from Rachel Torres, CISSP (Privacy & Compliance Officer)**:
- **Implementation**: Created `export_counselor_audit_record()` in `src/main_model.py` and Tab 5 in `dashboard/app.py`. Translates all 21 raw model columns into plain-language educational definitions (`FEATURE_PLAIN_LANGUAGE_MAPPING`), groups them into primary behavioral drivers and student protective strengths, and appends mandatory FERPA privacy notices confirming that early warning scores are ephemeral decision-support aids that never enter permanent transcripts.
