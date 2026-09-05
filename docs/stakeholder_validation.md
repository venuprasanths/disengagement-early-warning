# Stakeholder Validation & Algorithmic Iteration Log

Simulated qualitative feedback sessions conducted with key stakeholder representatives, capturing observed strengths, operational concerns, and direct code-level modifications prompted by their reviews.

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

## 4. Concrete System Changes Prompted by Feedback

Based on this stakeholder validation feedback, the following **two concrete code enhancements** were directly implemented in the codebase:

### Change 1: "Sparse History Data Guardrail" (`src/main_model.py`, `src/uncertainty.py`, `dashboard/app.py`)
- **Direct Prompt from Marcus Chen (Parent Advocate)**:
- **Implementation**: We added `apply_governance_guardrails()` to the modeling and dashboard pipeline.
- **Rule**: If `weeks_available < 4`, the system automatically suppresses any high-priority disengagement alert and reclassifies the student's status as `MONITOR_ONLY_SPARSE_DATA`, displaying a prominent yellow shield in the dashboard indicating that algorithmic decision-making is suspended until sufficient baseline history is acquired.

### Change 2: "Positive Recovery Velocity Discount" (`src/features.py`, `src/main_model.py`)
- **Direct Prompt from Sarah Jenkins (Teacher) & Dr. Elena Rostova (Counselor)**:
- **Implementation**: When a student exhibits an upward score trajectory (`score_trend_slope > +2.5`) coupled with active help-seeking (`tutoring_3wk_sum >= 1` or `office_hours_3wk_sum >= 1`), the system applies a protective discount to the raw risk score.
- **Impact**: Eliminates persistent false alarms for Archetype 3 ("Genuinely Improving") students whose cumulative GPAs are still temporarily dragged down by early-semester marks.
