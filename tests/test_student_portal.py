"""
Automated Unit Tests for Student Self-Advocacy Reflection Portal.

Verifies:
1. Strict Student Privacy: Confirms NO administrative risk scores, disengagement flags,
   or deficit labels are exposed in the student portal view.
2. Strengths-Oriented Data Access: Confirms the view renders positive assets and self-directed resources.
3. Streamlit Compliance: Confirms width='stretch' is used without deprecated use_container_width.
"""

import inspect
import pandas as pd
import pytest
from dashboard.app import render_student_portal


def test_student_portal_privacy_guarantee():
    """
    Verifies that the Student Portal source code strictly adheres to privacy:
    - Never calls model.predict_risk_score()
    - Never accesses 'is_disengaged' ground-truth labels
    - Never exposes counselor intervention thresholds
    - Uses non-punitive, strengths-based language
    """
    source_code = inspect.getsource(render_student_portal)

    # Must NOT call risk prediction
    assert "predict_risk_score" not in source_code, "Student portal must not compute or display administrative risk scores"
    assert "is_disengaged" not in source_code, "Student portal must not access ground-truth disengagement states"
    assert "threshold" not in source_code, "Student portal must not expose intervention cutoffs"

    # Must NOT contain deficit or punitive labels
    forbidden_terms = ["at-risk", "punitive", "truant", "disciplinary", "failing score"]
    for term in forbidden_terms:
        assert term not in source_code.lower(), f"Forbidden deficit term '{term}' found in student portal"

    # Must contain privacy notice and strengths orientation
    assert "Student Privacy Shield" in source_code
    assert "FERPA" in source_code
    assert "Office Hours" in source_code
    assert "Tutoring" in source_code


def test_student_portal_layout_compliance():
    """Confirms no deprecated use_container_width parameter is used."""
    source_code = inspect.getsource(render_student_portal)
    assert "use_container_width" not in source_code, "Deprecated use_container_width found in render_student_portal"
    assert 'width="stretch"' in source_code, "Modern width='stretch' must be used for plots in student portal"
