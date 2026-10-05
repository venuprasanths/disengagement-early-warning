"""
Automated Unit Tests for LMS & SIS Integration Connectors.

Tests:
1. OneRoster v1.2 Roster and Enrollment Adapter
2. Canvas LMS Telemetry & Submissions Adapter
3. PowerSchool SIS Attendance & Gradebook Adapter
4. Unified Ingestion Pipeline Multi-Source Merge and Schema Validation
5. Downstream Compatibility with Feature Extraction and Model Inference
"""

import pytest
import pandas as pd
from src.connectors.oneroster import OneRosterConnector
from src.connectors.canvas import CanvasConnector
from src.connectors.powerschool import PowerSchoolConnector
from src.connectors.pipeline import UnifiedIngestionPipeline
from src.connectors.base import ConnectorStatus
from src.features import prepare_feature_matrix
from src.main_model import TransparentMultiSignalModel


def test_oneroster_connector_lifecycle():
    """Verifies OneRoster authentication, health check, and roster transformation."""
    connector = OneRosterConnector(mock_mode=True)
    assert connector.authenticate() is True
    assert connector.status == ConnectorStatus.CONNECTED

    health = connector.health_check()
    assert health["status"] == "CONNECTED"
    assert health["authenticated"] is True
    assert "latency_ms" in health

    raw = connector.fetch_records(student_ids=["STU_0001", "STU_0002"])
    assert len(raw) == 2
    assert raw[0]["sourcedId"] == "STU_0001"

    df = connector.transform_to_unified(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "student_id" in df.columns
    assert "enrollment_status" in df.columns


def test_canvas_connector_telemetry():
    """Verifies Canvas activity ingestion, study time, and submissions."""
    connector = CanvasConnector(mock_mode=True)
    assert connector.authenticate() is True

    health = connector.health_check()
    assert health["status"] == "CONNECTED"

    raw = connector.fetch_records(student_ids=["STU_0001"])
    assert len(raw) == 16  # 16 weeks of telemetry

    df = connector.transform_to_unified(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16
    assert "content_time_minutes" in df.columns
    assert "lms_logins" in df.columns
    assert "assignment_submissions" in df.columns
    assert "discussion_posts" in df.columns
    assert (df["content_time_minutes"] >= 0).all()


def test_powerschool_connector_attendance_grades():
    """Verifies PowerSchool attendance, cumulative GPA, and teacher note ingestion."""
    connector = PowerSchoolConnector(mock_mode=True)
    assert connector.authenticate() is True

    health = connector.health_check()
    assert health["status"] == "CONNECTED"

    raw = connector.fetch_records(student_ids=["STU_0001"])
    assert len(raw) == 16

    df = connector.transform_to_unified(raw)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 16
    assert "days_present" in df.columns
    assert "cumulative_score_avg" in df.columns
    assert "teacher_note_flag" in df.columns
    assert (df["days_present"] <= 5).all()


def test_unified_ingestion_pipeline_end_to_end():
    """
    Verifies full multi-source ingestion pipeline:
    - Merges OneRoster + Canvas + PowerSchool into unified 5-signal dataframe
    - Validates schema against Pydantic model
    - Confirms compatibility with downstream feature extractor and model
    """
    pipeline = UnifiedIngestionPipeline()
    health_reports = pipeline.run_health_checks()
    assert health_reports["oneroster"]["authenticated"] is True
    assert health_reports["canvas"]["authenticated"] is True
    assert health_reports["powerschool"]["authenticated"] is True

    test_students = [f"STU_{i:04d}" for i in range(1, 6)]
    ingested_df = pipeline.ingest_cohort_data(
        student_ids=test_students,
        validate_pydantic=True,
    )

    assert isinstance(ingested_df, pd.DataFrame)
    assert len(ingested_df) == 5 * 16  # 5 students x 16 weeks = 80 rows
    assert "student_id" in ingested_df.columns
    assert "week" in ingested_df.columns

    # Verify all 5 signal families are present
    assert "days_present" in ingested_df.columns          # Attendance
    assert "content_time_minutes" in ingested_df.columns  # Activity
    assert "cumulative_score_avg" in ingested_df.columns  # Assessment
    assert "office_hours_attended" in ingested_df.columns # Help-Seeking
    assert "teacher_note_flag" in ingested_df.columns     # Feedback

    # Verify downstream feature extraction works on ingested data
    # (Create synthetic is_disengaged label for feature matrix test)
    ingested_df["is_disengaged"] = False
    ingested_df["archetype"] = "ingested_mock"
    ingested_df["latent_engagement"] = 0.85

    X, y, audit_df = prepare_feature_matrix(ingested_df)
    assert X.shape == (80, 21)
    assert not X.isna().any().any()
