"""
Unified Multi-Source Ingestion Pipeline.

Orchestrates data ingestion across OneRoster (roster/enrollments), Canvas (digital activity),
and PowerSchool (attendance/grades/support logs), fuses them into the standardized 5-signal
format, and enforces schema integrity and information barriers.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from pydantic import ValidationError

from src.schema import (
    RAW_FEATURE_COLUMNS,
    ID_COLUMNS,
    StudentWeeklyRecord,
    assert_no_ground_truth_leakage,
)
from src.connectors.oneroster import OneRosterConnector
from src.connectors.canvas import CanvasConnector
from src.connectors.powerschool import PowerSchoolConnector


class UnifiedIngestionPipeline:
    """
    Ingestion coordinator that synchronizes school data feeds and outputs
    a consolidated, validated 5-signal DataFrame.
    """

    def __init__(
        self,
        oneroster: Optional[OneRosterConnector] = None,
        canvas: Optional[CanvasConnector] = None,
        powerschool: Optional[PowerSchoolConnector] = None,
    ):
        self.oneroster = oneroster or OneRosterConnector()
        self.canvas = canvas or CanvasConnector()
        self.powerschool = powerschool or PowerSchoolConnector()

    def run_health_checks(self) -> Dict[str, Any]:
        """Runs health checks on all registered connectors."""
        return {
            "oneroster": self.oneroster.health_check(),
            "canvas": self.canvas.health_check(),
            "powerschool": self.powerschool.health_check(),
        }

    def ingest_cohort_data(
        self,
        start_date: str = "2026-01-05",
        end_date: str = "2026-05-15",
        student_ids: Optional[List[str]] = None,
        validate_pydantic: bool = True,
    ) -> pd.DataFrame:
        """
        Executes parallel extraction across feeds and merges into unified schema.

        Processing Steps:
        1. Query OneRoster for active roster and student identifiers.
        2. Query Canvas LMS for weekly telemetry (logins, reading time, submissions, forum posts).
        3. Query PowerSchool SIS for attendance, GPA, teacher notes, and tutoring records.
        4. Inner-join on ('student_id', 'week').
        5. Verify no ground-truth leakage and validate records with Pydantic.

        Returns:
            pd.DataFrame: Merged DataFrame containing ID_COLUMNS + RAW_FEATURE_COLUMNS.
        """
        # 1. Fetch Roster
        raw_roster = self.oneroster.fetch_records(start_date, end_date, student_ids)
        roster_df = self.oneroster.transform_to_unified(raw_roster)
        active_ids = roster_df[roster_df["enrollment_status"] == "active"]["student_id"].tolist()

        # 2. Fetch Canvas Activity
        raw_canvas = self.canvas.fetch_records(start_date, end_date, active_ids)
        canvas_df = self.canvas.transform_to_unified(raw_canvas)

        # 3. Fetch PowerSchool Attendance & Grades
        raw_ps = self.powerschool.fetch_records(start_date, end_date, active_ids)
        ps_df = self.powerschool.transform_to_unified(raw_ps)

        # 4. Multi-Way Merge on (student_id, week)
        merged = pd.merge(ps_df, canvas_df, on=["student_id", "week"], how="inner")

        # Ensure all raw feature columns are present
        for col in RAW_FEATURE_COLUMNS:
            if col not in merged.columns:
                merged[col] = 0.0

        # Enforce Information Barrier
        assert_no_ground_truth_leakage(list(merged.columns))

        # Reorder columns
        output_cols = ID_COLUMNS + RAW_FEATURE_COLUMNS
        final_df = merged[output_cols].sort_values(["student_id", "week"]).reset_index(drop=True)

        # 5. Pydantic Schema Validation (Sample check for speed)
        if validate_pydantic:
            sample_records = final_df.head(20).to_dict(orient="records")
            for rec in sample_records:
                sanitized_rec = {k: (None if pd.isna(v) else v) for k, v in rec.items()}
                try:
                    StudentWeeklyRecord(**sanitized_rec)
                except ValidationError as ve:
                    raise ValueError(f"Ingested record failed schema validation: {ve}") from ve

        return final_df
