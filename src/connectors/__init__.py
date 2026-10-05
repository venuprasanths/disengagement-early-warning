"""
LMS & SIS Integration Connectors for School Data Ingestion.

Provides production-ready interface specifications and realistic mock adapters
for industry-standard school data feeds:
- OneRoster v1.2 (IMS Global / 1EdTech Roster Exchange)
- Canvas LMS REST API (Instructure Learning Platform)
- PowerSchool SIS REST API (Student Information System)
- UnifiedIngestionPipeline (Multi-source temporal behavioral fusion)
"""

from src.connectors.base import BaseConnector, ConnectorStatus
from src.connectors.oneroster import OneRosterConnector
from src.connectors.canvas import CanvasConnector
from src.connectors.powerschool import PowerSchoolConnector
from src.connectors.pipeline import UnifiedIngestionPipeline

__all__ = [
    "BaseConnector",
    "ConnectorStatus",
    "OneRosterConnector",
    "CanvasConnector",
    "PowerSchoolConnector",
    "UnifiedIngestionPipeline",
]
