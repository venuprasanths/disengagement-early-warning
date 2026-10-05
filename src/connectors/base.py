"""
Base Connector Interface for Educational LMS & SIS Ingestion.

Defines the contract, authentication protocols, retry policies, and status models
shared across all school data ingestion adapters.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
import time
import pandas as pd


class ConnectorStatus(str, Enum):
    """Operational health state of an LMS/SIS connection."""
    CONNECTED = "CONNECTED"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    RATE_LIMITED = "RATE_LIMITED"
    CIRCUIT_BREAKER_OPEN = "CIRCUIT_BREAKER_OPEN"
    UNREACHABLE = "UNREACHABLE"
    DISCONNECTED = "DISCONNECTED"


class BaseConnector(ABC):
    """
    Abstract interface for school data feed connectors.

    All production connectors (OneRoster, Canvas, PowerSchool) must implement
    this contract to enable unified ingestion into the early-warning pipeline.
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        oauth_token: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        timeout_seconds: int = 30,
        max_retries: int = 3,
        rate_limit_per_minute: int = 600,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self.oauth_token = oauth_token
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute
        self.status = ConnectorStatus.DISCONNECTED
        self.last_sync_timestamp: Optional[float] = None
        self._request_count: int = 0

    @abstractmethod
    def authenticate(self) -> bool:
        """
        Authenticates against the remote provider (OAuth2 Bearer or API Key).

        Returns:
            bool: True if authentication succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Pings provider health/ping endpoint to verify connectivity and latency.

        Returns:
            Dict[str, Any]: Diagnostic report containing status, latency_ms, endpoint.
        """
        pass

    @abstractmethod
    def fetch_records(
        self,
        start_date: str,
        end_date: str,
        student_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts raw payload records from the provider REST API.

        Parameters:
            start_date (str): ISO format date 'YYYY-MM-DD'.
            end_date (str): ISO format date 'YYYY-MM-DD'.
            student_ids (Optional[List[str]]): Optional filter for specific student IDs.

        Returns:
            List[Dict[str, Any]]: Raw API records parsed from provider JSON response.
        """
        pass

    @abstractmethod
    def transform_to_unified(self, raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Maps provider-specific schema fields into standard behavioral columns.

        Parameters:
            raw_records (List[Dict[str, Any]]): Raw JSON records from fetch_records.

        Returns:
            pd.DataFrame: Tabular DataFrame aligned to internal schema standards.
        """
        pass

    def execute_with_retry(self, operation_name: str, callable_func, *args, **kwargs) -> Any:
        """
        Executes a network or extraction call with exponential backoff retry.

        Parameters:
            operation_name (str): Semantic name of the operation for logging.
            callable_func: Function or method to execute.

        Returns:
            Any: Result of the callable.

        Raises:
            ConnectionError: If max retries are exhausted.
        """
        retries = 0
        backoff_delay = 0.5
        while retries <= self.max_retries:
            try:
                self._request_count += 1
                return callable_func(*args, **kwargs)
            except Exception as ex:
                retries += 1
                if retries > self.max_retries:
                    self.status = ConnectorStatus.UNREACHABLE
                    raise ConnectionError(
                        f"Connector [{self.__class__.__name__}] failed operation '{operation_name}' "
                        f"after {self.max_retries} retries: {str(ex)}"
                    ) from ex
                time.sleep(backoff_delay)
                backoff_delay *= 2.0
