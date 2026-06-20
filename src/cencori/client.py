"""Cencori SDK client."""

import time
from typing import Any, Dict, Optional, cast

import httpx

from .agents import AgentsModule
from .ai import AIModule
from .api_keys import APIKeysModule
from .errors import AuthenticationError, CencoriError, RateLimitError, SafetyError
from .memory import MemoryModule
from .metrics import MetricsModule
from .projects import ProjectsModule
from .telemetry import TelemetryModule

DEFAULT_BASE_URL = "https://api.cencori.com"
MAX_RETRIES = 3


class ComputeModule:
    """
    Compute module - Serverless functions & GPU access.

    🚧 Coming Soon
    """

    def run(self, function_name: str, **kwargs: Any) -> Any:
        """Run a serverless function."""
        raise NotImplementedError("Compute module coming soon")

    def deploy(self, name: str, code: str) -> Any:
        """Deploy a function."""
        raise NotImplementedError("Compute module coming soon")

    def list(self) -> Any:
        """List deployed functions."""
        raise NotImplementedError("Compute module coming soon")


class WorkflowModule:
    """
    Workflow module - AI pipelines & orchestration.

    🚧 Coming Soon
    """

    def trigger(self, workflow_id: str, **kwargs: Any) -> Any:
        """Trigger a workflow."""
        raise NotImplementedError("Workflow module coming soon")

    def create(self, name: str, steps: list[Any]) -> Any:
        """Create a workflow."""
        raise NotImplementedError("Workflow module coming soon")

    def status(self, run_id: str) -> Any:
        """Get workflow run status."""
        raise NotImplementedError("Workflow module coming soon")

    def list(self) -> Any:
        """List workflows."""
        raise NotImplementedError("Workflow module coming soon")


class StorageModule:
    """
    Storage module - Vector database, knowledge base, RAG.

    🚧 Coming Soon
    """

    def __init__(self) -> None:
        self.vectors = VectorsSubmodule()
        self.knowledge = KnowledgeSubmodule()
        self.files = FilesSubmodule()


class VectorsSubmodule:
    """Vector storage submodule."""

    def search(self, query: str, **kwargs: Any) -> Any:
        """Search vectors."""
        raise NotImplementedError("Storage module coming soon")

    def upsert(self, vectors: Any, **kwargs: Any) -> Any:
        """Upsert vectors."""
        raise NotImplementedError("Storage module coming soon")

    def delete(self, ids: list[Any]) -> Any:
        """Delete vectors by ID."""
        raise NotImplementedError("Storage module coming soon")


class KnowledgeSubmodule:
    """Knowledge base submodule."""

    def query(self, question: str) -> Any:
        """Query the knowledge base."""
        raise NotImplementedError("Storage module coming soon")

    def add(self, documents: list[Any]) -> Any:
        """Add documents to knowledge base."""
        raise NotImplementedError("Storage module coming soon")


class FilesSubmodule:
    """File storage submodule."""

    def upload(self, file: Any, name: str) -> Any:
        """Upload a file."""
        raise NotImplementedError("Storage module coming soon")

    def process(self, file_id: str) -> Any:
        """Process a file."""
        raise NotImplementedError("Storage module coming soon")


class Cencori:
    """
    Cencori SDK client.

    One SDK for AI Gateway, Compute, Workflow, and Storage.
    Every operation is secured, logged, and tracked.

    Args:
        api_key: Your Cencori API key (starts with 'csk_')
        base_url: API base URL (default: https://api.cencori.com)
        timeout: Request timeout in seconds (default: 30)
        headers: Custom headers to include in requests

    Example:
        >>> from cencori import Cencori
        >>> cencori = Cencori(api_key="csk_...")
        >>> response = cencori.ai.chat(
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
        >>> print(response.content)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        import os

        # Get API key from argument or environment
        resolved_api_key = api_key or os.environ.get("CENCORI_API_KEY")

        if not resolved_api_key:
            raise ValueError(
                "Cencori API key is required. "
                "Pass it via Cencori(api_key='csk_...') or set CENCORI_API_KEY environment variable."
            )

        self._api_key = resolved_api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = headers or {}

        # Initialize modules
        self.ai = AIModule(self)
        self.agents = AgentsModule(self)
        self.projects = ProjectsModule(self)
        self.api_keys = APIKeysModule(self)
        self.metrics = MetricsModule(self)
        self.memory = MemoryModule(self)
        self.telemetry = TelemetryModule(self)

        self.compute = ComputeModule()
        self.workflow = WorkflowModule()
        self.storage = StorageModule()

    # =========================================================================
    # Retry Logic
    # =========================================================================

    @staticmethod
    def _sleep(seconds: float) -> None:
        """Sleep for the given number of seconds."""
        time.sleep(seconds)

    def _fetch_with_retry(
        self,
        method: str,
        url: str,
        json_body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        max_retries: int = MAX_RETRIES,
    ) -> httpx.Response:
        """Make an HTTP request with automatic retry on 5xx errors.

        Uses exponential backoff: 1s, 2s, 4s
        """
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.request(
                        method=method,
                        url=url,
                        json=json_body,
                        headers=headers,
                    )

                    # Return immediately if successful or client error (4xx)
                    if response.is_success or (400 <= response.status_code < 500):
                        return response

                    # Retry on 5xx errors
                    last_error = CencoriError(
                        f"HTTP {response.status_code}: {response.reason_phrase}",
                        status_code=response.status_code,
                    )

                    # Don't retry on last attempt
                    if attempt == max_retries - 1:
                        return response

                    # Exponential backoff: 1s, 2s, 4s
                    self._sleep(pow(2, attempt) * 1.0)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                self._sleep(pow(2, attempt) * 1.0)

            except httpx.HTTPError as e:
                last_error = e
                if attempt == max_retries - 1:
                    raise
                self._sleep(pow(2, attempt) * 1.0)

        if last_error:
            raise last_error

        raise CencoriError("Max retries reached")

    # =========================================================================
    # Synchronous Request Methods
    # =========================================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make a synchronous HTTP request to the Cencori API with retry."""
        url = f"{self._base_url}{endpoint}"

        request_headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._api_key,
            **self._headers,
        }
        if headers:
            request_headers.update(headers)

        response = self._fetch_with_retry(
            method=method,
            url=url,
            json_body=json,
            headers=request_headers,
        )

        return self._handle_response(response)

    def request(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a generic API request.

        Args:
            endpoint: API endpoint path (e.g., '/api/v1/custom')
            method: HTTP method (GET, POST, PUT, DELETE)
            body: Request body as dict
            headers: Additional headers

        Returns:
            Response data as dict

        Example:
            >>> data = cencori.request('/api/v1/custom', method='POST', body={'foo': 'bar'})
        """
        return self._request(method, endpoint, json=body, headers=headers)

    # =========================================================================
    # Async Request Methods
    # =========================================================================

    async def _async_request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Make an async HTTP request to the Cencori API."""
        url = f"{self._base_url}{endpoint}"

        request_headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._api_key,
            **self._headers,
        }
        if headers:
            request_headers.update(headers)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.request(
                method=method,
                url=url,
                json=json,
                headers=request_headers,
            )

        return self._handle_response(response)

    async def async_request(
        self,
        endpoint: str,
        method: str = "GET",
        body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Make a generic async API request.

        Same as request() but async.
        """
        return await self._async_request(method, endpoint, json=body, headers=headers)

    # =========================================================================
    # Response Handling
    # =========================================================================

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """Handle HTTP response and raise appropriate errors."""
        if response.status_code == 401:
            raise AuthenticationError()

        if response.status_code == 429:
            raise RateLimitError()

        data = cast(Dict[str, Any], response.json())

        if response.status_code == 400 and "reasons" in data:
            raise SafetyError(
                message=data.get("error", "Content safety violation"),
                reasons=data.get("reasons", []),
            )

        if not response.is_success:
            raise CencoriError(
                message=data.get("error", "Request failed"),
                status_code=response.status_code,
            )

        return data

    # =========================================================================
    # Raw HTTP Methods (for modules that need direct access)
    # =========================================================================

    def _request_raw(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> httpx.Response:
        """Make a request and return the raw response.

        Used by modules that need to inspect raw response status (e.g., telemetry).
        """
        url = f"{self._base_url}{endpoint}"

        request_headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._api_key,
            **self._headers,
        }
        if headers:
            request_headers.update(headers)

        with httpx.Client(timeout=self._timeout) as client:
            response = client.request(
                method=method,
                url=url,
                json=json,
                headers=request_headers,
            )

        return response

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_base_url(self) -> str:
        """Get the base URL for API calls."""
        return self._base_url

    def get_api_key(self) -> str:
        """Get the API key."""
        return self._api_key

    def get_config(self) -> Dict[str, str]:
        """Get the current configuration (API key is masked)."""
        return {
            "base_url": self._base_url,
            "api_key_hint": f"{self._api_key[:6]}...{self._api_key[-4:]}",
        }
