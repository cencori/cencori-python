"""Telemetry module for reporting web traffic to the Cencori dashboard."""

from typing import TYPE_CHECKING, Optional

from .types import WebTelemetryPayload

if TYPE_CHECKING:
    from .client import Cencori


class TelemetryModule:
    """
    Module for reporting web traffic to the Cencori dashboard.

    Logs appear under the project's "Web Gateway" tab with your app's
    real domain as the host.

    Fire-and-forget — methods never throw and never block.
    """

    def __init__(self, client: "Cencori") -> None:
        self._client = client

    def report_web_request(
        self,
        host: str,
        method: str,
        path: str,
        status_code: int,
        request_id: Optional[str] = None,
        query_string: Optional[str] = None,
        message: Optional[str] = None,
        user_agent: Optional[str] = None,
        referer: Optional[str] = None,
        ip_address: Optional[str] = None,
        country_code: Optional[str] = None,
        latency_ms: Optional[int] = None,
    ) -> None:
        """
        Report a web request to the Cencori dashboard.

        Fire-and-forget — this method never throws and never blocks
        your application. If the request fails, the error is silently
        swallowed.

        Args:
            host: The hostname of your application (e.g., "myapp.vercel.app")
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            path: Request path (e.g., "/api/chat")
            status_code: HTTP status code returned
            request_id: Optional unique request ID for tracing
            query_string: Query string without leading "?"
            message: Optional human-readable log message
            user_agent: Client User-Agent string
            referer: Referer header value
            ip_address: Client IP address
            country_code: ISO 3166-1 alpha-2 country code
            latency_ms: Request latency in milliseconds

        Example:
            >>> cencori.telemetry.report_web_request(
            ...     host="myapp.vercel.app",
            ...     method="GET",
            ...     path="/api/chat",
            ...     status_code=200,
            ... )
        """
        payload = WebTelemetryPayload(
            host=host,
            method=method,
            path=path,
            status_code=status_code,
            request_id=request_id,
            query_string=query_string,
            message=message,
            user_agent=user_agent,
            referer=referer,
            ip_address=ip_address,
            country_code=country_code,
            latency_ms=latency_ms,
        )

        try:
            # Use raw request to avoid error handling overhead
            self._client._request_raw(
                "POST",
                "/api/v1/telemetry/web",
                json=payload.__dict__,
            )
        except Exception:
            # Telemetry is best-effort — never disrupt the customer's app.
            pass
