"""Telemetry module for reporting web request logs to Cencori."""

from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict

import httpx

from .types import WebTelemetryPayload

if TYPE_CHECKING:
    from .client import Cencori


def _to_api_payload(payload: WebTelemetryPayload) -> Dict[str, Any]:
    """Convert Python snake_case fields to API camelCase fields."""
    raw = asdict(payload)
    wire_payload = {
        "host": raw["host"],
        "method": raw["method"],
        "path": raw["path"],
        "statusCode": raw["status_code"],
        "requestId": raw["request_id"],
        "queryString": raw["query_string"],
        "message": raw["message"],
        "userAgent": raw["user_agent"],
        "referer": raw["referer"],
        "ipAddress": raw["ip_address"],
        "countryCode": raw["country_code"],
        "latencyMs": raw["latency_ms"],
    }
    return {k: v for k, v in wire_payload.items() if v is not None}


class TelemetryModule:
    """Module for best-effort web telemetry ingestion."""

    def __init__(self, client: "Cencori") -> None:
        self._client = client

    def report_web_request(self, payload: WebTelemetryPayload) -> None:
        """
        Report a web request to the Cencori telemetry endpoint.

        This method is best-effort and never raises. Failures are intentionally
        swallowed to avoid impacting application critical paths.
        """
        url = f"{self._client._base_url}/api/v1/telemetry/web"
        headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._client._api_key,
        }

        try:
            with httpx.Client(timeout=self._client._timeout) as http_client:
                http_client.post(url, json=_to_api_payload(payload), headers=headers)
        except Exception:
            return

    async def async_report_web_request(self, payload: WebTelemetryPayload) -> None:
        """
        Async variant of report_web_request.

        This method is best-effort and never raises.
        """
        url = f"{self._client._base_url}/api/v1/telemetry/web"
        headers = {
            "Content-Type": "application/json",
            "CENCORI_API_KEY": self._client._api_key,
        }

        try:
            async with httpx.AsyncClient(timeout=self._client._timeout) as http_client:
                await http_client.post(url, json=_to_api_payload(payload), headers=headers)
        except Exception:
            return
