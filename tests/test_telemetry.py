"""Tests for telemetry module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from cencori import Cencori
from cencori.types import WebTelemetryPayload


class TestTelemetryModule:
    """Test telemetry best-effort reporting behavior."""

    def test_telemetry_module_exists(self, api_key: str) -> None:
        """Telemetry module is initialized on the client."""
        client = Cencori(api_key=api_key)
        assert hasattr(client, "telemetry")

    def test_report_web_request_posts_expected_payload(self, api_key: str) -> None:
        """Sync telemetry maps snake_case to API camelCase fields."""
        client = Cencori(api_key=api_key)
        payload = WebTelemetryPayload(
            host="example.com",
            method="GET",
            path="/api/chat",
            status_code=200,
            request_id="req_123",
            query_string="q=1",
            message="ok",
            user_agent="pytest",
            referer="https://ref.example",
            ip_address="1.2.3.4",
            country_code="US",
            latency_ms=42,
        )

        with patch("cencori.telemetry.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client

            client.telemetry.report_web_request(payload)

        expected_json = {
            "host": "example.com",
            "method": "GET",
            "path": "/api/chat",
            "statusCode": 200,
            "requestId": "req_123",
            "queryString": "q=1",
            "message": "ok",
            "userAgent": "pytest",
            "referer": "https://ref.example",
            "ipAddress": "1.2.3.4",
            "countryCode": "US",
            "latencyMs": 42,
        }

        mock_client.post.assert_called_once_with(
            "https://cencori.com/api/v1/telemetry/web",
            json=expected_json,
            headers={
                "Content-Type": "application/json",
                "CENCORI_API_KEY": api_key,
            },
        )

    def test_report_web_request_never_raises(self, api_key: str) -> None:
        """Sync telemetry swallows transport errors."""
        client = Cencori(api_key=api_key)
        payload = WebTelemetryPayload(
            host="example.com",
            method="POST",
            path="/api/chat",
            status_code=500,
        )

        with patch("cencori.telemetry.httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.post.side_effect = RuntimeError("network down")
            mock_client_cls.return_value.__enter__.return_value = mock_client

            client.telemetry.report_web_request(payload)

    def test_async_report_web_request_never_raises(self, api_key: str) -> None:
        """Async telemetry swallows transport errors."""
        client = Cencori(api_key=api_key)
        payload = WebTelemetryPayload(
            host="example.com",
            method="POST",
            path="/api/chat",
            status_code=500,
        )

        with patch("cencori.telemetry.httpx.AsyncClient") as mock_client_cls:
            mock_client = MagicMock()
            async_post = AsyncMock(side_effect=RuntimeError("network down"))
            mock_client.post = async_post
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            asyncio.run(client.telemetry.async_report_web_request(payload))
