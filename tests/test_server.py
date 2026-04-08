"""Integration tests for the FastAPI proxy server."""

from __future__ import annotations

import json

import httpx
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from llm_proxy.config import AuthConfig, EndpointConfig, FailoverConfig, LoggingConfig, ProxyConfig, ProxyServerConfig
from llm_proxy.server import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_no_auth() -> ProxyConfig:
    return ProxyConfig(
        proxy=ProxyServerConfig(host="127.0.0.1", port=8000),
        endpoints=[
            EndpointConfig(
                name="mock-ep",
                url="https://mock.example.com/v1",
                timeout_ms=5000,
            )
        ],
        failover=FailoverConfig(max_retries=0, circuit_breaker_threshold=3),
        logging=LoggingConfig(db_path=":memory:", log_request_body=False),
    )


@pytest.fixture
def config_with_auth(config_no_auth) -> ProxyConfig:
    config_no_auth.auth = AuthConfig(api_keys=["valid-key"])
    return config_no_auth


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_client(config: ProxyConfig) -> TestClient:
    """Create a synchronous TestClient for the app."""
    app = create_app(config)
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Health & basic routes
# ---------------------------------------------------------------------------


def test_health_endpoint(config_no_auth):
    with make_client(config_no_auth) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_root_redirects_to_dashboard(config_no_auth):
    with make_client(config_no_auth) as client:
        resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (301, 302, 307, 308)
    assert "/dashboard" in resp.headers["location"]


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------


def test_no_auth_required_when_not_configured(config_no_auth, respx_mock):
    respx_mock.post("https://mock.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"id": "chatcmpl-1", "choices": []}
        )
    )
    with make_client(config_no_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
    assert resp.status_code == 200


def test_auth_rejects_missing_key(config_with_auth):
    with make_client(config_with_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
    assert resp.status_code == 401


def test_auth_rejects_wrong_key(config_with_auth):
    with make_client(config_with_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-key"},
            json={"model": "gpt-4", "messages": []},
        )
    assert resp.status_code == 401


def test_auth_accepts_valid_key(config_with_auth, respx_mock):
    respx_mock.post("https://mock.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"id": "ok", "choices": []})
    )
    with make_client(config_with_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer valid-key"},
            json={"model": "gpt-4", "messages": []},
        )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Proxy behaviour
# ---------------------------------------------------------------------------


def test_proxy_forwards_response(config_no_auth, respx_mock):
    upstream_body = {"id": "chatcmpl-xyz", "choices": [{"message": {"content": "Hello"}}]}
    respx_mock.post("https://mock.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=upstream_body)
    )
    with make_client(config_no_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "chatcmpl-xyz"


def test_proxy_returns_502_when_all_fail(config_no_auth, respx_mock):
    respx_mock.post("https://mock.example.com/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("timeout")
    )
    with make_client(config_no_auth) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": []},
        )
    assert resp.status_code == 502


# ---------------------------------------------------------------------------
# Status / stats APIs
# ---------------------------------------------------------------------------


def test_api_status_returns_endpoint_list(config_no_auth):
    with make_client(config_no_auth) as client:
        resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert data[0]["name"] == "mock-ep"
    assert data[0]["circuit_state"] == "closed"


def test_api_requests_returns_empty_initially(config_no_auth):
    with make_client(config_no_auth) as client:
        resp = client.get("/api/requests")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["rows"] == []
