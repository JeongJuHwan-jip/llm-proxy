"""End-to-end tests using real HTTP mock upstream servers.

Starts mock upstreams on real TCP ports (background threads), then routes
proxy requests through them to verify failover, streaming, circuit breaking,
and direct addressing over actual HTTP connections.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from llm_proxy.config import (
    EndpointConfig,
    FailoverConfig,
    LoggingConfig,
    ProxyConfig,
    ProxyServerConfig,
    RouteConfig,
    RouteStepConfig,
)
from llm_proxy.server import create_app
from mock_upstream import MockServer, create_mock_upstream, get_free_port


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_servers():
    """Start three mock upstreams: timeout, error(503), and ok."""
    configs = [
        ("alpha", "timeout", {"delay": 30.0}),
        ("beta", "error", {"status_code": 503}),
        ("gamma", "ok", {}),
    ]
    servers: dict[str, MockServer] = {}
    for name, behavior, kwargs in configs:
        port = get_free_port()
        app = create_mock_upstream(behavior=behavior, name=name, **kwargs)
        server = MockServer(app, port)
        server.start()
        servers[name] = server

    yield servers

    for s in servers.values():
        s.stop()


@pytest.fixture
def proxy_config(mock_servers, tmp_path) -> ProxyConfig:
    """Proxy config pointing at the three mock upstreams.

    Uses tmp_path for db_path so that settings_path resolves to a non-existent
    file inside the temp dir — preventing a stray settings.json in the CWD
    from overriding inline routing.
    """
    return ProxyConfig(
        proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
        endpoints=[
            EndpointConfig(name="alpha", url=mock_servers["alpha"].url),
            EndpointConfig(name="beta", url=mock_servers["beta"].url),
            EndpointConfig(name="gamma", url=mock_servers["gamma"].url),
        ],
        failover=FailoverConfig(
            max_retries=3,
            circuit_breaker_threshold=10,
            circuit_breaker_cooldown=60,
        ),
        logging=LoggingConfig(db_path=str(tmp_path / "test.db"), log_request_body=True),
        routing=[
            RouteConfig(
                name="test-route",
                chain=[
                    RouteStepConfig(endpoint="alpha", model="mock-model", timeout_ms=1000),
                    RouteStepConfig(endpoint="beta", model="mock-model", timeout_ms=5000),
                    RouteStepConfig(endpoint="gamma", model="mock-model", timeout_ms=5000),
                ],
            ),
        ],
    )


def _chat_body(model: str = "test-route", stream: bool = False) -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        **({"stream": True} if stream else {}),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestE2EFailover:
    """Verify failover across real HTTP connections."""

    def test_failover_timeout_then_503_then_success(self, proxy_config):
        """alpha hangs (timeout), beta returns 503, gamma succeeds."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["x-served-by"] == "gamma"

    def test_request_log_records_all_attempts(self, proxy_config):
        """The request log should show 3 attempts: timeout, 503, success."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/v1/chat/completions", json=_chat_body())
            log_resp = client.get("/api/requests?limit=1")

        assert log_resp.status_code == 200
        data = log_resp.json()
        assert data["total"] >= 1

        row = data["rows"][0]
        assert row["status"] == "success"
        assert row["selected_endpoint"] == "gamma"
        assert len(row["attempts"]) == 3
        assert row["attempts"][0]["is_timeout"] is True        # alpha timed out
        assert row["attempts"][1]["success"] is False           # beta 503
        assert "503" in row["attempts"][1].get("error", "")
        assert row["attempts"][2]["success"] is True            # gamma ok

    def test_streaming_failover(self, proxy_config):
        """Streaming requests should also failover correctly."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body(stream=True))

        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct
        body = resp.text
        assert "data:" in body
        assert "gamma" in body

    def test_streaming_midcut_falls_over_to_next(self, mock_servers, tmp_path):
        """If the first upstream cuts mid-SSE, byte_generator should pull
        the next endpoint and continue streaming so the client receives content
        from the recovering endpoint instead of a truncated response."""
        cut_port = get_free_port()
        cut_server = MockServer(
            create_mock_upstream(behavior="stream_cut", name="cutter"),
            cut_port,
        )
        cut_server.start()
        try:
            config = ProxyConfig(
                proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
                endpoints=[
                    EndpointConfig(name="cutter", url=cut_server.url),
                    EndpointConfig(name="gamma", url=mock_servers["gamma"].url),
                ],
                failover=FailoverConfig(max_retries=3),
                logging=LoggingConfig(db_path=str(tmp_path / "midcut_oai.db")),
                routing=[
                    RouteConfig(
                        name="cut-then-ok",
                        chain=[
                            RouteStepConfig(endpoint="cutter", model="mock-model", timeout_ms=5000),
                            RouteStepConfig(endpoint="gamma", model="mock-model", timeout_ms=5000),
                        ],
                    ),
                ],
            )
            app = create_app(config)
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json=_chat_body("cut-then-ok", stream=True),
                )

            assert resp.status_code == 200
            text = resp.text
            # Recovering endpoint's content must reach the client
            assert "gamma" in text
            # Stream terminator present
            assert "[DONE]" in text
        finally:
            cut_server.stop()

    def test_streaming_request_logged(self, proxy_config):
        """Streaming requests must also be persisted to the request log
        (regression guard: byte_generator's finally block previously fire-and-
        forgot run_in_executor without awaiting it)."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body(stream=True))
            assert resp.status_code == 200
            _ = resp.text  # drain body so generator's finally runs
            log_resp = client.get("/api/requests?limit=1")

        assert log_resp.status_code == 200
        data = log_resp.json()
        assert data["total"] >= 1
        row = data["rows"][0]
        assert row["status"] == "success"
        assert row["is_stream"] is True
        assert row["selected_endpoint"] == "gamma"


class TestE2EAllFail:
    """When every endpoint in the chain fails, the proxy should return 502."""

    def test_all_fail_returns_502(self, mock_servers, tmp_path):
        config = ProxyConfig(
            proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
            endpoints=[
                EndpointConfig(name="alpha", url=mock_servers["alpha"].url),
                EndpointConfig(name="beta", url=mock_servers["beta"].url),
            ],
            failover=FailoverConfig(max_retries=2),
            logging=LoggingConfig(db_path=str(tmp_path / "test.db")),
            routing=[
                RouteConfig(
                    name="doomed",
                    chain=[
                        RouteStepConfig(endpoint="alpha", model="mock-model", timeout_ms=500),
                        RouteStepConfig(endpoint="beta", model="mock-model", timeout_ms=5000),
                    ],
                ),
            ],
        )
        app = create_app(config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body("doomed"))

        assert resp.status_code == 502


class TestE2EDirectRequest:
    """Direct endpoint/model addressing bypasses routing and circuit breaker."""

    def test_direct_to_healthy_endpoint(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body("gamma/mock-model"),
            )
        assert resp.status_code == 200
        assert resp.json()["x-served-by"] == "gamma"

    def test_direct_to_error_endpoint_passes_through(self, proxy_config):
        """Direct to beta (503) should return 503, not 502."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json=_chat_body("beta/mock-model"),
            )
        assert resp.status_code == 503


class TestE2EEndpointStatus:
    """Verify endpoint health reporting after requests."""

    def test_status_after_failover(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            # Trigger a failover chain
            client.post("/v1/chat/completions", json=_chat_body())

            status_resp = client.get("/api/status")

        assert status_resp.status_code == 200
        statuses = {s["name"]: s for s in status_resp.json()}

        # alpha timed out → failure recorded
        assert statuses["alpha"]["total_failures"] >= 1
        assert statuses["alpha"]["total_timeouts"] >= 1
        # beta returned 503 → failure recorded
        assert statuses["beta"]["total_failures"] >= 1
        # gamma succeeded
        assert statuses["gamma"]["total_requests"] >= 1
        assert statuses["gamma"]["total_failures"] == 0

    def test_health_endpoint(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestE2EErrorServer:
    """Test with dedicated error-code mock servers."""

    @pytest.fixture
    def error_server_408(self):
        """Mock upstream returning 408 Request Timeout."""
        port = get_free_port()
        app = create_mock_upstream(behavior="error", name="error-408", status_code=408)
        server = MockServer(app, port)
        server.start()
        yield server
        server.stop()

    @pytest.fixture
    def error_server_429(self):
        """Mock upstream returning 429 Rate Limited."""
        port = get_free_port()
        app = create_mock_upstream(behavior="error", name="error-429", status_code=429)
        server = MockServer(app, port)
        server.start()
        yield server
        server.stop()

    @pytest.fixture
    def ok_server(self):
        port = get_free_port()
        app = create_mock_upstream(behavior="ok", name="healthy-ep")
        server = MockServer(app, port)
        server.start()
        yield server
        server.stop()

    def test_failover_on_408(self, error_server_408, ok_server, tmp_path):
        """HTTP 408 from first endpoint should trigger failover to second."""
        config = ProxyConfig(
            proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
            endpoints=[
                EndpointConfig(name="bad", url=error_server_408.url),
                EndpointConfig(name="good", url=ok_server.url),
            ],
            failover=FailoverConfig(max_retries=2),
            logging=LoggingConfig(db_path=str(tmp_path / "test.db")),
            routing=[
                RouteConfig(
                    name="r",
                    chain=[
                        RouteStepConfig(endpoint="bad", model="mock-model"),
                        RouteStepConfig(endpoint="good", model="mock-model"),
                    ],
                ),
            ],
        )
        app = create_app(config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body("r"))

        assert resp.status_code == 200
        assert resp.json()["x-served-by"] == "healthy-ep"

    def test_failover_on_429(self, error_server_429, ok_server, tmp_path):
        """HTTP 429 from first endpoint should trigger failover to second."""
        config = ProxyConfig(
            proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
            endpoints=[
                EndpointConfig(name="bad", url=error_server_429.url),
                EndpointConfig(name="good", url=ok_server.url),
            ],
            failover=FailoverConfig(max_retries=2),
            logging=LoggingConfig(db_path=str(tmp_path / "test.db")),
            routing=[
                RouteConfig(
                    name="r",
                    chain=[
                        RouteStepConfig(endpoint="bad", model="mock-model"),
                        RouteStepConfig(endpoint="good", model="mock-model"),
                    ],
                ),
            ],
        )
        app = create_app(config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/chat/completions", json=_chat_body("r"))

        assert resp.status_code == 200
        assert resp.json()["x-served-by"] == "healthy-ep"
