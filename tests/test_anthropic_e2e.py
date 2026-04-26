"""End-to-end tests for the Anthropic Messages API adapter.

Uses real HTTP mock upstream servers (same as test_e2e.py) to verify the
full request flow: Anthropic client → adapter → OpenAI upstream → adapter → client.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from llm_proxy.config import (
    AuthConfig,
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


def _anthropic_body(model: str = "test-route", stream: bool = False) -> dict:
    return {
        "model": model,
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hello"}],
        **({"stream": True} if stream else {}),
    }


# ---------------------------------------------------------------------------
# Tests — Non-streaming
# ---------------------------------------------------------------------------


class TestAnthropicNonStreaming:

    def test_basic_text_response(self, proxy_config):
        """Anthropic request → failover chain → Anthropic response format."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body())

        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["stop_reason"] == "end_turn"
        assert len(data["content"]) >= 1
        assert data["content"][0]["type"] == "text"
        assert "gamma" in data["content"][0]["text"]  # served by gamma
        assert data["usage"]["input_tokens"] >= 0
        assert data["usage"]["output_tokens"] >= 0

    def test_system_prompt_forwarded(self, proxy_config):
        """System prompt in Anthropic format is forwarded to upstream."""
        app = create_app(proxy_config)
        body = _anthropic_body()
        body["system"] = "You are a helpful assistant."
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=body)

        assert resp.status_code == 200
        assert resp.json()["type"] == "message"

    def test_response_has_msg_id_prefix(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body())

        assert resp.json()["id"].startswith("msg_")

    def test_failover_works(self, proxy_config):
        """alpha(timeout) → beta(503) → gamma(ok) should succeed."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body())

        assert resp.status_code == 200
        data = resp.json()
        # gamma is the one that succeeds
        assert "gamma" in data["content"][0]["text"]

    def test_request_logged(self, proxy_config):
        """Request should appear in the log."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            client.post("/v1/messages", json=_anthropic_body())
            log_resp = client.get("/api/requests?limit=1")

        assert log_resp.status_code == 200
        data = log_resp.json()
        assert data["total"] >= 1
        row = data["rows"][0]
        assert row["status"] == "success"


class TestAnthropicDirect:

    def test_direct_to_healthy_endpoint(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                json=_anthropic_body("gamma/mock-model"),
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert "gamma" in data["content"][0]["text"]

    def test_direct_to_error_returns_anthropic_error(self, proxy_config):
        """Direct to beta (503) should return Anthropic error format."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                json=_anthropic_body("beta/mock-model"),
            )
        assert resp.status_code == 503
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"


class TestAnthropicErrors:

    def test_unknown_model_returns_anthropic_404(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                json=_anthropic_body("nonexistent-route"),
            )
        assert resp.status_code == 404
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "not_found_error"

    def test_invalid_json_returns_anthropic_error(self, proxy_config):
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                content=b"not json",
                headers={"content-type": "application/json"},
            )
        assert resp.status_code == 400
        data = resp.json()
        assert data["type"] == "error"

    def test_all_fail_returns_anthropic_502(self, mock_servers, tmp_path):
        """When every endpoint in the chain fails → 502 in Anthropic format."""
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
            resp = client.post("/v1/messages", json=_anthropic_body("doomed"))

        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "api_error"


class TestAnthropicAuth:
    """The /v1/messages endpoint must enforce the same API-key middleware
    as /v1/chat/completions."""

    @pytest.fixture
    def auth_config(self, mock_servers, tmp_path) -> ProxyConfig:
        return ProxyConfig(
            proxy=ProxyServerConfig(host="127.0.0.1", port=9999),
            endpoints=[
                EndpointConfig(name="gamma", url=mock_servers["gamma"].url),
            ],
            failover=FailoverConfig(max_retries=0),
            logging=LoggingConfig(db_path=str(tmp_path / "test.db")),
            auth=AuthConfig(api_keys=["valid-key"]),
            routing=[
                RouteConfig(
                    name="r",
                    chain=[RouteStepConfig(endpoint="gamma", model="mock-model")],
                ),
            ],
        )

    def test_rejects_missing_key(self, auth_config):
        app = create_app(auth_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body("r"))
        assert resp.status_code == 401

    def test_rejects_wrong_key(self, auth_config):
        app = create_app(auth_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer wrong-key"},
                json=_anthropic_body("r"),
            )
        assert resp.status_code == 401

    def test_accepts_valid_key(self, auth_config):
        app = create_app(auth_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer valid-key"},
                json=_anthropic_body("r"),
            )
        assert resp.status_code == 200
        assert resp.json()["type"] == "message"



# ---------------------------------------------------------------------------
# Tests — Streaming
# ---------------------------------------------------------------------------


class TestAnthropicStreaming:

    def test_streaming_returns_sse(self, proxy_config):
        """Streaming request should return text/event-stream with Anthropic events."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))

        assert resp.status_code == 200
        ct = resp.headers.get("content-type", "")
        assert "text/event-stream" in ct

    def test_streaming_event_sequence(self, proxy_config):
        """Verify the Anthropic SSE events arrive in spec-compliant order."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))

        # Extract event types in arrival order
        events = [
            line.removeprefix("event: ").strip()
            for line in resp.text.split("\n")
            if line.startswith("event: ")
        ]
        assert events, "no SSE events found"

        # Anthropic spec ordering:
        #   message_start → (ping)? → content_block_start → content_block_delta+
        #   → content_block_stop → message_delta → message_stop
        assert events[0] == "message_start"
        assert events[-1] == "message_stop"
        assert events[-2] == "message_delta"
        # First content block event must be a start, before any delta
        first_cb_start = events.index("content_block_start")
        first_cb_delta = events.index("content_block_delta")
        first_cb_stop = events.index("content_block_stop")
        assert first_cb_start < first_cb_delta < first_cb_stop
        # message_delta must come after every content_block_stop
        assert first_cb_stop < events.index("message_delta")

    def test_streaming_contains_text(self, proxy_config):
        """The streamed text should contain gamma's response."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))

        body = resp.text
        assert "gamma" in body

    def test_streaming_request_logged(self, proxy_config):
        """Streaming requests must also be persisted to the request log
        (regression guard: the streaming finally-block previously fire-and-
        forgot run_in_executor without awaiting it)."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))
            # Drain the body so the wrapped_generator's finally block runs
            assert resp.status_code == 200
            _ = resp.text
            log_resp = client.get("/api/requests?limit=1")

        assert log_resp.status_code == 200
        data = log_resp.json()
        assert data["total"] >= 1
        row = data["rows"][0]
        assert row["status"] == "success"
        assert row["is_stream"] is True
        assert row["selected_endpoint"] == "gamma"

    def test_streaming_message_start_has_model(self, proxy_config):
        """message_start event should have the model field."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))

        body = resp.text
        # Find message_start data
        for line in body.split("\n"):
            if line.startswith("data:") and "message_start" in line:
                data = json.loads(line[5:].strip())
                assert data["message"]["model"] == "test-route"
                assert data["message"]["role"] == "assistant"
                break
        else:
            pytest.fail("message_start event not found")

    def test_streaming_failover(self, proxy_config):
        """Streaming with failover should succeed through gamma."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/v1/messages", json=_anthropic_body(stream=True))

        assert resp.status_code == 200
        assert "gamma" in resp.text

    def test_streaming_direct_to_healthy(self, proxy_config):
        """Direct streaming to gamma should work."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/messages",
                json=_anthropic_body("gamma/mock-model", stream=True),
            )

        assert resp.status_code == 200
        assert "event: message_start" in resp.text

    def test_streaming_all_fail_returns_error(self, mock_servers, tmp_path):
        """When all steps fail during streaming → Anthropic error response."""
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
            resp = client.post("/v1/messages", json=_anthropic_body("doomed", stream=True))

        assert resp.status_code == 502
        data = resp.json()
        assert data["type"] == "error"


# ---------------------------------------------------------------------------
# Tests — OpenAI endpoint still works (regression)
# ---------------------------------------------------------------------------


class TestOpenAIRegressionFromAnthropicTests:

    def test_openai_chat_still_works(self, proxy_config):
        """Ensure adding /v1/messages didn't break /v1/chat/completions."""
        app = create_app(proxy_config)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-route",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["x-served-by"] == "gamma"
