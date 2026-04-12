"""Comprehensive failover and HTTP error handling tests.

Covers:
  - _should_failover() for all relevant status codes
  - router.execute() failover on httpx exceptions (Timeout, Connect, Read, Pool)
  - router.execute() failover on retriable HTTP status codes (5xx, 429, 408)
  - router.execute() NO failover on client errors (400, 401, 403, 404, 422)
  - Mixed error chains (timeout → 503 → success)
  - All-fail scenarios
  - max_retries limiting
  - Direct request behaviour (circuit breaker bypass, 5xx passthrough)
  - Circuit breaker state transitions during execute
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llm_proxy.config import (
    EndpointConfig,
    FailoverConfig,
    LoggingConfig,
    ProxyConfig,
    ProxyServerConfig,
    RouteConfig,
    RouteStepConfig,
)
from llm_proxy.models import RouteStep
from llm_proxy.router import AllEndpointsFailedError, Router, _should_failover


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_ep_config() -> ProxyConfig:
    """Config with 3 endpoints and a route chain alpha -> beta -> gamma."""
    return ProxyConfig(
        proxy=ProxyServerConfig(host="127.0.0.1", port=8000),
        endpoints=[
            EndpointConfig(name="alpha", url="https://alpha.example.com/v1", timeout_ms=2000),
            EndpointConfig(name="beta", url="https://beta.example.com/v1", timeout_ms=3000),
            EndpointConfig(name="gamma", url="https://gamma.example.com/v1", timeout_ms=5000),
        ],
        failover=FailoverConfig(
            max_retries=3,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            routing_strategy="priority",
        ),
        logging=LoggingConfig(db_path=":memory:", log_request_body=False),
        routing=[
            RouteConfig(
                name="test-route",
                chain=[
                    RouteStepConfig(endpoint="alpha", model="gpt-4"),
                    RouteStepConfig(endpoint="beta", model="gpt-4"),
                    RouteStepConfig(endpoint="gamma", model="gpt-4"),
                ],
            ),
        ],
    )


def _mock_client(side_effect):
    """Create an AsyncMock httpx client with a custom post handler."""
    client = AsyncMock()
    client.post = side_effect if callable(side_effect) else AsyncMock(side_effect=side_effect)
    return client


# ═══════════════════════════════════════════════════════════════════════════════
#  _should_failover
# ═══════════════════════════════════════════════════════════════════════════════


class TestShouldFailover:
    """Verify which HTTP status codes trigger failover."""

    @pytest.mark.parametrize(
        "code, expected",
        [
            # 2xx — success, never failover
            (200, False),
            (201, False),
            # 4xx client errors — don't failover (request is broken, not the server)
            (400, False),
            (401, False),
            (403, False),
            (404, False),
            (422, False),
            # 408 Request Timeout — server timed out, worth retrying elsewhere
            (408, True),
            # 429 Rate Limited — server overloaded, try another
            (429, True),
            # 5xx server errors — always failover
            (500, True),
            (502, True),
            (503, True),
            (504, True),
            (520, True),  # Cloudflare-style errors
            (529, True),  # site overloaded
        ],
    )
    def test_status_code(self, code: int, expected: bool):
        assert _should_failover(code) is expected


# ═══════════════════════════════════════════════════════════════════════════════
#  router.execute() — httpx exception failover
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteHttpxExceptions:
    """Failover should trigger on all httpx transport errors."""

    async def test_failover_on_timeout_exception(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.TimeoutException("read timed out")
            return ok

        steps = router.get_route("test-route")
        resp, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert resp.status_code == 200
        assert winner.endpoint.name == "beta"
        assert len(attempts) == 2
        assert attempts[0].is_timeout is True
        assert attempts[0].success is False
        assert attempts[1].success is True

    async def test_failover_on_connect_timeout(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ConnectTimeout("connect timed out")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"
        assert attempts[0].is_timeout is True  # ConnectTimeout is a TimeoutException

    async def test_failover_on_read_timeout(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ReadTimeout("read timed out")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"
        assert attempts[0].is_timeout is True

    async def test_failover_on_pool_timeout(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.PoolTimeout("pool exhausted")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"
        assert attempts[0].is_timeout is True

    async def test_failover_on_connect_error(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ConnectError("connection refused")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"
        assert attempts[0].is_timeout is False
        assert "connection refused" in attempts[0].error_message

    async def test_failover_on_read_error(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ReadError("connection reset by peer")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"
        assert attempts[0].is_timeout is False

    async def test_failover_on_remote_protocol_error(self, three_ep_config):
        router = Router(three_ep_config)
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.RemoteProtocolError("malformed HTTP")
            return ok

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "beta"


# ═══════════════════════════════════════════════════════════════════════════════
#  router.execute() — HTTP status code failover
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecuteStatusCodeFailover:
    """Failover should trigger on 5xx, 429, and 408 — but NOT on other 4xx."""

    @pytest.mark.parametrize("status_code", [500, 502, 503, 504, 429, 408])
    async def test_failover_on_retriable_status(self, three_ep_config, status_code):
        router = Router(three_ep_config)

        async def post(url, **kw):
            if "alpha" in url:
                return MagicMock(status_code=status_code)
            return MagicMock(status_code=200)

        steps = router.get_route("test-route")
        resp, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert resp.status_code == 200
        assert winner.endpoint.name == "beta"
        assert attempts[0].success is False
        assert f"HTTP {status_code}" in attempts[0].error_message
        assert attempts[1].success is True

    @pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
    async def test_no_failover_on_client_error(self, three_ep_config, status_code):
        """4xx client errors (except 408/429) return directly — no failover."""
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=status_code)))

        steps = router.get_route("test-route")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert resp.status_code == status_code
        assert winner.endpoint.name == "alpha"  # first endpoint, no failover
        assert len(attempts) == 1
        assert attempts[0].success is True


# ═══════════════════════════════════════════════════════════════════════════════
#  Mixed error scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestMixedErrors:

    async def test_timeout_then_503_then_success(self, three_ep_config):
        """alpha times out, beta returns 503, gamma succeeds."""
        router = Router(three_ep_config)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.TimeoutException("timeout")
            if "beta" in url:
                return MagicMock(status_code=503)
            return MagicMock(status_code=200)

        steps = router.get_route("test-route")
        resp, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "gamma"
        assert len(attempts) == 3
        assert attempts[0].is_timeout is True
        assert attempts[1].error_message == "HTTP 503"
        assert attempts[2].success is True

    async def test_connect_error_then_429_then_success(self, three_ep_config):
        router = Router(three_ep_config)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ConnectError("refused")
            if "beta" in url:
                return MagicMock(status_code=429)
            return MagicMock(status_code=200)

        steps = router.get_route("test-route")
        _, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        assert winner.endpoint.name == "gamma"
        assert len(attempts) == 3

    async def test_all_endpoints_timeout(self, three_ep_config):
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        assert len(exc_info.value.attempts) == 3
        assert all(a.is_timeout for a in exc_info.value.attempts)

    async def test_all_endpoints_return_5xx(self, three_ep_config):
        router = Router(three_ep_config)
        call_idx = 0

        async def post(url, **kw):
            nonlocal call_idx
            codes = [500, 502, 503]
            code = codes[min(call_idx, 2)]
            call_idx += 1
            return MagicMock(status_code=code)

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        assert len(exc_info.value.attempts) == 3
        assert all(not a.success for a in exc_info.value.attempts)

    async def test_mixed_connection_and_http_errors(self, three_ep_config):
        """alpha: ConnectError, beta: 500, gamma: ReadTimeout → all fail."""
        router = Router(three_ep_config)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.ConnectError("refused")
            if "beta" in url:
                return MagicMock(status_code=500)
            raise httpx.ReadTimeout("read timed out")

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        a = exc_info.value.attempts
        assert len(a) == 3
        assert a[0].is_timeout is False  # ConnectError
        assert a[1].is_timeout is False  # HTTP 500
        assert a[2].is_timeout is True   # ReadTimeout


# ═══════════════════════════════════════════════════════════════════════════════
#  max_retries limiting
# ═══════════════════════════════════════════════════════════════════════════════


class TestMaxRetries:

    async def test_max_retries_limits_attempts(self, three_ep_config):
        """With max_retries=1, only 2 attempts (initial + 1 retry)."""
        three_ep_config.failover.max_retries = 1
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        assert len(exc_info.value.attempts) == 2  # not 3

    async def test_max_retries_zero_means_single_attempt(self, three_ep_config):
        """max_retries=0 means no retry at all — only the first endpoint is tried."""
        three_ep_config.failover.max_retries = 0
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        assert len(exc_info.value.attempts) == 1

    async def test_max_retries_exceeds_endpoints(self, three_ep_config):
        """max_retries > number of endpoints → limited by endpoint count."""
        three_ep_config.failover.max_retries = 100
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))

        steps = router.get_route("test-route")
        with pytest.raises(AllEndpointsFailedError) as exc_info:
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
            )
        assert len(exc_info.value.attempts) == 3  # only 3 endpoints


# ═══════════════════════════════════════════════════════════════════════════════
#  Direct requests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectRequests:

    async def test_direct_skips_circuit_breaker_recording(self, three_ep_config):
        """Direct requests don't update circuit breaker stats."""
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))
        alpha = router.get_endpoint_by_name("alpha")

        step = RouteStep(alpha, "gpt-4")
        with pytest.raises(AllEndpointsFailedError):
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {},
                [step], "gpt-4", is_direct=True,
            )
        assert alpha.total_failures == 0
        assert alpha.total_requests == 0
        assert alpha.circuit_state == "closed"

    async def test_direct_success_no_stats(self, three_ep_config):
        """Direct success doesn't update endpoint stats."""
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=200)))
        alpha = router.get_endpoint_by_name("alpha")

        step = RouteStep(alpha, "gpt-4")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {},
            [step], "gpt-4", is_direct=True,
        )
        assert resp.status_code == 200
        assert alpha.total_requests == 0

    async def test_direct_passes_through_5xx(self, three_ep_config):
        """Direct request to an endpoint returning 503 should pass 503 through,
        not convert to AllEndpointsFailedError / 502."""
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=503)))
        alpha = router.get_endpoint_by_name("alpha")

        step = RouteStep(alpha, "gpt-4")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {},
            [step], "gpt-4", is_direct=True,
        )
        assert resp.status_code == 503  # passed through, not 502
        assert attempts[0].error_message == "HTTP 503"

    async def test_direct_passes_through_408(self, three_ep_config):
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=408)))
        alpha = router.get_endpoint_by_name("alpha")

        step = RouteStep(alpha, "gpt-4")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {},
            [step], "gpt-4", is_direct=True,
        )
        assert resp.status_code == 408

    async def test_direct_passes_through_429(self, three_ep_config):
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=429)))
        alpha = router.get_endpoint_by_name("alpha")

        step = RouteStep(alpha, "gpt-4")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {},
            [step], "gpt-4", is_direct=True,
        )
        assert resp.status_code == 429


# ═══════════════════════════════════════════════════════════════════════════════
#  Circuit breaker state during execute
# ═══════════════════════════════════════════════════════════════════════════════


class TestCircuitBreakerDuringExecute:

    async def test_circuit_opens_after_threshold_across_calls(self, three_ep_config):
        """Repeated failures across multiple execute calls should open the breaker."""
        three_ep_config.failover.circuit_breaker_threshold = 2
        three_ep_config.failover.max_retries = 0  # single attempt per call
        router = Router(three_ep_config)
        client = _mock_client(AsyncMock(side_effect=httpx.TimeoutException("timeout")))
        alpha = router.get_endpoint_by_name("alpha")

        steps_alpha_only = [router.get_route("test-route")[0]]  # only alpha step

        # First call: alpha fails, circuit still closed
        with pytest.raises(AllEndpointsFailedError):
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {},
                steps_alpha_only, "gpt-4",
            )
        assert alpha.circuit_state == "closed"
        assert alpha.consecutive_failures == 1

        # Second call: alpha fails again → circuit opens
        with pytest.raises(AllEndpointsFailedError):
            await router.execute(
                client, "/chat/completions", {"model": "gpt-4"}, {},
                steps_alpha_only, "gpt-4",
            )
        assert alpha.circuit_state == "open"

    async def test_open_circuit_skips_endpoint_in_route(self, three_ep_config):
        """Once alpha's circuit opens, the full route should skip it."""
        three_ep_config.failover.circuit_breaker_threshold = 1
        router = Router(three_ep_config)
        alpha = router.get_endpoint_by_name("alpha")

        # Open alpha's circuit
        router.record_failure(alpha, is_timeout=True)
        assert alpha.circuit_state == "open"

        ok = MagicMock(status_code=200)
        client = _mock_client(AsyncMock(return_value=ok))

        steps = router.get_route("test-route")
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )
        # alpha should be filtered out; beta serves the request
        assert winner.endpoint.name == "beta"
        assert len(attempts) == 1

    async def test_failure_stats_updated_on_failover(self, three_ep_config):
        """Each failed attempt during failover should update endpoint stats."""
        router = Router(three_ep_config)

        async def post(url, **kw):
            if "alpha" in url:
                raise httpx.TimeoutException("timeout")
            if "beta" in url:
                return MagicMock(status_code=503)
            return MagicMock(status_code=200)

        steps = router.get_route("test-route")
        await router.execute(
            _mock_client(post), "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4",
        )

        alpha = router.get_endpoint_by_name("alpha")
        beta = router.get_endpoint_by_name("beta")
        gamma = router.get_endpoint_by_name("gamma")

        assert alpha.total_failures == 1
        assert alpha.total_timeouts == 1
        assert beta.total_failures == 1
        assert beta.total_timeouts == 0
        assert gamma.total_failures == 0
        assert gamma.total_requests == 1
