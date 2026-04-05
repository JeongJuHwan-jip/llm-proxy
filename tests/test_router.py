"""Tests for the Router class — circuit breaker and failover logic."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_proxy.models import EndpointState
from llm_proxy.router import AllEndpointsFailedError, Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ep(router: Router, name: str) -> EndpointState:
    for ep in router._table["*"]:
        if ep.name == name:
            return ep
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------


def test_candidates_priority_order(minimal_config):
    router = Router(minimal_config)
    candidates = router.get_candidates("gpt-4")
    assert [c.name for c in candidates] == ["alpha", "beta"]


def test_candidates_wildcard_fallback(minimal_config):
    router = Router(minimal_config)
    # No model-specific key exists — should fall back to "*"
    candidates = router.get_candidates("some-unknown-model")
    assert len(candidates) == 2


def test_candidates_latency_strategy(minimal_config):
    minimal_config.failover.routing_strategy = "latency"
    router = Router(minimal_config)

    # Give beta a very low latency so it ranks first
    beta = get_ep(router, "beta")
    beta.latency_samples.append(10.0)

    alpha = get_ep(router, "alpha")
    alpha.latency_samples.append(500.0)

    candidates = router.get_candidates("*")
    assert candidates[0].name == "beta"


# ---------------------------------------------------------------------------
# Circuit breaker state machine
# ---------------------------------------------------------------------------


def test_circuit_opens_after_threshold(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 3
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    assert alpha.circuit_state == "closed"
    for _ in range(3):
        router.record_failure(alpha, is_timeout=True)

    assert alpha.circuit_state == "open"


def test_circuit_excludes_open_endpoint(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 1
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_failure(alpha, is_timeout=False)
    assert alpha.circuit_state == "open"

    candidates = router.get_candidates("*")
    assert all(c.name != "alpha" for c in candidates)
    assert len(candidates) == 1
    assert candidates[0].name == "beta"


def test_circuit_transitions_to_half_open_after_cooldown(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 1
    minimal_config.failover.circuit_breaker_cooldown = 0  # instant
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_failure(alpha, is_timeout=True)
    assert alpha.circuit_state == "open"

    # Force open_since to be in the past
    alpha.open_since = time.monotonic() - 1

    candidates = router.get_candidates("*")
    # After cooldown expired, alpha should be HALF_OPEN and included
    assert alpha.circuit_state == "half_open"
    assert any(c.name == "alpha" for c in candidates)


def test_circuit_closes_on_success_from_half_open(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 1
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_failure(alpha, is_timeout=True)
    alpha.circuit_state = "half_open"

    router.record_success(alpha, latency_ms=100.0)
    assert alpha.circuit_state == "closed"
    assert alpha.consecutive_failures == 0


def test_circuit_reopens_on_failure_from_half_open(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 1
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    alpha.circuit_state = "half_open"
    router.record_failure(alpha, is_timeout=False)
    assert alpha.circuit_state == "open"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_record_success_updates_stats(minimal_config):
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_success(alpha, latency_ms=200.0)
    assert alpha.total_requests == 1
    assert alpha.total_failures == 0
    assert alpha.avg_latency_ms == pytest.approx(200.0)


def test_record_failure_updates_stats(minimal_config):
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_failure(alpha, is_timeout=True)
    assert alpha.total_requests == 1
    assert alpha.total_failures == 1
    assert alpha.total_timeouts == 1
    assert alpha.timeout_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


def test_get_status_returns_all_endpoints(minimal_config):
    router = Router(minimal_config)
    statuses = router.get_status()
    names = {s.name for s in statuses}
    assert names == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# execute (non-streaming) — uses mocked httpx client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_returns_on_first_success(minimal_config):
    router = Router(minimal_config)

    mock_response = MagicMock()
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    response, ep, attempts = await router.execute(
        mock_client, "/chat/completions", {"model": "gpt-4"}, {}
    )
    assert response.status_code == 200
    assert ep.name == "alpha"
    assert len(attempts) == 1
    assert attempts[0].success is True


@pytest.mark.asyncio
async def test_execute_failover_on_timeout(minimal_config):
    import httpx

    router = Router(minimal_config)
    minimal_config.failover.max_retries = 2

    mock_response = MagicMock()
    mock_response.status_code = 200

    call_count = 0

    async def mock_post(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "alpha" in url:
            raise httpx.TimeoutException("timed out")
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = mock_post

    response, ep, attempts = await router.execute(
        mock_client, "/chat/completions", {"model": "gpt-4"}, {}
    )
    assert ep.name == "beta"
    assert attempts[0].success is False
    assert attempts[0].is_timeout is True
    assert attempts[1].success is True


@pytest.mark.asyncio
async def test_execute_raises_when_all_fail(minimal_config):
    import httpx

    router = Router(minimal_config)

    async def mock_post(url, **kwargs):
        raise httpx.TimeoutException("timeout")

    mock_client = AsyncMock()
    mock_client.post = mock_post

    with pytest.raises(AllEndpointsFailedError) as exc_info:
        await router.execute(
            mock_client, "/chat/completions", {"model": "gpt-4"}, {}
        )
    assert len(exc_info.value.attempts) > 0
