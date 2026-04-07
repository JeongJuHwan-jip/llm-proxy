"""Tests for the Router class — circuit breaker and failover logic."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_proxy.models import EndpointState, RouteStep
from llm_proxy.router import AllEndpointsFailedError, Router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_ep(router: Router, name: str) -> EndpointState:
    """Return the shared EndpointState for the given endpoint name."""
    ep = router.get_endpoint_by_name(name)
    if ep is None:
        raise KeyError(name)
    return ep


def step_names(steps: list[RouteStep]) -> list[str]:
    return [s.endpoint.name for s in steps]


# ---------------------------------------------------------------------------
# Basic routing
# ---------------------------------------------------------------------------


def test_candidates_priority_order(minimal_config):
    router = Router(minimal_config)
    steps = router.filter_steps(router.get_route("gpt-4"))
    assert step_names(steps) == ["alpha", "beta"]


def test_candidates_wildcard_fallback(minimal_config):
    router = Router(minimal_config)
    # No named route exists — get_route falls back to "*"
    steps = router.filter_steps(router.get_route("some-unknown-model"))
    assert len(steps) == 2


def test_candidates_latency_strategy(minimal_config):
    minimal_config.failover.routing_strategy = "latency"
    router = Router(minimal_config)

    beta = get_ep(router, "beta")
    beta.latency_samples.append(10.0)

    alpha = get_ep(router, "alpha")
    alpha.latency_samples.append(500.0)

    steps = router.filter_steps(router.get_route("*"))
    assert steps[0].endpoint.name == "beta"


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

    steps = router.filter_steps(router.get_route("*"))
    assert all(s.endpoint.name != "alpha" for s in steps)
    assert len(steps) == 1
    assert steps[0].endpoint.name == "beta"


def test_circuit_transitions_to_half_open_after_cooldown(minimal_config):
    minimal_config.failover.circuit_breaker_threshold = 1
    minimal_config.failover.circuit_breaker_cooldown = 0  # instant
    router = Router(minimal_config)
    alpha = get_ep(router, "alpha")

    router.record_failure(alpha, is_timeout=True)
    assert alpha.circuit_state == "open"

    alpha.open_since = time.monotonic() - 1

    steps = router.filter_steps(router.get_route("*"))
    assert alpha.circuit_state == "half_open"
    assert any(s.endpoint.name == "alpha" for s in steps)


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

    steps = router.get_route("gpt-4")
    response, winning_step, attempts = await router.execute(
        mock_client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4"
    )
    assert response.status_code == 200
    assert winning_step.endpoint.name == "alpha"
    assert len(attempts) == 1
    assert attempts[0].success is True


@pytest.mark.asyncio
async def test_execute_failover_on_timeout(minimal_config):
    import httpx

    router = Router(minimal_config)
    minimal_config.failover.max_retries = 2

    mock_response = MagicMock()
    mock_response.status_code = 200

    async def mock_post(url, **kwargs):
        if "alpha" in url:
            raise httpx.TimeoutException("timed out")
        return mock_response

    mock_client = AsyncMock()
    mock_client.post = mock_post

    steps = router.get_route("gpt-4")
    response, winning_step, attempts = await router.execute(
        mock_client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4"
    )
    assert winning_step.endpoint.name == "beta"
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

    steps = router.get_route("gpt-4")
    with pytest.raises(AllEndpointsFailedError) as exc_info:
        await router.execute(
            mock_client, "/chat/completions", {"model": "gpt-4"}, {}, steps, "gpt-4"
        )
    assert len(exc_info.value.attempts) > 0
