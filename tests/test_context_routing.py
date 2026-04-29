"""Tests for context-window-aware routing.

Verifies that named-route steps declare a ``max_context_tokens`` budget and
that the router filters out steps whose budget cannot fit the estimated
request, so a smaller fallback model never silently truncates / stalls on an
over-long prompt.
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
from llm_proxy.router import (
    AllEndpointsFailedError,
    Router,
    estimate_request_tokens,
)


# ---------------------------------------------------------------------------
# estimate_request_tokens
# ---------------------------------------------------------------------------


class TestEstimateRequestTokens:
    """Coarse char/4 heuristic — verify it counts the right pieces."""

    def test_empty_body_is_zero(self):
        assert estimate_request_tokens({}) == 0

    def test_simple_string_message(self):
        body = {"messages": [{"role": "user", "content": "abcd"}]}
        # 4 chars / 4 = 1 token
        assert estimate_request_tokens(body) == 1

    def test_multiple_messages_summed(self):
        body = {
            "messages": [
                {"role": "system", "content": "x" * 400},
                {"role": "user", "content": "x" * 400},
            ]
        }
        assert estimate_request_tokens(body) == 200  # 800 / 4

    def test_max_tokens_added_to_estimate(self):
        body = {
            "messages": [{"role": "user", "content": "x" * 400}],
            "max_tokens": 50,
        }
        assert estimate_request_tokens(body) == 100 + 50

    def test_content_block_array_text_extracted(self):
        body = {"messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "x" * 200},
                {"type": "text", "text": "y" * 200},
            ],
        }]}
        assert estimate_request_tokens(body) == 100  # 400 / 4

    def test_tool_calls_arguments_counted(self):
        body = {"messages": [{
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "f", "arguments": "x" * 400}},
            ],
        }]}
        assert estimate_request_tokens(body) == 100

    def test_tools_definitions_counted(self):
        body = {
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "x" * 400}}],
        }
        # JSON-serialised tool def includes braces/keys, so >= 100 tokens
        assert estimate_request_tokens(body) >= 100

    def test_anthropic_top_level_system_string(self):
        body = {"system": "x" * 400, "messages": []}
        assert estimate_request_tokens(body) == 100

    def test_anthropic_top_level_system_blocks(self):
        body = {
            "system": [{"type": "text", "text": "x" * 200},
                       {"type": "text", "text": "y" * 200}],
            "messages": [],
        }
        assert estimate_request_tokens(body) == 100

    def test_negative_max_tokens_ignored(self):
        body = {"messages": [{"role": "user", "content": "abcd"}], "max_tokens": -5}
        assert estimate_request_tokens(body) == 1


# ---------------------------------------------------------------------------
# Router.filter_by_context
# ---------------------------------------------------------------------------


def _two_tier_config() -> ProxyConfig:
    """Big-model (1M) and small-model (256K) on two endpoints."""
    return ProxyConfig(
        proxy=ProxyServerConfig(host="127.0.0.1", port=8000),
        endpoints=[
            EndpointConfig(name="big", url="https://big.example.com/v1"),
            EndpointConfig(name="small", url="https://small.example.com/v1"),
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
                name="ctx-route",
                chain=[
                    RouteStepConfig(
                        endpoint="big", model="big-1m",
                        max_context_tokens=1_000_000,
                    ),
                    RouteStepConfig(
                        endpoint="small", model="small-256k",
                        max_context_tokens=256_000,
                    ),
                ],
            ),
        ],
    )


class TestFilterByContext:

    def test_step_kept_when_fits(self):
        router = Router(_two_tier_config())
        steps = router.get_route("ctx-route")
        body = {"messages": [{"role": "user", "content": "hi"}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big", "small"]

    def test_small_step_dropped_when_request_too_big(self):
        router = Router(_two_tier_config())
        steps = router.get_route("ctx-route")
        # ~300K tokens (1.2M chars / 4) → fits in 1M big, exceeds 256K small
        body = {"messages": [{"role": "user", "content": "x" * 1_200_000}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big"]

    def test_all_steps_dropped_when_too_big_for_everyone(self):
        router = Router(_two_tier_config())
        steps = router.get_route("ctx-route")
        # ~5M tokens — exceeds both
        body = {"messages": [{"role": "user", "content": "x" * 20_000_000}]}
        eligible = router.filter_by_context(steps, body)
        assert eligible == []

    def test_step_with_no_limit_always_kept(self):
        """A step with max_context_tokens=None is treated as unbounded."""
        cfg = _two_tier_config()
        cfg.routing[0].chain[1].max_context_tokens = None  # small now unbounded
        router = Router(cfg)
        steps = router.get_route("ctx-route")
        body = {"messages": [{"role": "user", "content": "x" * 20_000_000}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["small"]

    def test_max_tokens_output_budget_counted(self):
        """A request that fits inputs but blows the budget once max_tokens is
        added should be skipped."""
        router = Router(_two_tier_config())
        steps = router.get_route("ctx-route")
        # Inputs ~ 200K tokens; max_tokens=100K → total 300K, exceeds 256K small
        body = {
            "messages": [{"role": "user", "content": "x" * 800_000}],
            "max_tokens": 100_000,
        }
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big"]


# ---------------------------------------------------------------------------
# Router.execute() — context filter wired into failover loop
# ---------------------------------------------------------------------------


def _mock_client(side_effect):
    client = AsyncMock()
    client.post = side_effect if callable(side_effect) else AsyncMock(side_effect=side_effect)
    return client


class TestExecuteWithContextFilter:

    async def test_long_request_skips_small_and_fails_on_big(self):
        """Big endpoint times out, small would normally take over but its
        context window is too small → AllEndpointsFailedError instead of a
        silent stall on a truncated prompt."""
        router = Router(_two_tier_config())

        async def post(url, **kw):
            if "big" in url:
                raise httpx.TimeoutException("timeout")
            return MagicMock(status_code=200)  # would succeed if we got here

        body = {
            "model": "ctx-route",
            "messages": [{"role": "user", "content": "x" * 1_200_000}],
        }
        steps = router.get_route("ctx-route")
        with pytest.raises(AllEndpointsFailedError) as exc:
            await router.execute(
                _mock_client(post), "/chat/completions", body, {}, steps, "ctx-route",
            )
        # Only big was even attempted — small was filtered out before the loop
        attempts = exc.value.attempts
        assert len(attempts) == 1
        assert attempts[0].endpoint_name == "big"

    async def test_long_request_succeeds_on_big_when_big_works(self):
        router = Router(_two_tier_config())
        ok = MagicMock(status_code=200)

        async def post(url, **kw):
            return ok

        body = {
            "model": "ctx-route",
            "messages": [{"role": "user", "content": "x" * 1_200_000}],
        }
        steps = router.get_route("ctx-route")
        resp, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", body, {}, steps, "ctx-route",
        )
        assert winner.endpoint.name == "big"
        assert resp.status_code == 200

    async def test_short_request_falls_over_to_small(self):
        """For a small request both steps are eligible — failover still works."""
        router = Router(_two_tier_config())

        async def post(url, **kw):
            if "big" in url:
                raise httpx.TimeoutException("timeout")
            return MagicMock(status_code=200)

        body = {
            "model": "ctx-route",
            "messages": [{"role": "user", "content": "hi"}],
        }
        steps = router.get_route("ctx-route")
        resp, winner, attempts = await router.execute(
            _mock_client(post), "/chat/completions", body, {}, steps, "ctx-route",
        )
        assert winner.endpoint.name == "small"
        assert len(attempts) == 2

    async def test_direct_request_skips_context_filter(self):
        """Direct endpoint/model requests bypass routing logic entirely —
        including the context filter — so the user's explicit choice is honored."""
        router = Router(_two_tier_config())
        small = router.get_endpoint_by_name("small")
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=200)))

        # Use a manually built RouteStep so we can pretend small has a hard limit
        step = RouteStep(small, "small-256k", 10000, max_context_tokens=256_000)
        body = {"messages": [{"role": "user", "content": "x" * 2_000_000}]}

        # is_direct=True → context filter skipped, request goes through
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", body, {},
            [step], "small-256k", is_direct=True,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Settings.json round-trip
# ---------------------------------------------------------------------------


class TestSettingsRoundTrip:

    def test_max_context_tokens_loaded_from_settings_json(self, tmp_path):
        from llm_proxy.config import load_settings_file
        import json

        spath = tmp_path / "settings.json"
        spath.write_text(json.dumps({
            "routes": [{
                "name": "r",
                "chain": [
                    {"endpoint": "a", "model": "m1", "max_context_tokens": 1_000_000},
                    {"endpoint": "b", "model": "m2"},  # no field → None
                ],
            }],
        }))
        sdata = load_settings_file(spath)
        chain = sdata.routes[0].chain
        assert chain[0].max_context_tokens == 1_000_000
        assert chain[1].max_context_tokens is None

    def test_invalid_max_context_tokens_rejected(self, tmp_path):
        from llm_proxy.config import load_settings_file
        import json

        spath = tmp_path / "settings.json"
        spath.write_text(json.dumps({
            "routes": [{
                "name": "r",
                "chain": [{"endpoint": "a", "model": "m", "max_context_tokens": 0}],
            }],
        }))
        with pytest.raises(Exception):
            load_settings_file(spath)
