"""Tests for context-window-aware routing.

Verifies that per-(endpoint, model) ``max_context_tokens`` settings filter out
route steps whose model can't fit the estimated request, so a smaller fallback
model never silently truncates / stalls on an over-long prompt.

Limits are configured once globally (``ModelSettingConfig`` /
``Router.set_model_settings``) — never on individual route steps — so the same
model can never end up with two different limits in two different routes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llm_proxy.config import (
    EndpointConfig,
    FailoverConfig,
    LoggingConfig,
    ModelSettingConfig,
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
# Helpers — two-tier setup with a 1M and a 256K model
# ---------------------------------------------------------------------------


def _two_tier_config() -> ProxyConfig:
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
                    RouteStepConfig(endpoint="big", model="big-1m"),
                    RouteStepConfig(endpoint="small", model="small-256k"),
                ],
            ),
        ],
    )


def _two_tier_router() -> Router:
    """Router with the standard 1M / 256K model limits applied globally."""
    router = Router(_two_tier_config())
    router.set_model_settings([
        ModelSettingConfig(endpoint="big", model="big-1m", max_context_tokens=1_000_000),
        ModelSettingConfig(endpoint="small", model="small-256k", max_context_tokens=256_000),
    ])
    return router


# ---------------------------------------------------------------------------
# Router.filter_by_context
# ---------------------------------------------------------------------------


class TestFilterByContext:

    def test_step_kept_when_fits(self):
        router = _two_tier_router()
        steps = router.get_route("ctx-route")
        body = {"messages": [{"role": "user", "content": "hi"}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big", "small"]

    def test_small_step_dropped_when_request_too_big(self):
        router = _two_tier_router()
        steps = router.get_route("ctx-route")
        # ~300K tokens (1.2M chars / 4) → fits in 1M big, exceeds 256K small
        body = {"messages": [{"role": "user", "content": "x" * 1_200_000}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big"]

    def test_all_steps_dropped_when_too_big_for_everyone(self):
        router = _two_tier_router()
        steps = router.get_route("ctx-route")
        # ~5M tokens — exceeds both
        body = {"messages": [{"role": "user", "content": "x" * 20_000_000}]}
        eligible = router.filter_by_context(steps, body)
        assert eligible == []

    def test_model_with_no_setting_is_unbounded(self):
        """A (endpoint, model) pair without an entry is treated as unbounded."""
        router = _two_tier_router()
        # Drop the small model's limit by re-applying without it
        router.set_model_settings([
            ModelSettingConfig(
                endpoint="big", model="big-1m", max_context_tokens=1_000_000,
            ),
        ])
        steps = router.get_route("ctx-route")
        body = {"messages": [{"role": "user", "content": "x" * 20_000_000}]}
        eligible = router.filter_by_context(steps, body)
        # big is over its limit, small is now unbounded
        assert [s.endpoint.name for s in eligible] == ["small"]

    def test_max_tokens_output_budget_counted(self):
        """A request that fits inputs but blows the budget once max_tokens is
        added should be skipped."""
        router = _two_tier_router()
        steps = router.get_route("ctx-route")
        # Inputs ~ 200K tokens; max_tokens=100K → total 300K, exceeds 256K small
        body = {
            "messages": [{"role": "user", "content": "x" * 800_000}],
            "max_tokens": 100_000,
        }
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big"]

    def test_empty_settings_means_no_filtering(self):
        """Without any model settings, no step is filtered regardless of size."""
        router = Router(_two_tier_config())  # no set_model_settings called
        steps = router.get_route("ctx-route")
        body = {"messages": [{"role": "user", "content": "x" * 20_000_000}]}
        eligible = router.filter_by_context(steps, body)
        assert [s.endpoint.name for s in eligible] == ["big", "small"]


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
        router = _two_tier_router()

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
        router = _two_tier_router()
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
        router = _two_tier_router()

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
        router = _two_tier_router()
        small = router.get_endpoint_by_name("small")
        client = _mock_client(AsyncMock(return_value=MagicMock(status_code=200)))

        step = RouteStep(small, "small-256k", 10000)
        body = {"messages": [{"role": "user", "content": "x" * 2_000_000}]}

        # is_direct=True → context filter skipped, request goes through
        resp, winner, attempts = await router.execute(
            client, "/chat/completions", body, {},
            [step], "small-256k", is_direct=True,
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Single source of truth — same model in two routes shares one limit
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruth:
    """The whole point of the refactor: per-model settings live globally,
    so the same (endpoint, model) pair has the same context limit no matter
    which route uses it."""

    def test_same_model_in_two_routes_has_one_limit(self):
        cfg = _two_tier_config()
        cfg.routing.append(RouteConfig(
            name="other-route",
            chain=[RouteStepConfig(endpoint="small", model="small-256k")],
        ))
        router = Router(cfg)
        router.set_model_settings([
            ModelSettingConfig(
                endpoint="small", model="small-256k", max_context_tokens=256_000,
            ),
        ])

        # A long request hits the global limit on both routes equally.
        body = {"messages": [{"role": "user", "content": "x" * 2_000_000}]}
        for route_name in ("ctx-route", "other-route"):
            eligible = router.filter_by_context(router.get_route(route_name), body)
            assert all(s.endpoint.name != "small" for s in eligible), route_name


# ---------------------------------------------------------------------------
# Settings.json round-trip
# ---------------------------------------------------------------------------


class TestSettingsRoundTrip:

    def test_model_settings_loaded_from_settings_json(self, tmp_path):
        from llm_proxy.config import load_settings_file
        import json

        spath = tmp_path / "settings.json"
        spath.write_text(json.dumps({
            "routes": [],
            "model_settings": [
                {"endpoint": "big", "model": "big-1m", "max_context_tokens": 1_000_000},
                {"endpoint": "small", "model": "small-256k", "max_context_tokens": 256_000},
            ],
        }))
        sdata = load_settings_file(spath)
        assert len(sdata.model_settings) == 2
        assert sdata.model_settings[0].endpoint == "big"
        assert sdata.model_settings[0].max_context_tokens == 1_000_000

    def test_invalid_max_context_tokens_rejected(self, tmp_path):
        from llm_proxy.config import load_settings_file
        import json

        spath = tmp_path / "settings.json"
        spath.write_text(json.dumps({
            "routes": [],
            "model_settings": [
                {"endpoint": "a", "model": "m", "max_context_tokens": 0},
            ],
        }))
        with pytest.raises(Exception):
            load_settings_file(spath)

    def test_chain_step_no_longer_accepts_max_context_tokens(self):
        """Regression guard — context limits must not live on RouteStepConfig.

        If anyone reintroduces the field on the step the same model could end
        up with conflicting limits in different routes, which is exactly what
        the per-model refactor is designed to prevent."""
        # Pydantic-2: extra fields are ignored by default, so we explicitly
        # check that it doesn't end up as an attribute.
        step = RouteStepConfig(
            endpoint="a", model="m",
            max_context_tokens=100_000,  # type: ignore[call-arg]
        )
        assert not hasattr(step, "max_context_tokens")

    def test_router_get_model_settings_reflects_set(self):
        router = _two_tier_router()
        out = router.get_model_settings()
        assert {(e["endpoint"], e["model"]): e["max_context_tokens"] for e in out} == {
            ("big", "big-1m"): 1_000_000,
            ("small", "small-256k"): 256_000,
        }
