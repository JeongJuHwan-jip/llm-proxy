"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from llm_proxy.config import EndpointConfig, FailoverConfig, LoggingConfig, ProxyConfig, ProxyServerConfig, RouteConfig, RouteStepConfig


@pytest.fixture
def minimal_config() -> ProxyConfig:
    """A minimal valid ProxyConfig with two endpoints."""
    return ProxyConfig(
        proxy=ProxyServerConfig(host="127.0.0.1", port=8000),
        endpoints=[
            EndpointConfig(
                name="alpha",
                url="https://alpha.example.com/v1",
                headers={"X-ID": "{{uuid}}"},
            ),
            EndpointConfig(
                name="beta",
                url="https://beta.example.com/v1",
            ),
        ],
        failover=FailoverConfig(
            max_retries=2,
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
                ],
            ),
        ],
    )
