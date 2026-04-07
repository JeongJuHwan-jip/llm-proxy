"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from llm_proxy.config import (
    FailoverConfig,
    LoggingConfig,
    ProxyConfig,
    RouteConfig,
    RouteStepConfig,
    ServerConfig,
)


@pytest.fixture
def minimal_config() -> ProxyConfig:
    """A minimal valid ProxyConfig with two servers (alpha, beta)."""
    return ProxyConfig(
        server=ServerConfig(host="127.0.0.1", port=8000),
        routing=[
            RouteConfig(
                name="default",
                chain=[
                    RouteStepConfig(
                        url="https://alpha.example.com/v1",
                        model="gpt-4",
                        timeout_ms=5000,
                        name="alpha",
                        headers={"X-ID": "{{uuid}}"},
                    ),
                    RouteStepConfig(
                        url="https://beta.example.com/v1",
                        model="gpt-4",
                        timeout_ms=8000,
                        name="beta",
                    ),
                ],
            )
        ],
        failover=FailoverConfig(
            max_retries=2,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=60,
            routing_strategy="priority",
        ),
        logging=LoggingConfig(db_path=":memory:", log_request_body=False),
    )
