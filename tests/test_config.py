"""Tests for config loading and header template resolution."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from llm_proxy.config import (
    ProxyConfig,
    load_config,
    resolve_headers,
)


# ---------------------------------------------------------------------------
# resolve_headers
# ---------------------------------------------------------------------------


def test_uuid_template_generates_value():
    result = resolve_headers({"X-ID": "{{uuid}}"})
    assert "X-ID" in result
    val = result["X-ID"]
    assert len(val) == 36
    assert val.count("-") == 4


def test_uuid_template_unique_per_call():
    r1 = resolve_headers({"X-ID": "{{uuid}}"})
    r2 = resolve_headers({"X-ID": "{{uuid}}"})
    assert r1["X-ID"] != r2["X-ID"]


def test_env_template_resolved(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "supersecret")
    result = resolve_headers({"Authorization": "Bearer {{env:MY_SECRET}}"})
    assert result["Authorization"] == "Bearer supersecret"


def test_env_template_missing_var(monkeypatch, caplog):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    import logging
    with caplog.at_level(logging.WARNING):
        result = resolve_headers({"X-Key": "{{env:MISSING_VAR}}"})
    assert result["X-Key"] == ""
    assert "MISSING_VAR" in caplog.text


def test_static_header_unchanged():
    result = resolve_headers({"X-Fixed": "hello-world"})
    assert result["X-Fixed"] == "hello-world"


def test_multiple_templates_in_one_value(monkeypatch):
    monkeypatch.setenv("TENANT", "acme")
    result = resolve_headers({"X-Meta": "tenant={{env:TENANT}}"})
    assert result["X-Meta"] == "tenant=acme"


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def _write_config(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def test_load_minimal_config():
    cfg_data = {
        "endpoints": [
            {"name": "alpha", "url": "https://alpha.example.com/v1", }
        ]
    }
    path = _write_config(cfg_data)
    try:
        cfg = load_config(path)
        assert len(cfg.endpoints) == 1
        assert cfg.endpoints[0].name == "alpha"
        assert cfg.proxy.port == 8000  # default
    finally:
        path.unlink()


def test_load_full_config():
    cfg_data = {
        "proxy": {"host": "0.0.0.0", "port": 9000},
        "endpoints": [
            {
                "name": "ep1",
                "url": "https://ep1.example.com/v1",
                "timeout_ms": 3000,
                "headers": {"X-Key": "val"},
            }
        ],
        "failover": {
            "max_retries": 1,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_cooldown": 30,
            "routing_strategy": "latency",
        },
        "logging": {"db_path": "/tmp/test.db", "log_request_body": True},
        "auth": {"api_keys": ["key123"]},
    }
    path = _write_config(cfg_data)
    try:
        cfg = load_config(path)
        assert cfg.proxy.port == 9000
        assert cfg.failover.routing_strategy == "latency"
        assert cfg.logging.log_request_body is True
        assert cfg.auth is not None
        assert "key123" in cfg.auth.api_keys
    finally:
        path.unlink()


def test_load_config_no_endpoints_raises():
    cfg_data = {"endpoints": []}
    path = _write_config(cfg_data)
    try:
        with pytest.raises(Exception):
            load_config(path)
    finally:
        path.unlink()


def test_load_config_invalid_url_raises():
    cfg_data = {
        "endpoints": [{"name": "bad", "url": "ftp://bad.example.com", }]
    }
    path = _write_config(cfg_data)
    try:
        with pytest.raises(Exception, match="http"):
            load_config(path)
    finally:
        path.unlink()


def test_load_config_env_key(monkeypatch):
    monkeypatch.setenv("PROXY_KEY", "runtime-secret")
    cfg_data = {
        "endpoints": [
            {"name": "ep", "url": "https://ep.example.com/v1", }
        ],
        "auth": {"api_keys": ["{{env:PROXY_KEY}}"]},
    }
    path = _write_config(cfg_data)
    try:
        cfg = load_config(path)
        assert cfg.auth is not None
        assert "runtime-secret" in cfg.auth.api_keys
    finally:
        path.unlink()


def test_load_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_same_endpoint_appears_twice_in_chain(minimal_config):
    """The same endpoint can appear twice in a chain with different models."""
    from llm_proxy.config import RouteConfig, RouteStepConfig
    from llm_proxy.router import Router

    minimal_config.routing = [
        RouteConfig(
            name="multi-model",
            chain=[
                RouteStepConfig(endpoint="alpha", model="gpt-4"),
                RouteStepConfig(endpoint="beta", model="llama-3"),
                RouteStepConfig(endpoint="alpha", model="gpt-3.5-turbo"),  # alpha again
            ],
        )
    ]
    router = Router(minimal_config)
    steps = router.get_route("multi-model")
    assert len(steps) == 3
    assert steps[0].endpoint.name == "alpha"
    assert steps[0].model == "gpt-4"
    assert steps[2].endpoint.name == "alpha"
    assert steps[2].model == "gpt-3.5-turbo"
    # Same EndpointState object → shared circuit breaker
    assert steps[0].endpoint is steps[2].endpoint
