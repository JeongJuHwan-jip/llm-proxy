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
    # Should look like a UUID (36 chars with dashes)
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
        "routing": [{
            "name": "default",
            "chain": [
                {"url": "https://alpha.example.com/v1", "model": "gpt-4"},
            ],
        }]
    }
    path = _write_config(cfg_data)
    try:
        cfg = load_config(path)
        assert len(cfg.routing) == 1
        assert cfg.routing[0].chain[0].url == "https://alpha.example.com/v1"
        assert cfg.server.port == 8000  # default
    finally:
        path.unlink()


def test_load_full_config():
    cfg_data = {
        "server": {"host": "0.0.0.0", "port": 9000},
        "routing": [{
            "name": "ep1-route",
            "chain": [{
                "url": "https://ep1.example.com/v1",
                "model": "gpt-4",
                "timeout_ms": 3000,
                "name": "ep1",
                "headers": {"X-Key": "val"},
            }],
        }],
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
        assert cfg.server.port == 9000
        assert cfg.failover.routing_strategy == "latency"
        assert cfg.logging.log_request_body is True
        assert cfg.auth is not None
        assert "key123" in cfg.auth.api_keys
    finally:
        path.unlink()


def test_load_config_no_routing_raises():
    cfg_data = {"routing": []}
    path = _write_config(cfg_data)
    try:
        with pytest.raises(Exception):
            load_config(path)
    finally:
        path.unlink()


def test_load_config_invalid_url_raises():
    cfg_data = {
        "routing": [{
            "name": "bad",
            "chain": [{"url": "ftp://bad.example.com", "model": "gpt-4"}],
        }]
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
        "routing": [{
            "name": "ep",
            "chain": [{"url": "https://ep.example.com/v1", "model": "gpt-4"}],
        }],
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


def test_same_url_appears_twice_shares_circuit_breaker(minimal_config):
    """Two steps with the same URL should map to the same EndpointState."""
    from llm_proxy.config import RouteConfig, RouteStepConfig
    from llm_proxy.router import Router

    minimal_config.routing[0].chain.append(
        RouteStepConfig(
            url="https://alpha.example.com/v1",  # same URL as step 0
            model="gpt-3.5-turbo",
            name="alpha",
        )
    )
    router = Router(minimal_config)
    # There should still be only 2 unique servers (alpha, beta)
    assert len(router.all_endpoints()) == 2
    # Both alpha steps should share the same EndpointState object
    steps = router.get_route("default")
    assert steps[0].endpoint is steps[2].endpoint
