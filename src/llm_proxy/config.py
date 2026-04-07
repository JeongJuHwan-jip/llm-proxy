"""Config file loading, validation, and header template resolution."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Patterns that suggest a hardcoded secret value (warn the user)
_SECRET_PATTERNS = re.compile(
    r"(Bearer\s+[A-Za-z0-9+/=._-]{8,}|sk-[A-Za-z0-9]{20,}|key-[A-Za-z0-9]{10,})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class FailoverConfig(BaseModel):
    max_retries: int = Field(default=2, ge=0)
    circuit_breaker_threshold: int = Field(default=3, ge=1)
    circuit_breaker_cooldown: int = Field(default=60, ge=0)
    routing_strategy: str = "priority"  # "priority" | "latency"
    default_model: str = "gpt-4"        # model name sent upstream when routing model is selected

    @field_validator("routing_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        if v not in ("priority", "latency"):
            raise ValueError("routing_strategy must be 'priority' or 'latency'")
        return v


class LoggingConfig(BaseModel):
    db_path: str = "./data/proxy.db"
    log_request_body: bool = False


class AuthConfig(BaseModel):
    api_keys: list[str] = Field(default_factory=list)


class RouteStepConfig(BaseModel):
    """One step in a route chain: the upstream server + model to request there.

    ``url``        — base URL of the upstream LLM server.
    ``model``      — model name sent to that server for this step.
    ``timeout_ms`` — per-step request timeout (ms).
    ``name``       — optional display name; derived from URL netloc when absent.
    ``headers``    — static/template headers added to this server's requests.

    Steps that share the same ``url`` share a single circuit-breaker state,
    so the same server can appear multiple times in a chain with different models:
      alpha/gpt-4  →  beta/llama-3  →  alpha/gpt-3.5  →  beta/mistral  → …
    """

    url: str
    model: str
    timeout_ms: int = Field(default=10000, gt=0)
    name: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"URL must start with http:// or https://: {v!r}")
        return v.rstrip("/")


class RouteConfig(BaseModel):
    """A named fallback chain exposed as a selectable model in /v1/models."""

    name: str
    chain: list[RouteStepConfig]

    @field_validator("chain")
    @classmethod
    def chain_not_empty(cls, v: list[RouteStepConfig]) -> list[RouteStepConfig]:
        if not v:
            raise ValueError("chain must contain at least one step")
        return v


class ProxyConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    failover: FailoverConfig = Field(default_factory=FailoverConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    auth: AuthConfig | None = None
    routing: list[RouteConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_routing_not_empty(self) -> "ProxyConfig":
        if not self.routing:
            raise ValueError("At least one routing entry must be configured")
        return self


# ---------------------------------------------------------------------------
# Template resolution
# ---------------------------------------------------------------------------

_TEMPLATE_UUID = re.compile(r"\{\{uuid\}\}", re.IGNORECASE)
_TEMPLATE_ENV = re.compile(r"\{\{env:([^}]+)\}\}")


def _resolve_value(value: str, *, warn_secrets: bool = True) -> str:
    """Resolve a single header value, replacing known templates."""
    # {{uuid}} → new UUID per call
    value = _TEMPLATE_UUID.sub(lambda _: str(uuid4()), value)

    # {{env:VAR}} → environment variable
    def _replace_env(m: re.Match) -> str:
        var_name = m.group(1).strip()
        resolved = os.environ.get(var_name)
        if resolved is None:
            logger.warning(
                "Environment variable %r referenced in config but not set; "
                "using empty string.",
                var_name,
            )
            return ""
        return resolved

    value = _TEMPLATE_ENV.sub(_replace_env, value)

    # Warn about apparent hardcoded secrets (only after template resolution
    # so we don't fire on env-var placeholders)
    if warn_secrets and _SECRET_PATTERNS.search(value):
        logger.warning(
            "Header value looks like a hardcoded secret. "
            "Consider using {{env:VARIABLE_NAME}} instead: %r",
            value[:30] + "..." if len(value) > 30 else value,
        )

    return value


def resolve_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a new dict with all template placeholders resolved.

    Called per-request so that {{uuid}} generates a fresh value each time.
    """
    return {k: _resolve_value(v) for k, v in headers.items()}


def resolve_api_keys(keys: list[str]) -> list[str]:
    """Resolve env-var templates in API key values (uuid not applicable here)."""
    return [_TEMPLATE_ENV.sub(
        lambda m: os.environ.get(m.group(1).strip(), ""), k
    ) for k in keys]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ProxyConfig:
    """Load and validate a YAML config file, returning a ProxyConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ValueError("Config file is empty")

    config = ProxyConfig.model_validate(raw)

    # Resolve auth keys eagerly (static values; no uuid template here)
    if config.auth and config.auth.api_keys:
        config.auth.api_keys = resolve_api_keys(config.auth.api_keys)
        # Remove empty strings that resulted from unset env vars
        config.auth.api_keys = [k for k in config.auth.api_keys if k]

    logger.info(
        "Loaded config: %d route(s), strategy=%s, db=%s",
        len(config.routing),
        config.failover.routing_strategy,
        config.logging.db_path,
    )
    return config
