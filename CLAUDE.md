# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install
pip install -e ".[test]"

# Run all tests
pytest

# Run a single test file / class / function
pytest tests/test_failover.py
pytest tests/test_e2e.py::TestE2EFailover
pytest tests/test_router.py::test_circuit_opens_after_threshold -v

# Validate config
llm-proxy validate --config config.yaml

# Start server
llm-proxy start --config config.yaml
```

No linter or type-checker is configured.

## Architecture

**Request flow (OpenAI):** Client → `server.py` (`POST /v1/chat/completions`) → `_resolve_route()` determines direct vs named route → non-streaming goes through `router.execute()`, streaming through `_handle_stream()` (inlined in server.py, intentionally duplicated for performance) → response logged to SQLite via thread executor → SSE event published to dashboard subscribers.

**Request flow (Anthropic):** Client → `server.py` (`POST /v1/messages`) → `adapters/anthropic.py` `handle_anthropic_messages()` → `translate_request()` (Anthropic → OpenAI) → same `router.execute()` / failover infrastructure → `translate_response()` or `anthropic_sse_generator()` (OpenAI → Anthropic) → response logged → Client. The adapter reuses the existing routing, failover, and circuit breaker — only the wire format changes.

**Adapter pattern:** `adapters/anthropic.py` uses lazy imports (`from ..server import ...` inside function bodies) to avoid circular dependency with `server.py`. The streaming handler duplicates the step iteration loop from `_handle_stream()` (intentional for performance).

**Two-file config split:**
- `config.yaml` — static infrastructure: endpoints, auth, proxy server settings (human-edited)
- `settings.json` — dynamic routing: failover params + named route chains (managed exclusively by dashboard GUI, lives next to config.yaml via `resolve_settings_path()`)

**Config injection for multi-worker uvicorn:** `create_app._config = cfg` is set by CLI before uvicorn calls `create_app()` with no arguments (factory mode). Tests pass config directly as `create_app(config)`.

**Shared EndpointState:** The same endpoint appearing in multiple route steps shares one `EndpointState` instance. Circuit breaker state is per-endpoint, not per-step.

**Failover semantics (`_should_failover`):** 5xx, 429, 408 trigger failover to next step. Other 4xx do not (request itself is broken, will fail everywhere).

**Direct requests (`endpoint/model` format):** Bypass circuit breaker entirely — skip `filter_steps()`, skip `record_success/failure()`, pass through upstream error codes as-is instead of converting to 502.

## Test structure

- `test_config.py` — config loading, header template resolution
- `test_router.py` — Router unit tests with AsyncMock httpx client, circuit breaker state machine
- `test_failover.py` — comprehensive failover matrix (status codes, httpx exceptions, direct requests, max retries)
- `test_server.py` — FastAPI integration tests with `TestClient` + `respx` transport-layer mocking
- `test_e2e.py` — real HTTP end-to-end tests using `mock_upstream.py` (uvicorn in daemon threads on random ports)
- `test_anthropic_adapter.py` — Anthropic adapter unit tests: request/response translation, SSE buffer, SSE data parsing
- `test_anthropic_e2e.py` — Anthropic adapter e2e tests: non-streaming, streaming, direct, failover, error format, OpenAI regression

E2E tests must use `tmp_path` for `db_path` (not `:memory:`) to avoid `resolve_settings_path` picking up a stray `settings.json` from CWD.

**Test catalog ([tests/TESTS.ko.md](tests/TESTS.ko.md)):** Korean-language summary of every test and what it verifies. Whenever a test is added, renamed, removed, or its intent changes, update `tests/TESTS.ko.md` in the same change so the catalog stays in sync with the suite.
