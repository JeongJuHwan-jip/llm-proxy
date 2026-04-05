# LLM Proxy

A lightweight local proxy server that routes requests across multiple OpenAI-compatible LLM API endpoints with automatic failover, circuit breaking, and a built-in dashboard.

## Features

- **Failover** — multiple endpoints per model; on timeout/error the next endpoint is tried automatically
- **Circuit breaker** — after N consecutive failures an endpoint is excluded for a cooldown period, then auto-recovered
- **Streaming** — full SSE / `stream: true` support
- **Custom headers** — per-endpoint static headers with `{{uuid}}` and `{{env:VAR}}` templates
- **Dashboard** — built-in web UI showing endpoint health, latency stats, and request history
- **SQLite logging** — WAL-mode, zero external dependencies
- **Auth** — optional API key protection
- **pip-installable** — single `pip install` + `llm-proxy start`

## Quick Start

```bash
pip install .
cp config.example.yaml config.yaml
# edit config.yaml to point at your endpoints
llm-proxy start --config config.yaml
```

The proxy listens on `http://0.0.0.0:8000` by default.  
Dashboard: `http://localhost:8000/dashboard/index.html`

## Config Reference

```yaml
server:
  host: "0.0.0.0"
  port: 8000

endpoints:
  - name: "llm-api-alpha"
    url: "https://internal-alpha.company.com/v1"
    timeout_ms: 5000          # per-request timeout in milliseconds
    priority: 1               # lower = tried first
    headers:
      X-Request-ID: "{{uuid}}"          # new UUID per request
      Authorization: "Bearer {{env:ALPHA_TOKEN}}"  # from env var

  - name: "llm-api-beta"
    url: "https://internal-beta.company.com/v1"
    timeout_ms: 8000
    priority: 2

failover:
  max_retries: 2                  # total attempts = max_retries + 1
  circuit_breaker_threshold: 3    # consecutive failures before OPEN
  circuit_breaker_cooldown: 60    # seconds before HALF_OPEN probe
  routing_strategy: "priority"    # "priority" (default) or "latency"

logging:
  db_path: "./data/proxy.db"
  log_request_body: false         # set true only for debugging (security risk)

# Optional proxy-level auth
# auth:
#   api_keys:
#     - "{{env:PROXY_API_KEY}}"
```

### Header templates

| Template | Resolved to |
|---|---|
| `{{uuid}}` | A fresh `uuid4()` string on every request |
| `{{env:VAR_NAME}}` | The value of the `VAR_NAME` environment variable |

## CLI

```
llm-proxy --help
llm-proxy start --config config.yaml [--host HOST] [--port PORT] [--workers N]
llm-proxy validate --config config.yaml
```

## Development

```bash
# Install in editable mode with test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with live reload (development)
llm-proxy start --config config.example.yaml
```

## Architecture

```
client → FastAPI (/v1/chat/completions)
           │
           ▼
        Router.execute()
           │   ┌──────────────────────────────────────┐
           ├──▶│ EndpointState (alpha)  [CLOSED]       │ ──▶ upstream
           │   │   circuit_breaker, latency_samples    │
           │   └──────────────────────────────────────┘
           │   ┌──────────────────────────────────────┐
           └──▶│ EndpointState (beta)   [CLOSED]       │ ──▶ upstream (failover)
               └──────────────────────────────────────┘
                           │
                           ▼
                       Database (SQLite WAL)
```

**Internal routing table**: `dict[model_name, list[EndpointState]]`  
The `"*"` wildcard key matches all models. Future releases will add per-model keys for fine-grained routing.

## Docker

```bash
docker build -t llm-proxy .
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml llm-proxy
```
