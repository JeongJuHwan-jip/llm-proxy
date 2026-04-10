# LLM Proxy

A lightweight local proxy server that routes requests across multiple OpenAI-compatible LLM API endpoints with automatic failover, circuit breaking, and a built-in dashboard.

## Features

- **Named route chains** — define fallback chains like `alpha/gpt-4 -> beta/llama-3 -> gamma/mistral-7b`; on timeout/error the next step is tried automatically
- **Direct addressing** — send to a specific endpoint with `endpoint_name/model_id` (bypasses routing & circuit breaker)
- **Circuit breaker** — after N consecutive failures an endpoint is excluded for a cooldown period, then auto-recovered via half-open probe
- **Streaming** — full SSE / `stream: true` support with per-chunk forwarding
- **Dashboard GUI** — real-time endpoint health monitoring, request log, and a drag-and-drop routing editor to configure failover + routes visually
- **Model discovery** — auto-detects available models from all endpoints at startup
- **Custom headers** — per-endpoint static headers with `{{uuid}}` and `{{env:VAR}}` templates
- **SQLite logging** — WAL-mode, zero external dependencies
- **Auth** — optional API key protection
- **pip-installable** — single `pip install` + `llm-proxy start`

## Getting Started

```bash
git clone https://github.com/JeongJuHwan-jip/llm-proxy.git
cd llm-proxy

# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
uv pip install -e ".[test]"

# Activate the virtual environment
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / Mac

# Copy and edit the example config
cp config.example.yaml config.yaml
# Edit config.yaml — set your endpoint URLs and headers

# Validate config
llm-proxy validate --config config.yaml

# Discover models from all endpoints
llm-proxy discover --config config.yaml

# Start the server
llm-proxy start --config config.yaml
```

The proxy listens on `http://0.0.0.0:8000` by default.  
Dashboard: `http://localhost:8000/dashboard/index.html`

## Dashboard

The dashboard has two tabs:

- **Monitor** — live endpoint health cards (circuit state, latency, failure rates) and a request history table, updated via SSE
- **Settings** — configure failover parameters and build routing chains with drag-and-drop

When you click **Apply Settings**, a `settings.json` file is created next to your config.yaml. No manual file editing needed — the dashboard manages it entirely.

## Config Reference

`config.yaml` defines your endpoints and infrastructure. Routing and failover are managed via the dashboard.

```yaml
proxy:
  host: "0.0.0.0"
  port: 8000
  header_priority: "config"   # "config" or "client" — who wins on header conflicts

endpoints:
  - name: "alpha"
    url: "https://alpha.company.com/v1"
    timeout_ms: 5000
    headers:
      X-Request-ID: "{{uuid}}"
      Authorization: "Bearer {{env:ALPHA_TOKEN}}"

  - name: "beta"
    url: "https://beta.company.com/v1"
    timeout_ms: 8000

logging:
  db_path: "./data/proxy.db"
  log_request_body: false

# auth:
#   api_keys:
#     - "{{env:PROXY_API_KEY}}"
```

### Header templates

| Template | Resolved to |
|---|---|
| `{{uuid}}` | A fresh `uuid4()` string on every request |
| `{{env:VAR_NAME}}` | The value of the `VAR_NAME` environment variable |

### Request routing

Clients set the `model` field in their request body:

| model value | Behavior |
|---|---|
| `alpha/gpt-4` | **Direct** — sends to endpoint `alpha` with model `gpt-4`. No circuit breaker or failover. |
| `best-available` | **Named route** — follows the failback chain defined in the dashboard (e.g. alpha/gpt-4 -> beta/llama-3 -> gamma/mistral). |

## CLI

```
llm-proxy --help
llm-proxy start    --config config.yaml [--host HOST] [--port PORT] [--workers N]
llm-proxy validate --config config.yaml
llm-proxy discover --config config.yaml [--snippet]
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Proxy with failover |
| GET | `/v1/models` | Merged model list (direct + named routes) |
| GET | `/api/status` | Endpoint health snapshots |
| GET | `/api/requests?limit=50` | Recent request log |
| GET | `/api/discovery` | Discovery results + diff |
| GET | `/api/config/settings` | Current failover + routing |
| PUT | `/api/config/settings` | Apply failover + routing (saves to settings.json) |
| POST | `/api/routing/reload` | Re-read settings.json without restart |
| GET | `/api/events` | SSE stream for dashboard |

## Local Testing with Mock Servers

```bash
# Start mock LLM servers
python scripts/mock_llm_server.py --port 8001 --name alpha --models gpt-4,claude-3
python scripts/mock_llm_server.py --port 8002 --name beta --models gpt-4,llama-3 --behavior flaky --fail-rate 0.5
python scripts/mock_llm_server.py --port 8003 --name gamma --models mistral-7b

# Mock server options:
#   --behavior ok|timeout|error|flaky|slow
#   --fail-rate 0.5          (for flaky)
#   --timeout-rate 0.3       (fraction of flaky failures that are timeouts)
#   --delay 5.0              (seconds for slow/timeout)
#   --latency-min 0.1 --latency-max 0.8   (random latency on success)

# Start proxy
llm-proxy start --config config.local-test.yaml
```

## Architecture

```
client -> FastAPI (/v1/chat/completions)
             |
             v
          Router.execute(steps=[...])
             |   +------------------------------------+
             +-->| EndpointState (alpha)  [CLOSED]     | --> upstream
             |   |   circuit breaker, latency samples  |
             |   +------------------------------------+
             |   +------------------------------------+
             +-->| EndpointState (beta)   [CLOSED]     | --> upstream (failover)
             |   +------------------------------------+
             |   +------------------------------------+
             +-->| EndpointState (gamma)  [CLOSED]     | --> upstream (failover)
                 +------------------------------------+
                             |
                             v
                     Database (SQLite WAL)
```

**Files:**
- `config.yaml` — endpoint definitions, auth, proxy settings (human-edited)
- `settings.json` — failover config + route chains (managed by dashboard, auto-generated)

## Development

```bash
# Run tests
pytest

# Run with local mock servers
llm-proxy start --config config.local-test.yaml --port 9000
```

## Docker

```bash
docker build -t llm-proxy .
docker run -p 8000:8000 -v $(pwd)/config.yaml:/app/config.yaml llm-proxy
```
