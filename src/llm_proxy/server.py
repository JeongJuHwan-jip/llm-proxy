"""FastAPI application: proxy handler, dashboard, and status APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ProxyConfig
from .database import Database
from .models import AttemptLog, EndpointState, RequestLog
from .router import AllEndpointsFailedError, Router

logger = logging.getLogger(__name__)

# Special model ID that activates the full failover/circuit-breaker algorithm
ROUTING_MODEL_ID = "llm-proxy/router"

# Headers we copy from the incoming request to the upstream call
_FORWARD_HEADERS = {"content-type", "accept", "accept-encoding"}

# Headers that must NOT be forwarded upstream
_STRIP_HEADERS = {"host", "content-length", "transfer-encoding", "connection"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(config: ProxyConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    ``config`` may be injected directly (useful in tests).
    When running via CLI, pass config through app.state.
    """
    if config is None:
        # config will be set on app.state before startup by the CLI
        config = getattr(create_app, "_config", None)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg: ProxyConfig = app.state.config

        # HTTP client — shared connection pool
        limits = httpx.Limits(
            max_connections=200,
            max_keepalive_connections=50,
            keepalive_expiry=30,
        )
        client = httpx.AsyncClient(limits=limits, follow_redirects=False)

        router = Router(cfg)
        db = Database(cfg.logging.db_path)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db.init)

        app.state.client = client
        app.state.router = router
        app.state.db = db

        logger.info(
            "LLM Proxy started — %d endpoint(s) registered",
            len(cfg.endpoints),
        )
        yield

        await client.aclose()
        await loop.run_in_executor(None, db.close)

    app = FastAPI(
        title="LLM Proxy",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    if config is not None:
        app.state.config = config

    _register_routes(app)
    _mount_dashboard(app)
    return app


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def _require_api_key(request: Request) -> None:
    cfg: ProxyConfig = request.app.state.config
    if cfg.auth is None or not cfg.auth.api_keys:
        return  # auth not configured
    auth_header = request.headers.get("Authorization", "")
    key = auth_header.removeprefix("Bearer ").strip()
    if not key or key not in cfg.auth.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:

    # ------------------------------------------------------------------ proxy
    @app.post(
        "/v1/chat/completions",
        dependencies=[Depends(_require_api_key)],
    )
    async def chat_completions(request: Request) -> Response:
        return await _handle_proxy(request, "/chat/completions")

    # GET /v1/models — fetch from first reachable upstream, fall back to config-based list
    @app.get(
        "/v1/models",
        dependencies=[Depends(_require_api_key)],
    )
    async def list_models(request: Request) -> JSONResponse:
        return await _handle_list_models(request)

    # Catch-all for other /v1/* paths — forward with correct HTTP method
    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        dependencies=[Depends(_require_api_key)],
    )
    async def proxy_passthrough(request: Request, path: str) -> Response:
        return await _handle_passthrough(request, f"/{path}")

    # ----------------------------------------------------------------- status
    @app.get("/api/status")
    async def api_status(request: Request) -> JSONResponse:
        router: Router = request.app.state.router
        statuses = router.get_status()
        return JSONResponse([asdict(s) for s in statuses])

    @app.get("/api/requests")
    async def api_requests(
        request: Request,
        limit: int = 100,
        offset: int = 0,
    ) -> JSONResponse:
        db: Database = request.app.state.db
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None, lambda: db.get_recent_requests(limit=limit, offset=offset)
        )
        total = await loop.run_in_executor(None, db.get_total_count)
        return JSONResponse({"total": total, "rows": rows})

    @app.get("/api/stats")
    async def api_stats(request: Request) -> JSONResponse:
        db: Database = request.app.state.db
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, db.get_endpoint_stats)
        return JSONResponse(stats)

    # ---------------------------------------------------------------- root
    @app.get("/")
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard/index.html")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------

def _parse_model(model_id: str, router: Router, cfg: ProxyConfig) -> tuple[str | None, str]:
    """Interpret the model ID sent by the client.

    Returns (endpoint_name_or_None, actual_model_to_send_upstream).

    Cases:
      "llm-proxy/router"   → (None, cfg.failover.default_model)   full failover
      "alpha/gpt-4"        → ("alpha", "gpt-4")                   direct to alpha
      "gpt-4"              → (None, "gpt-4")                       full failover, legacy
    """
    if model_id == ROUTING_MODEL_ID:
        return None, cfg.failover.default_model

    # Check if it matches an "endpoint_name/model" pattern
    if "/" in model_id:
        ep_name, _, actual_model = model_id.partition("/")
        ep = router.get_endpoint_by_name(ep_name)
        if ep is not None:
            return ep_name, actual_model

    # Unknown format — treat as regular model name with full failover
    return None, model_id


async def _handle_proxy(request: Request, upstream_path: str) -> Response:
    client: httpx.AsyncClient = request.app.state.client
    router: Router = request.app.state.router
    db: Database = request.app.state.db
    cfg: ProxyConfig = request.app.state.config

    # Parse request body
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    raw_model: str = body.get("model", "unknown")
    target_ep_name, actual_model = _parse_model(raw_model, router, cfg)

    # Replace model in the body that will be forwarded upstream
    body = {**body, "model": actual_model}

    is_stream: bool = bool(body.get("stream", False))
    model: str = raw_model  # keep original for logging

    # Build headers to forward
    forward_headers: dict[str, str] = {
        k: v
        for k, v in request.headers.items()
        if k.lower() in _FORWARD_HEADERS and k.lower() not in _STRIP_HEADERS
    }

    t_start = time.monotonic()
    log_body = body if cfg.logging.log_request_body else None

    if is_stream:
        return await _handle_stream(
            client, router, db, cfg,
            upstream_path, body, forward_headers,
            model, log_body, t_start,
            target_ep_name=target_ep_name,
        )
    else:
        return await _handle_normal(
            client, router, db, cfg,
            upstream_path, body, forward_headers,
            model, log_body, t_start,
            target_ep_name=target_ep_name,
        )


async def _handle_normal(
    client: httpx.AsyncClient,
    router: Router,
    db: Database,
    cfg: ProxyConfig,
    path: str,
    body: dict,
    extra_headers: dict[str, str],
    model: str,
    log_body: dict | None,
    t_start: float,
    target_ep_name: str | None = None,
) -> Response:
    loop = asyncio.get_event_loop()
    try:
        if target_ep_name is not None:
            ep = router.get_endpoint_by_name(target_ep_name)
            if ep is None:
                raise HTTPException(status_code=400, detail=f"Unknown endpoint: {target_ep_name!r}")
            response, attempts = await router.execute_direct(client, ep, path, body, extra_headers)
        else:
            response, ep, attempts = await router.execute(client, path, body, extra_headers)
    except AllEndpointsFailedError as exc:
        total_ms = (time.monotonic() - t_start) * 1000
        log = RequestLog(
            timestamp=time.time(),
            model=model,
            selected_endpoint=None,
            attempts=exc.attempts,
            status="failure",
            total_latency_ms=total_ms,
            is_stream=False,
            request_body=log_body,
        )
        await loop.run_in_executor(None, db.insert_request_log, log)
        raise HTTPException(status_code=502, detail="All upstream endpoints failed")

    total_ms = (time.monotonic() - t_start) * 1000
    log = RequestLog(
        timestamp=time.time(),
        model=model,
        selected_endpoint=ep.name,
        attempts=attempts,
        status="success",
        total_latency_ms=total_ms,
        is_stream=False,
        request_body=log_body,
    )
    await loop.run_in_executor(None, db.insert_request_log, log)

    # Forward the upstream response back to the caller
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=response.headers.get("content-type", "application/json"),
    )


async def _handle_stream(
    client: httpx.AsyncClient,
    router: Router,
    db: Database,
    cfg: ProxyConfig,
    path: str,
    body: dict,
    extra_headers: dict[str, str],
    model: str,
    log_body: dict | None,
    t_start: float,
    target_ep_name: str | None = None,
) -> StreamingResponse:
    from .config import resolve_headers as _resolve

    loop = asyncio.get_event_loop()

    if target_ep_name is not None:
        _ep = router.get_endpoint_by_name(target_ep_name)
        if _ep is None:
            raise HTTPException(status_code=400, detail=f"Unknown endpoint: {target_ep_name!r}")
        candidates = [_ep]
        max_attempts = 1
    else:
        candidates = router.get_candidates(model)
        if not candidates:
            raise HTTPException(status_code=502, detail="No endpoints available")
        max_attempts = min(len(candidates), router._failover.max_retries + 1)

    attempts: list[AttemptLog] = []

    async def _try_stream() -> tuple[httpx.Response, object, list[AttemptLog]]:
        """Try each candidate until one starts streaming successfully."""
        import httpx as _httpx

        for ep in candidates[:max_attempts]:
            headers = _resolve(ep.headers)
            headers.update(extra_headers)
            url = f"{ep.url}{path}"
            timeout = ep.timeout_ms / 1000.0
            t0 = time.monotonic()
            try:
                # Use a streaming context — keep it open for the generator
                resp = await client.send(
                    client.build_request("POST", url, json=body, headers=headers),
                    stream=True,
                    timeout=timeout,
                )
                latency_ms = (time.monotonic() - t0) * 1000
                router.record_success(ep, latency_ms)
                attempts.append(
                    AttemptLog(
                        endpoint_name=ep.name,
                        latency_ms=latency_ms,
                        success=True,
                        is_timeout=False,
                    )
                )
                return resp, ep, attempts
            except _httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream timeout on %r: %s", ep.name, exc)
                router.record_failure(ep, is_timeout=True)
                attempts.append(
                    AttemptLog(
                        endpoint_name=ep.name,
                        latency_ms=latency_ms,
                        success=False,
                        is_timeout=True,
                        error_message=str(exc),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream error on %r: %s", ep.name, exc)
                router.record_failure(ep, is_timeout=False)
                attempts.append(
                    AttemptLog(
                        endpoint_name=ep.name,
                        latency_ms=latency_ms,
                        success=False,
                        is_timeout=False,
                        error_message=str(exc),
                    )
                )
        raise AllEndpointsFailedError(attempts)

    try:
        upstream_resp, winning_ep, final_attempts = await _try_stream()
    except AllEndpointsFailedError as exc:
        total_ms = (time.monotonic() - t_start) * 1000
        log = RequestLog(
            timestamp=time.time(),
            model=model,
            selected_endpoint=None,
            attempts=exc.attempts,
            status="failure",
            total_latency_ms=total_ms,
            is_stream=True,
            request_body=log_body,
        )
        await loop.run_in_executor(None, db.insert_request_log, log)
        raise HTTPException(status_code=502, detail="All upstream endpoints failed")

    async def byte_generator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_bytes():
                yield chunk
        finally:
            await upstream_resp.aclose()
            total_ms = (time.monotonic() - t_start) * 1000
            log = RequestLog(
                timestamp=time.time(),
                model=model,
                selected_endpoint=winning_ep.name,
                attempts=final_attempts,
                status="success",
                total_latency_ms=total_ms,
                is_stream=True,
                request_body=log_body,
            )
            # Fire-and-forget DB write from a thread
            asyncio.get_event_loop().run_in_executor(
                None, db.insert_request_log, log
            )

    return StreamingResponse(
        byte_generator(),
        status_code=upstream_resp.status_code,
        media_type=upstream_resp.headers.get("content-type", "text/event-stream"),
        headers={"X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# /v1/models handler
# ---------------------------------------------------------------------------

async def _handle_list_models(request: Request) -> JSONResponse:
    """Query ALL endpoints concurrently and aggregate models as {endpoint}/{model_id}.

    Always prepends the routing model (llm-proxy/router) as the first entry.
    Falls back to endpoint names if an upstream is unreachable.
    """
    import time as _time
    from .config import resolve_headers as _resolve

    client: httpx.AsyncClient = request.app.state.client
    router: Router = request.app.state.router

    async def _fetch(ep: "EndpointState") -> list[dict]:
        try:
            headers = _resolve(ep.headers)
            resp = await client.get(
                f"{ep.url}/models",
                headers=headers,
                timeout=ep.timeout_ms / 1000.0,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                return [
                    {**m, "id": f"{ep.name}/{m['id']}"}
                    for m in data
                    if isinstance(m.get("id"), str)
                ]
        except Exception:
            pass
        # Fallback: represent the endpoint itself as a model
        return [{
            "id": ep.name,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "llm-proxy",
        }]

    all_eps = router.all_endpoints()
    results = await asyncio.gather(*[_fetch(ep) for ep in all_eps])
    aggregated = [m for models in results for m in models]

    # Prepend the routing model
    routing_entry = {
        "id": ROUTING_MODEL_ID,
        "object": "model",
        "created": int(_time.time()),
        "owned_by": "llm-proxy",
        "description": "Uses priority/latency failover across all endpoints",
    }
    return JSONResponse({"object": "list", "data": [routing_entry, *aggregated]})


# ---------------------------------------------------------------------------
# Generic passthrough for non-chat /v1/* routes
# ---------------------------------------------------------------------------

async def _handle_passthrough(request: Request, upstream_path: str) -> Response:
    """Forward non-chat requests to the first reachable upstream as-is.
    No failover logic — just forward with the correct HTTP method.
    """
    client: httpx.AsyncClient = request.app.state.client
    router: Router = request.app.state.router

    candidates = router.get_candidates("*")
    if not candidates:
        raise HTTPException(status_code=502, detail="No endpoints available")

    ep = candidates[0]
    from .config import resolve_headers as _resolve
    headers = _resolve(ep.headers)
    url = f"{ep.url}{upstream_path}"

    # Read body only for methods that can carry one
    body_bytes = b""
    if request.method in ("POST", "PUT", "PATCH"):
        body_bytes = await request.body()

    try:
        upstream = await client.request(
            method=request.method,
            url=url,
            content=body_bytes or None,
            headers=headers,
            timeout=ep.timeout_ms / 1000.0,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# ---------------------------------------------------------------------------
# Dashboard static files
# ---------------------------------------------------------------------------

def _mount_dashboard(app: FastAPI) -> None:
    static_dir = Path(__file__).parent / "dashboard" / "static"
    if static_dir.exists():
        app.mount(
            "/dashboard",
            StaticFiles(directory=str(static_dir), html=True),
            name="dashboard",
        )
    else:
        logger.warning("Dashboard static directory not found: %s", static_dir)
