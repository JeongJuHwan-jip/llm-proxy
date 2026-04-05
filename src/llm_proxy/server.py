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
from .models import AttemptLog, RequestLog
from .router import AllEndpointsFailedError, Router

logger = logging.getLogger(__name__)

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

    # Catch-all for other /v1/* paths (e.g. /v1/models) — forward as-is
    @app.api_route(
        "/v1/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
        dependencies=[Depends(_require_api_key)],
    )
    async def proxy_passthrough(request: Request, path: str) -> Response:
        return await _handle_proxy(request, f"/{path}")

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

    is_stream: bool = bool(body.get("stream", False))
    model: str = body.get("model", "unknown")

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
        )
    else:
        return await _handle_normal(
            client, router, db, cfg,
            upstream_path, body, forward_headers,
            model, log_body, t_start,
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
) -> Response:
    loop = asyncio.get_event_loop()
    try:
        response, ep, attempts = await router.execute(
            client, path, body, extra_headers
        )
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
) -> StreamingResponse:
    from .config import resolve_headers as _resolve

    loop = asyncio.get_event_loop()
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
