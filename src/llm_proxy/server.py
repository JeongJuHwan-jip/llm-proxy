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

from .config import ProxyConfig, load_routing_file, resolve_routing_file
from .database import Database
from .discovery import (
    DiscoveryResult,
    diff_discovery,
    load_snapshot,
    log_discovery_diff,
    log_first_discovery,
    run_discovery,
    save_snapshot,
)
from .models import AttemptLog, EndpointState, RequestLog, RouteStep
from .router import AllEndpointsFailedError, Router

logger = logging.getLogger(__name__)

# Special model ID that activates the full failover/circuit-breaker algorithm
ROUTING_MODEL_ID = "llm-proxy/router"

# Headers forwarded upstream — content-type is intentionally excluded:
# httpx sets it automatically when using json=body, avoiding conflicts.
_FORWARD_HEADERS = {"accept", "accept-encoding"}
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
    config_path_from_cli = getattr(create_app, "_config_path", None)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg: ProxyConfig = app.state.config
        config_path: Path | None = getattr(app.state, "config_path", None) or config_path_from_cli

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

        # ---- load routing file (overrides inline routing: in config) ----
        routing_path = resolve_routing_file(
            config_path or Path(cfg.logging.db_path),
            cfg.routing_file,
        )
        app.state.routing_path = routing_path
        if routing_path is not None and routing_path.exists():
            try:
                routes = load_routing_file(routing_path)
                router.reload_routing(routes)
                logger.info("Loaded routing from %s", routing_path)
            except Exception as exc:
                logger.warning("Could not load routing file %s: %s — using config routing", routing_path, exc)
        # ------------------------------------------------------------

        # ---- model discovery & change detection ----
        snapshot_path = Path(cfg.logging.db_path).parent / "model_snapshot.json"
        old_snapshot = load_snapshot(snapshot_path)

        discovery = await run_discovery(client, router.all_endpoints())
        router.update_routing_from_discovery(discovery.models)
        save_snapshot(snapshot_path, discovery)

        app.state.discovery = discovery
        if old_snapshot is None:
            log_first_discovery(discovery)
            app.state.discovery_diff = None
        else:
            routed = set(router.get_routed_models().keys())
            diff = diff_discovery(old_snapshot, discovery, routed)
            log_discovery_diff(diff)
            app.state.discovery_diff = diff
        # --------------------------------------------

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

    @app.post("/api/routing/reload")
    async def api_routing_reload(request: Request) -> JSONResponse:
        """Re-read routing.yaml and apply it live — no restart needed."""
        router: Router = request.app.state.router
        routing_path: Path | None = request.app.state.routing_path

        if routing_path is None:
            return JSONResponse(
                {"error": "No routing file configured. Set routing_file in config or create routing.yaml next to it."},
                status_code=400,
            )
        try:
            routes = load_routing_file(routing_path)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to read {routing_path}: {exc}"}, status_code=422)

        router.reload_routing(routes)
        routed = router.get_routed_models()
        return JSONResponse({
            "status": "reloaded",
            "routing_file": str(routing_path),
            "routes": list(routed.keys()),
        })

    @app.get("/api/discovery")
    async def api_discovery(request: Request) -> JSONResponse:
        discovery: DiscoveryResult = request.app.state.discovery
        diff = request.app.state.discovery_diff  # DiscoveryDiff | None

        diff_payload = None
        if diff is not None:
            diff_payload = {
                "new_models": diff.new_models,
                "removed_models": diff.removed_models,
                "routing_lost": diff.routing_lost,
                "new_endpoints_for": diff.new_endpoints_for,
                "lost_endpoints_for": diff.lost_endpoints_for,
                "has_changes": diff.has_changes,
                "requires_onboarding": diff.requires_onboarding,
            }

        return JSONResponse({
            "scanned_at": discovery.scanned_at,
            "models": discovery.models,
            "endpoint_models": discovery.endpoint_models,
            "endpoint_reachable": discovery.endpoint_reachable,
            "diff": diff_payload,
        })

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

def _resolve_route(
    model_id: str,
    router: Router,
    cfg: ProxyConfig,
) -> tuple[list[RouteStep], str]:
    """Resolve a client-supplied model ID into (steps, fallback_model).

    ``steps``          — ordered list of (endpoint, model) pairs to attempt.
    ``fallback_model`` — model name used when a step has model=None (wildcard steps).

    Cases:
      "llm-proxy/router"   → wildcard steps, fallback = cfg.failover.default_model
      "best-available"     → named route steps (each step has its own model)
      "alpha/gpt-4"        → single direct step: alpha endpoint with gpt-4
      "gpt-4"              → auto-discovered route, or wildcard fallback with "gpt-4"
    """
    # 1. Special routing model → full wildcard failover
    if model_id == ROUTING_MODEL_ID:
        return router.get_route("*"), cfg.failover.default_model

    # 2. "endpoint_name/model" → single direct step
    if "/" in model_id:
        ep_name, _, actual_model = model_id.partition("/")
        ep = router.get_endpoint_by_name(ep_name)
        if ep is not None:
            return [RouteStep(ep, actual_model)], actual_model

    # 3. Named route or auto-discovered model → use routing table
    steps = router.get_route(model_id)
    return steps, model_id


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
    steps, fallback_model = _resolve_route(raw_model, router, cfg)

    is_stream: bool = bool(body.get("stream", False))

    # Build headers to forward (model will be set per-step in the handlers)
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
            raw_model, log_body, t_start,
            steps=steps, fallback_model=fallback_model,
        )
    else:
        return await _handle_normal(
            client, router, db, cfg,
            upstream_path, body, forward_headers,
            raw_model, log_body, t_start,
            steps=steps, fallback_model=fallback_model,
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
    steps: list[RouteStep],
    fallback_model: str,
) -> Response:
    loop = asyncio.get_event_loop()
    try:
        response, winning_step, attempts = await router.execute(
            client, path, body, extra_headers, steps, fallback_model,
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
        selected_endpoint=winning_step.endpoint.name,
        attempts=attempts,
        status="success",
        total_latency_ms=total_ms,
        is_stream=False,
        request_body=log_body,
    )
    await loop.run_in_executor(None, db.insert_request_log, log)

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
    steps: list[RouteStep],
    fallback_model: str,
) -> StreamingResponse:
    from .config import resolve_headers as _resolve
    from .router import _should_failover

    loop = asyncio.get_event_loop()

    eligible = router.filter_steps(steps)
    if not eligible:
        raise HTTPException(status_code=502, detail="No endpoints available")

    max_attempts = min(len(eligible), router._failover.max_retries + 1)
    attempts: list[AttemptLog] = []

    async def _try_stream() -> tuple[httpx.Response, RouteStep, list[AttemptLog]]:
        import httpx as _httpx

        for step in eligible[:max_attempts]:
            ep = step.endpoint
            model_for_step = step.model or fallback_model
            body_for_step = {**body, "model": model_for_step}

            headers = _resolve(ep.headers)
            headers.update(extra_headers)
            url = f"{ep.url}{path}"
            timeout = ep.timeout_ms / 1000.0
            t0 = time.monotonic()
            try:
                # httpx >=0.20: timeout must be in request extensions, not send()
                _t = {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}
                resp = await client.send(
                    client.build_request(
                        "POST", url, json=body_for_step, headers=headers,
                        extensions={"timeout": _t},
                    ),
                    stream=True,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                if _should_failover(resp.status_code):
                    await resp.aclose()
                    logger.warning(
                        "Stream upstream %r/%r returned %d — trying next step",
                        ep.name, model_for_step, resp.status_code,
                    )
                    router.record_failure(ep, is_timeout=False)
                    attempts.append(AttemptLog(
                        endpoint_name=ep.name, latency_ms=latency_ms,
                        success=False, is_timeout=False,
                        error_message=f"HTTP {resp.status_code}",
                    ))
                    continue

                router.record_success(ep, latency_ms)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=True, is_timeout=False,
                ))
                return resp, step, attempts
            except _httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream timeout on %r: %s", ep.name, exc)
                router.record_failure(ep, is_timeout=True)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=True, error_message=str(exc),
                ))
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream error on %r: %s", ep.name, exc)
                router.record_failure(ep, is_timeout=False)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=False, error_message=str(exc),
                ))
        raise AllEndpointsFailedError(attempts)

    try:
        upstream_resp, winning_step, final_attempts = await _try_stream()
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
                selected_endpoint=winning_step.endpoint.name,
                attempts=final_attempts,
                status="success",
                total_latency_ms=total_ms,
                is_stream=True,
                request_body=log_body,
            )
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
    """Return a merged model list.

    Response layout:
      1. llm-proxy/router  — full failover across all endpoints
      2. Per-model routed entries (from routing config or auto-discovery)
         with x-routing metadata showing the endpoint priority order
      3. Direct {endpoint}/{model} entries (bypass failover, go straight to
         that endpoint) — fetched live from each upstream
    """
    import time as _time
    from .config import resolve_headers as _resolve

    client: httpx.AsyncClient = request.app.state.client
    router: Router = request.app.state.router

    # --- Section 2: named routes (config-defined chains + auto-discovered) --
    routed_models = router.get_routed_models()
    routed_entries = [
        {
            "id": route_name,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "llm-proxy",
            # x-routing shows each step as {server, model} for tooling/dashboards
            "x-routing": chain,
        }
        for route_name, chain in routed_models.items()
    ]

    # --- Section 3: direct {endpoint}/{model} entries (live fetch) -------
    async def _fetch(ep: EndpointState) -> list[dict]:
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
        return [{
            "id": ep.name,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "llm-proxy",
        }]

    all_eps = router.all_endpoints()
    results = await asyncio.gather(*[_fetch(ep) for ep in all_eps])
    direct_entries = [m for models in results for m in models]

    # --- Section 1: routing model -----------------------------------------
    routing_entry = {
        "id": ROUTING_MODEL_ID,
        "object": "model",
        "created": int(_time.time()),
        "owned_by": "llm-proxy",
        "x-routing": [ep.name for ep in router.all_endpoints()],
    }

    return JSONResponse({
        "object": "list",
        "data": [routing_entry, *routed_entries, *direct_entries],
    })


# ---------------------------------------------------------------------------
# Generic passthrough for non-chat /v1/* routes
# ---------------------------------------------------------------------------

async def _handle_passthrough(request: Request, upstream_path: str) -> Response:
    """Forward non-chat requests to the first reachable upstream as-is.
    No failover logic — just forward with the correct HTTP method.
    """
    client: httpx.AsyncClient = request.app.state.client
    router: Router = request.app.state.router

    candidates = router.filter_steps(router.get_route("*"))
    if not candidates:
        raise HTTPException(status_code=502, detail="No endpoints available")

    ep = candidates[0].endpoint
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
