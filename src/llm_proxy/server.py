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

from .config import ProxyConfig, load_settings_file, resolve_settings_path
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


# ---------------------------------------------------------------------------
# SSE event broadcaster — dashboard subscribes to this
# ---------------------------------------------------------------------------

class EventBroadcaster:
    """Lightweight pub/sub for SSE. Each dashboard tab holds one asyncio.Queue."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[str]] = set()

    def subscribe(self) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=32)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[str]) -> None:
        self._subscribers.discard(q)

    def publish(self, event: str = "refresh") -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # slow consumer — skip


# Headers that must NOT be forwarded to upstream (hop-by-hop / set by httpx)
_STRIP_HEADERS = {
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "authorization",  # proxy auth — endpoint headers handle upstream auth
}


def merge_headers(
    config_headers: dict[str, str],
    client_headers: dict[str, str],
    priority: str,
) -> dict[str, str]:
    """Merge endpoint config headers with client-forwarded headers.

    ``priority`` controls which side wins on key conflicts:
      - "config": config headers overwrite client headers (default)
      - "client": client headers overwrite config headers
    """
    if priority == "client":
        merged = {**config_headers, **client_headers}
    else:
        merged = {**client_headers, **config_headers}
    return merged


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

        # HTTP clients — one per ssl_verify value (True/False)
        limits = httpx.Limits(
            max_connections=200,
            max_keepalive_connections=50,
            keepalive_expiry=30,
        )
        client = httpx.AsyncClient(limits=limits, follow_redirects=False, verify=True)
        needs_nossl = any(not ep.ssl_verify for ep in cfg.endpoints)
        client_nossl = (
            httpx.AsyncClient(limits=limits, follow_redirects=False, verify=False)
            if needs_nossl else None
        )
        clients: dict[bool, httpx.AsyncClient] = {True: client}
        if client_nossl is not None:
            clients[False] = client_nossl

        router = Router(cfg)
        db = Database(cfg.logging.db_path)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, db.init)

        app.state.client = client
        app.state.clients = clients
        app.state.router = router
        app.state.db = db
        app.state.events = EventBroadcaster()

        # ---- load settings.json (overrides inline routing + failover defaults) ----
        settings_path = resolve_settings_path(
            config_path or Path(cfg.logging.db_path),
        )
        app.state.settings_path = settings_path
        if settings_path.exists():
            try:
                sdata = load_settings_file(settings_path)
                router.reload_routing(sdata.routes)
                if sdata.failover is not None:
                    cfg.failover = sdata.failover
                    logger.info("Loaded failover settings from %s", settings_path)
                logger.info("Loaded settings from %s (%d routes)", settings_path, len(sdata.routes))
            except Exception as exc:
                logger.warning("Could not load %s: %s — using defaults", settings_path, exc)
        # ------------------------------------------------------------

        # ---- model discovery & change detection ----
        snapshot_path = Path(cfg.logging.db_path).parent / "model_snapshot.json"
        old_snapshot = load_snapshot(snapshot_path)

        discovery = await run_discovery(clients, router.all_endpoints())
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
        if client_nossl is not None:
            await client_nossl.aclose()
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

    # POST /v1/messages — Anthropic Messages API adapter
    @app.post(
        "/v1/messages",
        dependencies=[Depends(_require_api_key)],
    )
    async def anthropic_messages(request: Request) -> Response:
        from .adapters.anthropic import handle_anthropic_messages
        return await handle_anthropic_messages(request)

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

    @app.delete("/api/requests")
    async def api_clear_requests(request: Request) -> JSONResponse:
        db: Database = request.app.state.db
        loop = asyncio.get_event_loop()
        deleted = await loop.run_in_executor(None, db.clear_all_requests)
        return JSONResponse({"deleted": deleted})

    @app.get("/api/stats")
    async def api_stats(request: Request) -> JSONResponse:
        db: Database = request.app.state.db
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, db.get_endpoint_stats)
        return JSONResponse(stats)

    @app.post("/api/routing/reload")
    async def api_routing_reload(request: Request) -> JSONResponse:
        """Re-read settings.json and apply it live — no restart needed."""
        router: Router = request.app.state.router
        cfg: ProxyConfig = request.app.state.config
        settings_path: Path = request.app.state.settings_path

        if not settings_path.exists():
            return JSONResponse(
                {"error": f"Settings file not found: {settings_path}. Use the dashboard to create it."},
                status_code=400,
            )
        try:
            sdata = load_settings_file(settings_path)
        except Exception as exc:
            return JSONResponse({"error": f"Failed to read {settings_path}: {exc}"}, status_code=422)

        router.reload_routing(sdata.routes)
        if sdata.failover is not None:
            cfg.failover = sdata.failover
        routed = router.get_routed_models()
        return JSONResponse({
            "status": "reloaded",
            "settings_file": str(settings_path),
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

    # --------------------------------------------------------- config helpers

    def _save_settings(app_instance: FastAPI, route_configs=None) -> None:
        """Save current failover + routing to settings.json."""
        cfg_: ProxyConfig = app_instance.state.config
        router_: Router = app_instance.state.router
        spath: Path = app_instance.state.settings_path

        def _step_dict(endpoint: str, model: str, timeout_ms: int,
                       max_context_tokens: int | None) -> dict:
            d: dict = {"endpoint": endpoint, "model": model, "timeout_ms": timeout_ms}
            if max_context_tokens is not None:
                d["max_context_tokens"] = max_context_tokens
            return d

        if route_configs is None:
            routed = router_.get_routed_models()
            routes_out = [
                {
                    "name": rname,
                    "chain": [
                        _step_dict(s["server"], s["model"], s["timeout_ms"], s["max_context_tokens"])
                        for s in chain
                    ],
                }
                for rname, chain in routed.items()
            ]
        else:
            routes_out = [
                {
                    "name": rc.name,
                    "chain": [
                        _step_dict(s.endpoint, s.model, s.timeout_ms, s.max_context_tokens)
                        for s in rc.chain
                    ],
                }
                for rc in route_configs
            ]

        data = {
            "failover": {
                "max_retries": cfg_.failover.max_retries,
                "circuit_breaker_threshold": cfg_.failover.circuit_breaker_threshold,
                "circuit_breaker_cooldown": cfg_.failover.circuit_breaker_cooldown,
                "routing_strategy": cfg_.failover.routing_strategy,
            },
            "routes": routes_out,
        }

        with open(spath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved settings to %s", spath)

    # --------------------------------------------------------- config APIs

    @app.get("/api/config/settings")
    async def api_get_settings(request: Request) -> JSONResponse:
        """Return failover + routing settings.

        Reads from settings.json if it exists; otherwise returns defaults
        with empty routes so the dashboard starts clean.
        """
        cfg: ProxyConfig = request.app.state.config
        spath: Path = request.app.state.settings_path

        routes: list[dict] = []
        if spath.exists():
            try:
                sdata = load_settings_file(spath)
                routes = [
                    {
                        "name": r.name,
                        "chain": [
                            {
                                "endpoint": s.endpoint,
                                "model": s.model,
                                "timeout_ms": s.timeout_ms,
                                "max_context_tokens": s.max_context_tokens,
                            }
                            for s in r.chain
                        ],
                    }
                    for r in sdata.routes
                ]
            except Exception:
                pass  # corrupted file → show empty

        return JSONResponse({
            "failover": {
                "max_retries": cfg.failover.max_retries,
                "circuit_breaker_threshold": cfg.failover.circuit_breaker_threshold,
                "circuit_breaker_cooldown": cfg.failover.circuit_breaker_cooldown,
                "routing_strategy": cfg.failover.routing_strategy,
            },
            "routes": routes,
        })

    @app.put("/api/config/settings")
    async def api_put_settings(request: Request) -> JSONResponse:
        """Apply failover + routing together, persist to settings.json."""
        cfg: ProxyConfig = request.app.state.config
        router: Router = request.app.state.router
        events: EventBroadcaster = request.app.state.events

        try:
            data = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        errors: list[str] = []

        # ── Validate failover ──
        fo = data.get("failover", {})
        if "max_retries" in fo:
            v = fo["max_retries"]
            if not isinstance(v, int) or v < 0:
                errors.append("max_retries must be a non-negative integer")
        if "circuit_breaker_threshold" in fo:
            v = fo["circuit_breaker_threshold"]
            if not isinstance(v, int) or v < 1:
                errors.append("circuit_breaker_threshold must be >= 1")
        if "circuit_breaker_cooldown" in fo:
            v = fo["circuit_breaker_cooldown"]
            if not isinstance(v, int) or v < 0:
                errors.append("circuit_breaker_cooldown must be >= 0")
        if "routing_strategy" in fo:
            v = fo["routing_strategy"]
            if v not in ("priority", "latency"):
                errors.append("routing_strategy must be 'priority' or 'latency'")

        # ── Validate routes ──
        routes_raw = data.get("routes", [])
        if not isinstance(routes_raw, list):
            errors.append("'routes' must be a list")

        if errors:
            return JSONResponse({"errors": errors}, status_code=422)

        from .config import RouteConfig
        try:
            route_configs = [RouteConfig.model_validate(r) for r in routes_raw]
        except Exception as exc:
            return JSONResponse({"error": f"Validation error: {exc}"}, status_code=422)

        # ── Apply failover in memory ──
        if "max_retries" in fo:
            cfg.failover.max_retries = fo["max_retries"]
        if "circuit_breaker_threshold" in fo:
            cfg.failover.circuit_breaker_threshold = fo["circuit_breaker_threshold"]
        if "circuit_breaker_cooldown" in fo:
            cfg.failover.circuit_breaker_cooldown = fo["circuit_breaker_cooldown"]
        if "routing_strategy" in fo:
            cfg.failover.routing_strategy = fo["routing_strategy"]

        # ── Save to settings.json ──
        _save_settings(request.app, route_configs=route_configs)

        # ── Reload routing ──
        router.reload_routing(route_configs)
        routed = router.get_routed_models()

        logger.info(
            "Settings updated via dashboard — failover + %d route(s), saved to %s",
            len(route_configs), request.app.state.settings_path,
        )
        events.publish()

        return JSONResponse({
            "status": "updated",
            "settings_file": str(request.app.state.settings_path),
            "routes": list(routed.keys()),
        })

    # ---------------------------------------------------------------- SSE
    @app.get("/api/events")
    async def api_events(request: Request) -> StreamingResponse:
        """SSE stream — sends 'refresh' whenever a proxy request completes."""
        broadcaster: EventBroadcaster = request.app.state.events
        q = broadcaster.subscribe()

        async def event_stream() -> AsyncIterator[str]:
            try:
                while True:
                    event = await q.get()
                    yield f"data: {event}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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
) -> tuple[list[RouteStep], str, bool]:
    """Resolve a client-supplied model ID into (steps, fallback_model, is_direct).

    ``steps``          — ordered list of (endpoint, model) pairs to attempt.
    ``fallback_model`` — model name used when a step has model=None.
    ``is_direct``      — True for direct endpoint/model requests (no failover/CB).

    Cases:
      "best-available"     → named route steps (each step has its own model)
      "alpha/gpt-4"        → single direct step: alpha endpoint with gpt-4
    """
    # 1. "endpoint_name/model" → single direct step (no circuit breaker)
    if "/" in model_id:
        ep_name, _, actual_model = model_id.partition("/")
        ep = router.get_endpoint_by_name(ep_name)
        if ep is not None:
            from .models import DEFAULT_TIMEOUT_MS
            return [RouteStep(ep, actual_model, DEFAULT_TIMEOUT_MS)], actual_model, True

    # 2. Named route only
    steps = router.get_route(model_id)
    if not steps:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model {model_id!r}. Use 'endpoint/model' or a named route.",
        )
    return steps, model_id, False


async def _handle_proxy(request: Request, upstream_path: str) -> Response:
    clients: dict[bool, httpx.AsyncClient] = request.app.state.clients
    router: Router = request.app.state.router
    db: Database = request.app.state.db
    cfg: ProxyConfig = request.app.state.config

    # Parse request body
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    raw_model: str = body.get("model", "unknown")
    steps, fallback_model, is_direct = _resolve_route(raw_model, router)

    is_stream: bool = bool(body.get("stream", False))

    # Build headers to forward (model will be set per-step in the handlers)
    forward_headers: dict[str, str] = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in _STRIP_HEADERS
    }

    t_start = time.monotonic()
    log_body = body if cfg.logging.log_request_body else None

    events: EventBroadcaster = request.app.state.events

    if is_stream:
        return await _handle_stream(
            clients, router, db, cfg, events,
            upstream_path, body, forward_headers,
            raw_model, log_body, t_start,
            steps=steps, fallback_model=fallback_model,
            is_direct=is_direct,
        )
    else:
        try:
            result = await _handle_normal(
                clients, router, db, cfg,
                upstream_path, body, forward_headers,
                raw_model, log_body, t_start,
                steps=steps, fallback_model=fallback_model,
                is_direct=is_direct,
            )
        finally:
            events.publish()
        return result


async def _handle_normal(
    clients: dict[bool, httpx.AsyncClient],
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
    is_direct: bool = False,
) -> Response:
    loop = asyncio.get_event_loop()
    try:
        response, winning_step, attempts = await router.execute(
            clients, path, body, extra_headers, steps, fallback_model,
            is_direct=is_direct,
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
    clients: dict[bool, httpx.AsyncClient],
    router: Router,
    db: Database,
    cfg: ProxyConfig,
    events: EventBroadcaster,
    path: str,
    body: dict,
    extra_headers: dict[str, str],
    model: str,
    log_body: dict | None,
    t_start: float,
    steps: list[RouteStep],
    fallback_model: str,
    is_direct: bool = False,
) -> StreamingResponse:
    from .config import resolve_headers as _resolve
    from .router import _should_failover

    loop = asyncio.get_event_loop()

    if is_direct:
        eligible = steps
    else:
        eligible = router.filter_by_context(router.filter_steps(steps), body)
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

            headers = merge_headers(
                _resolve(ep.headers), extra_headers, cfg.proxy.header_priority,
            )
            url = f"{ep.url}{path}"
            timeout = step.timeout_ms / 1000.0
            logger.debug(
                "Upstream request → %s  headers=%s  model=%s",
                url, headers, model_for_step,
            )
            t0 = time.monotonic()
            _c = clients.get(ep.ssl_verify, clients[True])
            try:
                # httpx >=0.20: timeout must be in request extensions, not send()
                _t = {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}
                resp = await _c.send(
                    _c.build_request(
                        "POST", url, json=body_for_step, headers=headers,
                        extensions={"timeout": _t},
                    ),
                    stream=True,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                if _should_failover(resp.status_code):
                    logger.warning(
                        "Stream upstream %r/%r returned %d — %s",
                        ep.name, model_for_step, resp.status_code,
                        "returning error (direct)" if is_direct else "trying next step",
                    )
                    if not is_direct:
                        await resp.aclose()
                        router.record_failure(ep, is_timeout=False)
                    attempts.append(AttemptLog(
                        endpoint_name=ep.name, latency_ms=latency_ms,
                        success=False, is_timeout=False,
                        error_message=f"HTTP {resp.status_code}",
                    ))
                    if is_direct:
                        return resp, step, attempts
                    continue

                if not is_direct:
                    router.record_success(ep, latency_ms)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=True, is_timeout=False,
                ))
                return resp, step, attempts
            except _httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream timeout on %r: %s", ep.name, exc)
                if not is_direct:
                    router.record_failure(ep, is_timeout=True)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=True, error_message=str(exc),
                ))
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Stream error on %r: %s", ep.name, exc)
                if not is_direct:
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
        events.publish()
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
            await loop.run_in_executor(None, db.insert_request_log, log)
            events.publish()

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
    """Return the model list.

    Response layout:
      1. {endpoint}/{model} — direct entries fetched live from each upstream
      2. Named routes — config-defined fallback chains
    """
    import time as _time
    from .config import resolve_headers as _resolve

    clients: dict[bool, httpx.AsyncClient] = request.app.state.clients
    router: Router = request.app.state.router

    # --- Section 1: endpoint/model direct entries (live fetch) ------------
    async def _fetch(ep: EndpointState) -> list[dict]:
        try:
            from .models import DEFAULT_TIMEOUT_MS
            headers = _resolve(ep.headers)
            _c = clients.get(ep.ssl_verify, clients[True])
            resp = await _c.get(
                f"{ep.url}/models",
                headers=headers,
                timeout=DEFAULT_TIMEOUT_MS / 1000.0,
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
        return []

    all_eps = router.all_endpoints()
    # gather preserves input order → results[i] corresponds to all_eps[i]
    results = await asyncio.gather(*[_fetch(ep) for ep in all_eps])
    # Flatten in endpoint config order (not by response arrival time)
    direct_entries = [m for ep_models in results for m in ep_models]

    # --- Section 2: named routes ------------------------------------------
    routed_models = router.get_routed_models()
    route_entries = [
        {
            "id": route_name,
            "object": "model",
            "created": int(_time.time()),
            "owned_by": "llm-proxy",
            "x-routing": chain,
        }
        for route_name, chain in routed_models.items()
    ]

    return JSONResponse({
        "object": "list",
        "data": [*direct_entries, *route_entries],
    })


# ---------------------------------------------------------------------------
# Generic passthrough for non-chat /v1/* routes
# ---------------------------------------------------------------------------

async def _handle_passthrough(request: Request, upstream_path: str) -> Response:
    """Forward non-chat requests to the first reachable upstream as-is.
    No failover logic — just forward with the correct HTTP method.
    """
    clients: dict[bool, httpx.AsyncClient] = request.app.state.clients
    router: Router = request.app.state.router

    all_eps = router.all_endpoints()
    if not all_eps:
        raise HTTPException(status_code=502, detail="No endpoints available")

    ep = all_eps[0]
    cfg: ProxyConfig = request.app.state.config
    from .config import resolve_headers as _resolve
    client_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _STRIP_HEADERS
    }
    headers = merge_headers(
        _resolve(ep.headers), client_headers, cfg.proxy.header_priority,
    )
    url = f"{ep.url}{upstream_path}"

    # Read body only for methods that can carry one
    body_bytes = b""
    if request.method in ("POST", "PUT", "PATCH"):
        body_bytes = await request.body()

    try:
        from .models import DEFAULT_TIMEOUT_MS
        _c = clients.get(ep.ssl_verify, clients[True])
        upstream = await _c.request(
            method=request.method,
            url=url,
            content=body_bytes or None,
            headers=headers,
            timeout=DEFAULT_TIMEOUT_MS / 1000.0,
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
