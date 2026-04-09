"""Routing table, circuit breaker, and failover logic."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .config import ProxyConfig, resolve_headers
from .models import (
    AttemptLog,
    CircuitState,
    EndpointState,
    EndpointStatus,
    RouteStep,
    RoutingTable,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _should_failover(status_code: int) -> bool:
    """Return True if an HTTP status code should trigger failover to the next step.

    5xx = server errors (worth retrying on another endpoint).
    429 = rate limited (try another endpoint).
    4xx other than 429 = client errors (same request will fail everywhere; don't retry).
    """
    return status_code >= 500 or status_code == 429


class AllEndpointsFailedError(Exception):
    def __init__(self, attempts: list[AttemptLog]) -> None:
        self.attempts = attempts
        super().__init__(f"All steps failed after {len(attempts)} attempt(s)")


class Router:
    """Manages the routing table and circuit-breaker state for all endpoints."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._failover = config.failover
        self._ep_by_name: dict[str, EndpointState] = {}
        self._table: RoutingTable = self._build_routing_table()
        # Last discovery result — kept so reload_routing() can re-apply it
        self._last_discovery: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_routing_table(self) -> RoutingTable:
        """Build the initial routing table.

        Creates one EndpointState per endpoint (shared across all routing entries
        and all chain positions). The same endpoint can appear multiple times in a
        chain with different models — those steps share a single circuit-breaker state.

        Table layout:
          "*"             — wildcard; step.model=None (inherit from request)
          "<route-name>"  — named chain; each step carries explicit (endpoint, model)
        """
        for ep in self._config.endpoints:
            self._ep_by_name[ep.name] = EndpointState(
                name=ep.name,
                url=ep.url,
                timeout_ms=ep.timeout_ms,
                headers=ep.headers,
            )

        # "*" wildcard: all endpoints in config order, model inherited per-request
        all_states = list(self._ep_by_name.values())
        table: RoutingTable = {
            "*": [RouteStep(ep, None) for ep in all_states]
        }

        for route in self._config.routing:
            steps: list[RouteStep] = []
            unknown: list[str] = []
            for step_cfg in route.chain:
                ep = self._ep_by_name.get(step_cfg.endpoint)
                if ep is not None:
                    steps.append(RouteStep(ep, step_cfg.model))
                else:
                    unknown.append(step_cfg.endpoint)

            if unknown:
                logger.warning(
                    "Route %r references unknown endpoint(s): %s — those steps skipped",
                    route.name, unknown,
                )
            if steps:
                table[route.name] = steps
                logger.info(
                    "Named route %r: %s",
                    route.name,
                    " -> ".join(f"{s.endpoint.name}/{s.model}" for s in steps),
                )

        return table

    def reload_routing(self, routes: list) -> None:  # routes: list[RouteConfig]
        """Hot-reload named routes without restarting.

        Replaces all config-defined named routes, then re-applies the last
        auto-discovery result so discovered model routes are preserved.
        Circuit-breaker state for endpoints is never touched.
        """
        # Drop every named route (keep wildcard and auto-discovered will be re-added)
        self._table = {"*": self._table["*"]}

        # Apply new routes
        for route in routes:
            steps: list[RouteStep] = []
            unknown: list[str] = []
            for step_cfg in route.chain:
                ep = self._ep_by_name.get(step_cfg.endpoint)
                if ep is not None:
                    steps.append(RouteStep(ep, step_cfg.model))
                else:
                    unknown.append(step_cfg.endpoint)
            if unknown:
                logger.warning(
                    "Route %r references unknown endpoint(s): %s — steps skipped",
                    route.name, unknown,
                )
            if steps:
                self._table[route.name] = steps
                logger.info(
                    "Reloaded route %r: %s",
                    route.name,
                    " -> ".join(f"{s.endpoint.name}/{s.model}" for s in steps),
                )

        logger.info("Routing reloaded — %d named route(s)", len(self._table) - 1)

    def update_routing_from_discovery(self, discovered: dict[str, list[str]]) -> None:
        """Store the latest discovery results for informational use.

        Discovery data is used by the ``/api/discovery`` endpoint and the
        ``discover`` CLI command. It is NOT added to the routing table — only
        explicitly configured named routes and ``endpoint/model`` direct
        addressing appear in the table.
        """
        self._last_discovery = discovered

    # ------------------------------------------------------------------
    # Route access
    # ------------------------------------------------------------------

    def get_route(self, name: str) -> list[RouteStep]:
        """Return the raw (unfiltered) step list for a route name.

        Returns the "*" wildcard list if ``name`` is not found.
        """
        return self._table.get(name) or self._table.get("*") or []

    def filter_steps(self, steps: list[RouteStep]) -> list[RouteStep]:
        """Remove circuit-OPEN steps; apply latency sort if configured."""
        eligible: list[RouteStep] = []
        for step in steps:
            state = self._maybe_transition_to_half_open(step.endpoint)
            if state == "open":
                logger.debug("Skipping %r — circuit OPEN", step.endpoint.name)
                continue
            eligible.append(step)

        if self._failover.routing_strategy == "latency":
            eligible = sorted(
                eligible,
                key=lambda s: (
                    s.endpoint.avg_latency_ms is None,
                    s.endpoint.avg_latency_ms or 0,
                ),
            )
        return eligible

    def get_routed_models(self) -> dict[str, list[dict[str, str]]]:
        """Return named routes (non-wildcard) as name → chain description.

        Each chain entry is {"server": ep_name, "model": model_id}.
        """
        result: dict[str, list[dict[str, str]]] = {}
        for name, steps in self._table.items():
            if name == "*":
                continue
            result[name] = [
                {"server": s.endpoint.name, "model": s.model or ""}
                for s in steps
            ]
        return result

    # ------------------------------------------------------------------
    # Circuit breaker helpers
    # ------------------------------------------------------------------

    def _maybe_transition_to_half_open(self, ep: EndpointState) -> CircuitState:
        if ep.circuit_state == "open" and ep.open_since is not None:
            elapsed = time.monotonic() - ep.open_since
            if elapsed >= self._failover.circuit_breaker_cooldown:
                logger.info(
                    "Circuit breaker HALF_OPEN for %r (cooldown expired after %.1fs)",
                    ep.name,
                    elapsed,
                )
                ep.circuit_state = "half_open"
                ep.open_since = None
        return ep.circuit_state

    def record_success(self, ep: EndpointState, latency_ms: float) -> None:
        ep.total_requests += 1
        ep.latency_samples.append(latency_ms)
        ep.consecutive_failures = 0
        if ep.circuit_state == "half_open":
            logger.info("Circuit breaker CLOSED for %r (probe succeeded)", ep.name)
            ep.circuit_state = "closed"

    def record_failure(self, ep: EndpointState, *, is_timeout: bool) -> None:
        ep.total_requests += 1
        ep.total_failures += 1
        if is_timeout:
            ep.total_timeouts += 1
        ep.consecutive_failures += 1
        threshold = self._failover.circuit_breaker_threshold

        if ep.circuit_state == "half_open":
            logger.warning("Circuit breaker OPEN for %r (probe failed)", ep.name)
            ep.circuit_state = "open"
            ep.open_since = time.monotonic()
        elif ep.circuit_state == "closed" and ep.consecutive_failures >= threshold:
            logger.warning(
                "Circuit breaker OPEN for %r (%d consecutive failures)",
                ep.name, ep.consecutive_failures,
            )
            ep.circuit_state = "open"
            ep.open_since = time.monotonic()

    # ------------------------------------------------------------------
    # Status snapshot (for API / dashboard)
    # ------------------------------------------------------------------

    def get_status(self) -> list[EndpointStatus]:
        return [
            EndpointStatus(
                name=ep.name,
                url=ep.url,
                circuit_state=ep.circuit_state,
                consecutive_failures=ep.consecutive_failures,
                total_requests=ep.total_requests,
                total_failures=ep.total_failures,
                total_timeouts=ep.total_timeouts,
                avg_latency_ms=ep.avg_latency_ms,
                timeout_rate=ep.timeout_rate,
                failure_rate=ep.failure_rate,
            )
            for ep in self._ep_by_name.values()
        ]

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_endpoint_by_name(self, name: str) -> EndpointState | None:
        return self._ep_by_name.get(name)

    def all_endpoints(self) -> list[EndpointState]:
        return list(self._ep_by_name.values())

    # ------------------------------------------------------------------
    # Failover execution (non-streaming)
    # ------------------------------------------------------------------

    async def execute(
        self,
        client: "httpx.AsyncClient",
        path: str,
        body: dict,
        extra_headers: dict[str, str],
        steps: list[RouteStep],
        fallback_model: str,
    ) -> tuple["httpx.Response", RouteStep, list[AttemptLog]]:
        """Try each RouteStep in order until one succeeds.

        ``body`` must already contain all fields except model — we set the
        model per step using ``step.model`` (or ``fallback_model`` when None).

        Returns (response, winning_step, attempts).
        Raises AllEndpointsFailedError if every step fails.
        """
        import httpx as _httpx

        eligible = self.filter_steps(steps)
        if not eligible:
            raise AllEndpointsFailedError([])

        max_attempts = min(len(eligible), self._failover.max_retries + 1)
        attempts: list[AttemptLog] = []

        for step in eligible[:max_attempts]:
            ep = step.endpoint
            model_for_step = step.model or fallback_model
            body_for_step = {**body, "model": model_for_step}

            from .server import merge_headers
            headers = merge_headers(
                resolve_headers(ep.headers), extra_headers,
                self._config.proxy.header_priority,
            )
            url = f"{ep.url}{path}"
            timeout = ep.timeout_ms / 1000.0
            t0 = time.monotonic()

            try:
                response = await client.post(url, json=body_for_step, headers=headers, timeout=timeout)
                latency_ms = (time.monotonic() - t0) * 1000

                if _should_failover(response.status_code):
                    logger.warning(
                        "Upstream %r/%r returned %d — trying next step",
                        ep.name, model_for_step, response.status_code,
                    )
                    self.record_failure(ep, is_timeout=False)
                    attempts.append(AttemptLog(
                        endpoint_name=ep.name, latency_ms=latency_ms,
                        success=False, is_timeout=False,
                        error_message=f"HTTP {response.status_code}",
                    ))
                    continue

                self.record_success(ep, latency_ms)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=True, is_timeout=False,
                ))
                return response, step, attempts

            except _httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Timeout on %r (%.0fms): %s", ep.name, latency_ms, exc)
                self.record_failure(ep, is_timeout=True)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=True, error_message=str(exc),
                ))

            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Error on %r: %s", ep.name, exc)
                self.record_failure(ep, is_timeout=False)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=False, error_message=str(exc),
                ))

        raise AllEndpointsFailedError(attempts)
