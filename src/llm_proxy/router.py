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
    RoutingTable,
)

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


def _should_failover(status_code: int) -> bool:
    """Return True if an HTTP status code should trigger failover to next endpoint.

    5xx = server errors (worth retrying on another endpoint).
    429 = rate limited (try another endpoint).
    4xx other than 429 = client errors (same request will fail everywhere; don't retry).
    """
    return status_code >= 500 or status_code == 429


class AllEndpointsFailedError(Exception):
    def __init__(self, attempts: list[AttemptLog]) -> None:
        self.attempts = attempts
        super().__init__(f"All endpoints failed after {len(attempts)} attempt(s)")


class Router:
    """Manages the routing table and circuit-breaker state for all endpoints."""

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config
        self._failover = config.failover
        self._table: RoutingTable = self._build_routing_table()

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_routing_table(self) -> RoutingTable:
        """Build the initial routing table.

        All endpoints are stored under the "*" wildcard key.
        Future extension: per-model keys like "gpt-4" can be added here
        once model discovery is implemented.
        """
        states = [
            EndpointState(
                name=ep.name,
                url=ep.url,
                timeout_ms=ep.timeout_ms,
                priority=ep.priority,
                headers=ep.headers,
            )
            for ep in self._config.endpoints
        ]
        states.sort(key=lambda s: s.priority)
        return {"*": states}

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def get_candidates(self, model: str) -> list[EndpointState]:
        """Return an ordered list of endpoints eligible to handle this model.

        Lookup order:
          1. model-specific key (e.g. "gpt-4")
          2. wildcard "*"

        Endpoints whose circuit breaker is OPEN are skipped unless their
        cooldown has expired, in which case they are transitioned to HALF_OPEN.
        """
        endpoints = self._table.get(model) or self._table.get("*") or []

        eligible: list[EndpointState] = []
        for ep in endpoints:
            state = self._maybe_transition_to_half_open(ep)
            if state == "open":
                logger.debug("Skipping %r — circuit OPEN", ep.name)
                continue
            eligible.append(ep)

        if self._failover.routing_strategy == "latency":
            eligible = self._sort_by_latency(eligible)

        return eligible

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

    @staticmethod
    def _sort_by_latency(endpoints: list[EndpointState]) -> list[EndpointState]:
        """Sort by average latency; unknown-latency endpoints go last."""
        return sorted(
            endpoints,
            key=lambda ep: (ep.avg_latency_ms is None, ep.avg_latency_ms or 0),
        )

    # ------------------------------------------------------------------
    # Result recording
    # ------------------------------------------------------------------

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
            logger.warning(
                "Circuit breaker OPEN for %r (probe failed)", ep.name
            )
            ep.circuit_state = "open"
            ep.open_since = time.monotonic()

        elif (
            ep.circuit_state == "closed"
            and ep.consecutive_failures >= threshold
        ):
            logger.warning(
                "Circuit breaker OPEN for %r (%d consecutive failures)",
                ep.name,
                ep.consecutive_failures,
            )
            ep.circuit_state = "open"
            ep.open_since = time.monotonic()

    # ------------------------------------------------------------------
    # Status snapshot (for API / dashboard)
    # ------------------------------------------------------------------

    def get_status(self) -> list[EndpointStatus]:
        seen: dict[str, EndpointStatus] = {}
        for endpoints in self._table.values():
            for ep in endpoints:
                if ep.name not in seen:
                    seen[ep.name] = EndpointStatus(
                        name=ep.name,
                        url=ep.url,
                        priority=ep.priority,
                        circuit_state=ep.circuit_state,
                        consecutive_failures=ep.consecutive_failures,
                        total_requests=ep.total_requests,
                        total_failures=ep.total_failures,
                        total_timeouts=ep.total_timeouts,
                        avg_latency_ms=ep.avg_latency_ms,
                        timeout_rate=ep.timeout_rate,
                        failure_rate=ep.failure_rate,
                    )
        return list(seen.values())

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get_endpoint_by_name(self, name: str) -> EndpointState | None:
        for endpoints in self._table.values():
            for ep in endpoints:
                if ep.name == name:
                    return ep
        return None

    def all_endpoints(self) -> list[EndpointState]:
        seen: dict[str, EndpointState] = {}
        for endpoints in self._table.values():
            for ep in endpoints:
                if ep.name not in seen:
                    seen[ep.name] = ep
        return list(seen.values())

    # ------------------------------------------------------------------
    # Direct execution (single endpoint, no failover)
    # ------------------------------------------------------------------

    async def execute_direct(
        self,
        client: "httpx.AsyncClient",
        ep: EndpointState,
        path: str,
        body: dict,
        extra_headers: dict[str, str],
    ) -> tuple["httpx.Response", list[AttemptLog]]:
        """Send to a specific endpoint without failover.

        Raises AllEndpointsFailedError if the single attempt fails.
        """
        import httpx as _httpx

        headers = resolve_headers(ep.headers)
        headers.update(extra_headers)
        url = f"{ep.url}{path}"
        timeout = ep.timeout_ms / 1000.0
        t0 = time.monotonic()

        try:
            response = await client.post(url, json=body, headers=headers, timeout=timeout)
            latency_ms = (time.monotonic() - t0) * 1000

            if _should_failover(response.status_code):
                logger.warning("Direct %r returned %d", ep.name, response.status_code)
                self.record_failure(ep, is_timeout=False)
                raise AllEndpointsFailedError([AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=False,
                    error_message=f"HTTP {response.status_code}",
                )])

            self.record_success(ep, latency_ms)
            attempts = [AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=True, is_timeout=False,
            )]
            return response, attempts

        except _httpx.TimeoutException as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning("Direct timeout on %r: %s", ep.name, exc)
            self.record_failure(ep, is_timeout=True)
            raise AllEndpointsFailedError([AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=False, is_timeout=True, error_message=str(exc),
            )])

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t0) * 1000
            logger.warning("Direct error on %r: %s", ep.name, exc)
            self.record_failure(ep, is_timeout=False)
            raise AllEndpointsFailedError([AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=False, is_timeout=False, error_message=str(exc),
            )])

    # ------------------------------------------------------------------
    # Failover execution helper
    # ------------------------------------------------------------------

    async def execute(
        self,
        client: "httpx.AsyncClient",
        path: str,
        body: dict,
        extra_headers: dict[str, str],
    ) -> tuple["httpx.Response", EndpointState, list[AttemptLog]]:
        """Attempt to forward a request through available endpoints.

        Returns (response, winning_endpoint_state, attempts_list).
        Raises AllEndpointsFailedError if every candidate fails.

        This method is used for non-streaming requests.
        For streaming, call execute_stream() instead.
        """
        import httpx as _httpx

        model: str = body.get("model", "*")
        candidates = self.get_candidates(model)

        if not candidates:
            raise AllEndpointsFailedError([])

        max_attempts = min(
            len(candidates), self._failover.max_retries + 1
        )
        attempts: list[AttemptLog] = []

        for ep in candidates[:max_attempts]:
            headers = resolve_headers(ep.headers)
            headers.update(extra_headers)
            url = f"{ep.url}{path}"
            timeout = ep.timeout_ms / 1000.0
            t0 = time.monotonic()

            try:
                response = await client.post(
                    url,
                    json=body,
                    headers=headers,
                    timeout=timeout,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                if _should_failover(response.status_code):
                    logger.warning(
                        "Upstream %r returned %d — treating as failure",
                        ep.name, response.status_code,
                    )
                    self.record_failure(ep, is_timeout=False)
                    attempts.append(AttemptLog(
                        endpoint_name=ep.name, latency_ms=latency_ms,
                        success=False, is_timeout=False,
                        error_message=f"HTTP {response.status_code}",
                    ))
                    continue

                self.record_success(ep, latency_ms)
                attempts.append(
                    AttemptLog(
                        endpoint_name=ep.name,
                        latency_ms=latency_ms,
                        success=True,
                        is_timeout=False,
                    )
                )
                return response, ep, attempts

            except _httpx.TimeoutException as exc:
                latency_ms = (time.monotonic() - t0) * 1000
                logger.warning("Timeout on %r (%.0fms): %s", ep.name, latency_ms, exc)
                self.record_failure(ep, is_timeout=True)
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
                logger.warning("Error on %r: %s", ep.name, exc)
                self.record_failure(ep, is_timeout=False)
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
