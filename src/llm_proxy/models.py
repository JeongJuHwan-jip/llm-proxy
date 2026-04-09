"""Internal runtime data models (not persisted; config models live in config.py)."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Literal


CircuitState = Literal["closed", "open", "half_open"]


@dataclass
class EndpointState:
    """Runtime state for a single upstream endpoint."""

    name: str
    url: str                              # base URL (no trailing slash)
    timeout_ms: int
    headers: dict[str, str]              # raw headers (templates not yet resolved)

    # circuit breaker
    circuit_state: CircuitState = "closed"
    consecutive_failures: int = 0
    open_since: float | None = None      # time.monotonic() timestamp

    # statistics
    total_requests: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    latency_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )

    @property
    def avg_latency_ms(self) -> float | None:
        if not self.latency_samples:
            return None
        return sum(self.latency_samples) / len(self.latency_samples)

    @property
    def timeout_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_timeouts / self.total_requests

    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_failures / self.total_requests


@dataclass
class RouteStep:
    """One step in a named route: the endpoint to try and the model to request.

    ``model`` is the model name sent to this specific endpoint.
    """

    endpoint: EndpointState
    model: str | None


# RoutingTable maps route-name → ordered list of RouteStep.
#   "best-available" — named route; each step carries its own (endpoint, model)
RoutingTable = dict[str, list[RouteStep]]


@dataclass
class AttemptLog:
    endpoint_name: str
    latency_ms: float
    success: bool
    is_timeout: bool
    error_message: str | None = None


@dataclass
class RequestLog:
    timestamp: float
    model: str
    selected_endpoint: str | None       # name of the endpoint that eventually succeeded
    attempts: list[AttemptLog]
    status: Literal["success", "failure"]
    total_latency_ms: float
    is_stream: bool
    request_body: dict | None = None    # populated only when log_request_body=true


@dataclass
class EndpointStatus:
    """Serialisable snapshot of an EndpointState for API / dashboard."""

    name: str
    url: str
    circuit_state: CircuitState
    consecutive_failures: int
    total_requests: int
    total_failures: int
    total_timeouts: int
    avg_latency_ms: float | None
    timeout_rate: float
    failure_rate: float
