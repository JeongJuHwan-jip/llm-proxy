"""Model discovery: scan upstream endpoints, persist snapshots, diff changes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from .models import EndpointState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """Outcome of a single discovery scan."""

    scanned_at: float
    # model_id → endpoint names that support it (sorted by endpoint priority)
    models: dict[str, list[str]]
    # endpoint_name → list of model_ids it exposes
    endpoint_models: dict[str, list[str]]
    # endpoint_name → True if the endpoint responded successfully
    endpoint_reachable: dict[str, bool]


@dataclass
class DiscoveryDiff:
    """Differences between two discovery scans."""

    new_models: list[str]                    # appeared since last scan
    removed_models: list[str]               # no longer available anywhere
    routing_lost: list[str]                 # routing-configured models that disappeared
    new_endpoints_for: dict[str, list[str]] # model → endpoints that gained support
    lost_endpoints_for: dict[str, list[str]]# model → endpoints that lost support

    @property
    def has_changes(self) -> bool:
        return bool(
            self.new_models
            or self.removed_models
            or self.new_endpoints_for
            or self.lost_endpoints_for
        )

    @property
    def requires_onboarding(self) -> bool:
        """True if routing-critical models disappeared and config must be updated."""
        return bool(self.routing_lost)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


async def run_discovery(
    client: "httpx.AsyncClient",
    endpoints: list["EndpointState"],
) -> DiscoveryResult:
    """Concurrently query /v1/models on all endpoints and aggregate results."""
    from .config import resolve_headers

    endpoint_models: dict[str, list[str]] = {}
    endpoint_reachable: dict[str, bool] = {}
    # model_id → [(priority, ep_name)]
    model_map: dict[str, list[tuple[int, str]]] = {}

    async def _fetch(ep: "EndpointState") -> None:
        try:
            headers = resolve_headers(ep.headers)
            resp = await client.get(
                f"{ep.url}/models",
                headers=headers,
                timeout=5.0,
            )
            if resp.status_code == 200:
                ids = [
                    m["id"]
                    for m in resp.json().get("data", [])
                    if isinstance(m.get("id"), str)
                ]
                endpoint_models[ep.name] = ids
                endpoint_reachable[ep.name] = True
                for mid in ids:
                    model_map.setdefault(mid, []).append((ep.priority, ep.name))
                return
        except Exception as exc:
            logger.debug("Discovery failed for %r: %s", ep.name, exc)

        endpoint_models[ep.name] = []
        endpoint_reachable[ep.name] = False

    await asyncio.gather(*[_fetch(ep) for ep in endpoints])

    models = {
        mid: [name for _, name in sorted(eps)]
        for mid, eps in model_map.items()
    }
    return DiscoveryResult(
        scanned_at=time.time(),
        models=models,
        endpoint_models=endpoint_models,
        endpoint_reachable=endpoint_reachable,
    )


def diff_discovery(
    old: DiscoveryResult,
    new: DiscoveryResult,
    routed_models: set[str],
) -> DiscoveryDiff:
    """Compare two discovery results, flagging changes relevant to routing."""
    old_set = set(old.models)
    new_set = set(new.models)

    removed_models = sorted(old_set - new_set)

    new_endpoints_for: dict[str, list[str]] = {}
    lost_endpoints_for: dict[str, list[str]] = {}
    for model in old_set & new_set:
        gained = sorted(set(new.models[model]) - set(old.models[model]))
        lost = sorted(set(old.models[model]) - set(new.models[model]))
        if gained:
            new_endpoints_for[model] = gained
        if lost:
            lost_endpoints_for[model] = lost

    return DiscoveryDiff(
        new_models=sorted(new_set - old_set),
        removed_models=removed_models,
        routing_lost=sorted(routed_models & set(removed_models)),
        new_endpoints_for=new_endpoints_for,
        lost_endpoints_for=lost_endpoints_for,
    )


# ---------------------------------------------------------------------------
# Snapshot persistence
# ---------------------------------------------------------------------------

_SNAPSHOT_VERSION = 1


def save_snapshot(path: Path, result: DiscoveryResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": _SNAPSHOT_VERSION,
        "scanned_at": result.scanned_at,
        "models": result.models,
        "endpoint_models": result.endpoint_models,
        "endpoint_reachable": result.endpoint_reachable,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.debug("Saved discovery snapshot to %s", path)


def load_snapshot(path: Path) -> DiscoveryResult | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return DiscoveryResult(
            scanned_at=data["scanned_at"],
            models=data["models"],
            endpoint_models=data["endpoint_models"],
            endpoint_reachable=data.get("endpoint_reachable", {}),
        )
    except Exception as exc:
        logger.warning("Could not load discovery snapshot (%s): %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Human-readable log output
# ---------------------------------------------------------------------------

_SEP = "-" * 60


def log_first_discovery(result: DiscoveryResult) -> None:
    """Print a friendly summary when no previous snapshot exists."""
    lines = [
        "",
        _SEP,
        "  MODEL DISCOVERY — first run",
        _SEP,
    ]
    reachable = [ep for ep, ok in result.endpoint_reachable.items() if ok]
    unreachable = [ep for ep, ok in result.endpoint_reachable.items() if not ok]

    lines.append(f"  Endpoints reachable : {', '.join(reachable) or 'none'}")
    if unreachable:
        lines.append(f"  Endpoints OFFLINE   : {', '.join(unreachable)}")

    lines.append(f"\n  {len(result.models)} model(s) found:\n")
    for model_id, eps in sorted(result.models.items()):
        lines.append(f"    {model_id:<35} [{', '.join(eps)}]")

    if result.models:
        lines += [
            "",
            "  To configure per-model routing, run:",
            "    llm-proxy discover --config <your-config.yaml>",
            "  and add a 'routing:' section to your config.",
        ]
    lines.append(_SEP)
    logger.info("\n".join(lines))


def log_discovery_diff(diff: DiscoveryDiff) -> None:
    """Log a structured diff; escalate to error if routing is broken."""
    if not diff.has_changes and not diff.requires_onboarding:
        logger.info("Model discovery: no changes since last scan")
        return

    lines = ["", _SEP, "  MODEL DISCOVERY — changes detected", _SEP]

    if diff.new_models:
        lines.append("  NEW models:")
        for m in diff.new_models:
            lines.append(f"    + {m}")

    if diff.removed_models:
        lines.append("  REMOVED models (no longer available on any endpoint):")
        for m in diff.removed_models:
            tag = "  !!ROUTING BROKEN!!" if m in diff.routing_lost else ""
            lines.append(f"    - {m}{tag}")

    for model, eps in diff.new_endpoints_for.items():
        lines.append(f"  {model}: now ALSO available on [{', '.join(eps)}]")

    for model, eps in diff.lost_endpoints_for.items():
        lines.append(f"  {model}: no longer available on [{', '.join(eps)}]")

    lines.append(_SEP)

    if diff.requires_onboarding:
        lines += [
            "",
            "  *** ROUTING ALERT ***",
            "  The following model(s) are configured in your routing rules",
            "  but are no longer available on any endpoint:",
            "",
        ]
        for m in diff.routing_lost:
            lines.append(f"    - {m}")
        lines += [
            "",
            "  Please run `llm-proxy discover --config <your-config.yaml>`",
            "  to see the current model list and update your routing config.",
            "",
        ]
        logger.error("\n".join(lines))
    else:
        logger.warning("\n".join(lines))
