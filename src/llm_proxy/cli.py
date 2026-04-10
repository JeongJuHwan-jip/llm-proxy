"""CLI entry point for llm-proxy."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
import uvicorn

from .config import load_config


@click.group()
@click.option(
    "--log-level",
    default="info",
    type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
    show_default=True,
    help="Log verbosity level.",
)
def main(log_level: str) -> None:
    """LLM API Proxy — route & failover across multiple LLM endpoints."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


@main.command("start")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option("--host", default=None, help="Override server.host from config")
@click.option("--port", default=None, type=int, help="Override server.port from config")
@click.option("--workers", default=1, type=int, show_default=True, help="Number of uvicorn workers")
def start(config: str, host: str | None, port: int | None, workers: int) -> None:
    """Start the LLM proxy server."""
    # Windows asyncio compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    cfg = load_config(config)
    if host is not None:
        cfg.proxy.host = host
    if port is not None:
        cfg.proxy.port = port

    # Inject config and config_path so the factory can pick them up
    from .server import create_app as _create_app
    _create_app._config = cfg          # type: ignore[attr-defined]
    _create_app._config_path = Path(config).resolve()  # type: ignore[attr-defined]

    click.echo(
        f"Starting LLM Proxy on http://{cfg.proxy.host}:{cfg.proxy.port}  "
        f"(dashboard: http://localhost:{cfg.proxy.port}/dashboard/index.html)"
    )

    uvicorn.run(
        "llm_proxy.server:create_app",
        factory=True,
        host=cfg.proxy.host,
        port=cfg.proxy.port,
        workers=workers,
        log_config=None,  # use our own logging config
        access_log=True,
    )


@main.command("discover")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
@click.option(
    "--snippet",
    is_flag=True,
    default=False,
    help="Print a ready-to-paste 'routing:' config snippet after the table.",
)
def discover(config: str, snippet: bool) -> None:
    """Discover models from all configured endpoints and show a routing overview.

    Run this to see which models are available and on which endpoints,
    then use the output to build (or update) the 'routing:' section in
    your config file.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import httpx as _httpx

    from .discovery import run_discovery
    from .router import Router

    cfg = load_config(config)
    router = Router(cfg)

    async def _run():
        async with _httpx.AsyncClient() as client:
            return await run_discovery(client, router.all_endpoints())

    result = asyncio.run(_run())

    # ---- endpoint reachability table ----
    click.echo()
    click.echo("Endpoint status:")
    for ep_name, reachable in sorted(result.endpoint_reachable.items()):
        status = "OK         " if reachable else "UNREACHABLE"
        model_count = len(result.endpoint_models.get(ep_name, []))
        click.echo(f"  {status}  {ep_name:<25} ({model_count} model(s))")

    # ---- model → endpoint table ----
    click.echo()
    if not result.models:
        click.echo("No models found on any reachable endpoint.")
    else:
        click.echo(f"Available models ({len(result.models)} total):\n")
        col = max(len(m) for m in result.models) + 2
        for model_id, eps in sorted(result.models.items()):
            ep_str = " -> ".join(eps)
            click.echo(f"  {model_id:<{col}} {ep_str}")

    # ---- routing snippet ----
    if snippet and result.models:
        click.echo()
        click.echo("# --- paste into your config.yaml ---")
        click.echo("routing:")
        for model_id, eps in sorted(result.models.items()):
            click.echo(f"  - model: \"{model_id}\"")
            click.echo(f"    endpoints: {eps}")
        click.echo("# ------------------------------------")


@main.command("validate")
@click.option(
    "--config",
    "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml",
)
def validate(config: str) -> None:
    """Validate a config file and print a summary."""
    try:
        cfg = load_config(config)
    except Exception as exc:  # noqa: BLE001
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Config OK — {len(cfg.endpoints)} endpoint(s):")
    for ep in cfg.endpoints:
        click.echo(
            f"  {ep.name}  {ep.url}  "
            f"timeout={ep.timeout_ms}ms  headers={list(ep.headers.keys())}"
        )

    # Check for settings.json
    from .config import load_settings_file, resolve_settings_path
    settings_path = resolve_settings_path(Path(config).resolve())
    if settings_path.exists():
        try:
            sdata = load_settings_file(settings_path)
            fo = sdata.failover or cfg.failover
            click.echo(
                f"\nSettings ({settings_path.name}): {len(sdata.routes)} route(s)  "
                f"max_retries={fo.max_retries}  "
                f"cb_threshold={fo.circuit_breaker_threshold}  "
                f"cb_cooldown={fo.circuit_breaker_cooldown}s  "
                f"strategy={fo.routing_strategy}"
            )
        except Exception as exc:
            click.echo(f"\nSettings: error reading {settings_path}: {exc}", err=True)
    else:
        click.echo(f"\nSettings: {settings_path.name} not found (will use defaults)")

    click.echo(
        f"Logging: db={cfg.logging.db_path}  "
        f"log_body={cfg.logging.log_request_body}"
    )
    auth_status = f"{len(cfg.auth.api_keys)} key(s)" if cfg.auth else "disabled"
    click.echo(f"Auth: {auth_status}")
