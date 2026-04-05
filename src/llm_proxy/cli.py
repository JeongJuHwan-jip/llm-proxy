"""CLI entry point for llm-proxy."""

from __future__ import annotations

import asyncio
import logging
import sys

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
        cfg.server.host = host
    if port is not None:
        cfg.server.port = port

    # Inject config so the factory can pick it up
    from .server import create_app as _create_app
    _create_app._config = cfg  # type: ignore[attr-defined]

    click.echo(
        f"Starting LLM Proxy on http://{cfg.server.host}:{cfg.server.port}  "
        f"(dashboard: http://localhost:{cfg.server.port}/dashboard/index.html)"
    )

    uvicorn.run(
        "llm_proxy.server:create_app",
        factory=True,
        host=cfg.server.host,
        port=cfg.server.port,
        workers=workers,
        log_config=None,  # use our own logging config
        access_log=True,
    )


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
    for ep in sorted(cfg.endpoints, key=lambda e: e.priority):
        click.echo(
            f"  [{ep.priority}] {ep.name}  {ep.url}  "
            f"timeout={ep.timeout_ms}ms  headers={list(ep.headers.keys())}"
        )
    click.echo(
        f"\nFailover: max_retries={cfg.failover.max_retries}  "
        f"cb_threshold={cfg.failover.circuit_breaker_threshold}  "
        f"cb_cooldown={cfg.failover.circuit_breaker_cooldown}s  "
        f"strategy={cfg.failover.routing_strategy}"
    )
    click.echo(
        f"Logging: db={cfg.logging.db_path}  "
        f"log_body={cfg.logging.log_request_body}"
    )
    auth_status = f"{len(cfg.auth.api_keys)} key(s)" if cfg.auth else "disabled"
    click.echo(f"Auth: {auth_status}")
