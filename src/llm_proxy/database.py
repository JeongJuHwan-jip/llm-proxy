"""SQLite persistence layer for request logs.

Uses WAL mode for concurrent read/write performance.
All writes are dispatched to a thread-pool executor so the async event loop
is never blocked by SQLite I/O.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import AttemptLog, RequestLog

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS requests (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           REAL    NOT NULL,
    model               TEXT    NOT NULL,
    selected_endpoint   TEXT,
    attempts_json       TEXT    NOT NULL,
    status              TEXT    NOT NULL,
    total_latency_ms    REAL    NOT NULL,
    is_stream           INTEGER NOT NULL,
    request_body_json   TEXT
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_timestamp ON requests (timestamp DESC);
"""


class Database:
    """Thin wrapper around a SQLite connection for request logging."""

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create the DB file, enable WAL, and create the schema."""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._path,
            check_same_thread=False,  # we protect writes with GIL + executor
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        self._conn.commit()
        logger.info("Database initialised at %r", self._path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_request_log(self, log: RequestLog) -> None:
        """Insert a completed request log (called from executor thread)."""
        assert self._conn is not None, "Database not initialised"

        attempts_json = json.dumps(
            [
                {
                    "endpoint": a.endpoint_name,
                    "latency_ms": round(a.latency_ms, 2),
                    "success": a.success,
                    "is_timeout": a.is_timeout,
                    "error": a.error_message,
                }
                for a in log.attempts
            ]
        )
        request_body_json = (
            json.dumps(log.request_body) if log.request_body is not None else None
        )

        self._conn.execute(
            """
            INSERT INTO requests
                (timestamp, model, selected_endpoint, attempts_json,
                 status, total_latency_ms, is_stream, request_body_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log.timestamp,
                log.model,
                log.selected_endpoint,
                attempts_json,
                log.status,
                round(log.total_latency_ms, 2),
                1 if log.is_stream else 0,
                request_body_json,
            ),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_recent_requests(
        self, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT id, timestamp, model, selected_endpoint,
                   attempts_json, status, total_latency_ms, is_stream
            FROM requests
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

        result = []
        for row in rows:
            d = dict(row)
            d["attempts"] = json.loads(d.pop("attempts_json"))
            d["is_stream"] = bool(d["is_stream"])
            result.append(d)
        return result

    def get_endpoint_stats(self) -> list[dict[str, Any]]:
        """Aggregate stats per endpoint from the DB (last 24 h)."""
        assert self._conn is not None
        since = time.time() - 86400  # 24 hours
        rows = self._conn.execute(
            """
            SELECT
                selected_endpoint                       AS endpoint,
                COUNT(*)                                AS total,
                SUM(CASE WHEN status='failure' THEN 1 ELSE 0 END) AS failures,
                AVG(total_latency_ms)                   AS avg_latency_ms
            FROM requests
            WHERE timestamp >= ? AND selected_endpoint IS NOT NULL
            GROUP BY selected_endpoint
            ORDER BY total DESC
            """,
            (since,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_total_count(self) -> int:
        assert self._conn is not None
        row = self._conn.execute("SELECT COUNT(*) FROM requests").fetchone()
        return row[0] if row else 0
