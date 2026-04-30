"""Mock upstream LLM servers for end-to-end testing.

Creates configurable FastAPI apps that simulate various upstream behaviors
(success, timeout, HTTP errors) and manages their lifecycle on real ports.

Usage in pytest::

    from mock_upstream import create_mock_upstream, MockServer, get_free_port

    @pytest.fixture(scope="module")
    def mock_servers():
        servers = []
        for name, behavior in [("ok-ep", "ok"), ("bad-ep", "error")]:
            server = MockServer(
                create_mock_upstream(behavior=behavior, name=name),
                get_free_port(),
            )
            server.start()
            servers.append(server)
        yield servers
        for s in servers:
            s.stop()
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


def create_mock_upstream(
    behavior: str = "ok",
    name: str = "mock",
    models: list[str] | None = None,
    status_code: int = 500,
    delay: float = 30.0,
) -> FastAPI:
    """Create a mock upstream LLM server with configurable behavior.

    Parameters
    ----------
    behavior
        ``"ok"``      - always returns 200 with a valid chat completion.
        ``"timeout"`` - sleeps *delay* seconds (proxy should time out first).
        ``"error"``   - always returns *status_code*.
        ``"slow"``    - sleeps *delay* seconds then returns 200.
    name
        Server identity embedded in responses (``x-served-by``).
    models
        Model IDs advertised on ``/v1/models``.
    status_code
        HTTP status returned by the ``"error"`` behavior.
    delay
        Sleep duration for ``"timeout"`` and ``"slow"`` behaviors.
    """
    models = models or ["mock-model"]
    app = FastAPI(title=f"Mock Upstream ({name})")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model", "mock-model")
        is_stream = body.get("stream", False)

        # ── Apply behavior ─────────────────────────────────────────────
        if behavior == "timeout":
            await asyncio.sleep(delay)
            return JSONResponse(
                status_code=504,
                content={"error": {"message": "Gateway timeout (mock)", "type": "server_error"}},
            )

        if behavior == "error":
            return JSONResponse(
                status_code=status_code,
                content={"error": {"message": f"Mock {status_code} from {name}", "type": "server_error"}},
            )

        if behavior == "slow":
            await asyncio.sleep(delay)

        # ── Success response ───────────────────────────────────────────
        content_text = f"Hello from {name}"
        response_body = {
            "id": f"chatcmpl-{name}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "x-served-by": name,
        }

        if is_stream:
            cut_midstream = behavior == "stream_cut"
            cut_in_tool_call = behavior == "stream_cut_in_tool_call"
            tool_call_ok = behavior == "tool_call_ok"

            def _tc_chunk(delta: dict, finish_reason: str | None = None) -> str:
                payload = {
                    "id": f"chatcmpl-{name}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }
                return f"data: {json.dumps(payload)}\n\n"

            async def stream_chunks() -> AsyncIterator[str]:
                if cut_in_tool_call:
                    # Start a tool_call, then cut mid-arguments — the client
                    # would otherwise receive a partial tool_call.
                    yield _tc_chunk({"tool_calls": [{
                        "index": 0, "id": "call_x",
                        "function": {"name": "apply_diff", "arguments": ""},
                    }]})
                    yield _tc_chunk({"tool_calls": [{
                        "index": 0, "function": {"arguments": '{"path":"a.t'},
                    }]})
                    raise RuntimeError(f"simulated tool_call mid-stream cut on {name}")

                if tool_call_ok:
                    yield _tc_chunk({"tool_calls": [{
                        "index": 0, "id": "call_y",
                        "function": {"name": "apply_diff", "arguments": ""},
                    }]})
                    yield _tc_chunk({"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '{"path":"b.txt","diff":"+ok"}'},
                    }]})
                    yield _tc_chunk({}, finish_reason="tool_calls")
                    yield "data: [DONE]\n\n"
                    return

                chunk = {
                    "id": f"chatcmpl-{name}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                if cut_midstream:
                    # Drop the connection without sending the final chunk
                    # or [DONE]; raise so uvicorn aborts the response stream
                    # ungracefully (mimics RemoteProtocolError on the client).
                    raise RuntimeError(f"simulated mid-stream cut on {name}")
                done = {
                    "id": f"chatcmpl-{name}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                stream_chunks(),
                media_type="text/event-stream",
                headers={"X-Accel-Buffering": "no"},
            )

        return JSONResponse(response_body)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [
                {"id": m, "object": "model", "created": int(time.time()), "owned_by": name}
                for m in models
            ],
        })

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok", "server": name})

    return app


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


def get_free_port() -> int:
    """Return a random available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockServer:
    """Run a FastAPI app on a real port in a background thread.

    Usage::

        server = MockServer(app, port)
        server.start()       # blocks until ready
        # ... use server.url ...
        server.stop()
    """

    def __init__(self, app: FastAPI, port: int) -> None:
        self.port = port
        self.url = f"http://127.0.0.1:{port}/v1"
        self._config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="error",
        )
        self._server = uvicorn.Server(self._config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def start(self, timeout: float = 10.0) -> None:
        """Start the server and block until it is accepting connections."""
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if time.monotonic() > deadline:
                raise RuntimeError(f"Mock server on port {self.port} did not start in {timeout}s")
            time.sleep(0.01)

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=5)
