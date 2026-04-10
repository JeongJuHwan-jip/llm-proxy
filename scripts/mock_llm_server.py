"""
Mock LLM server for local testing without real LLM endpoints.

Simulates an OpenAI-compatible /v1/chat/completions endpoint
with configurable behavior: success, delay, timeout, error.

Usage:
    # Normal server on port 8001 with default models (mock-model, gpt-4)
    python scripts/mock_llm_server.py --port 8001

    # Server that always times out (for failover testing)
    python scripts/mock_llm_server.py --port 8002 --behavior timeout

    # Server that fails 70% of the time
    python scripts/mock_llm_server.py --port 8003 --behavior flaky --fail-rate 0.7

    # Flaky server where 40% of failures are timeouts (not just HTTP errors)
    python scripts/mock_llm_server.py --port 8003 --behavior flaky --fail-rate 0.5 --timeout-rate 0.4

    # Server with 3s artificial delay
    python scripts/mock_llm_server.py --port 8004 --behavior slow --delay 3.0

    # Random latency on all responses (100ms–800ms)
    python scripts/mock_llm_server.py --port 8001 --latency-min 0.1 --latency-max 0.8

    # Combine: flaky server with random latency on successful responses
    python scripts/mock_llm_server.py --port 8003 --behavior flaky --fail-rate 0.3 \
        --timeout-rate 0.3 --latency-min 0.05 --latency-max 1.5

    # Server with a custom model list (simulates different endpoints having
    # different models — pass comma-separated model IDs)
    python scripts/mock_llm_server.py --port 8001 --models gpt-4,claude-3-opus
    python scripts/mock_llm_server.py --port 8002 --models gpt-4,llama-3,mistral-7b
    python scripts/mock_llm_server.py --port 8003 --models mistral-7b

Behaviors:
    ok       — always returns a valid response (default)
    timeout  — hangs forever (triggers proxy timeout)
    error    — always returns HTTP 500
    flaky    — randomly fails at --fail-rate probability;
               --timeout-rate controls what fraction of failures are
               timeouts (slow response) vs HTTP 503 errors
    slow     — adds --delay seconds before responding

Latency:
    --latency-min / --latency-max add random delay to ALL successful
    responses (applied on top of behavior-specific delays).
"""

import argparse
import asyncio
import json
import random
import time
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

parser = argparse.ArgumentParser(description="Mock LLM server")
parser.add_argument("--port", type=int, default=8001)
parser.add_argument(
    "--behavior",
    choices=["ok", "timeout", "error", "flaky", "slow"],
    default="ok",
)
parser.add_argument("--fail-rate", type=float, default=0.5,
                    help="Failure probability for --behavior flaky (0.0–1.0)")
parser.add_argument("--delay", type=float, default=2.0,
                    help="Delay in seconds for --behavior slow")
parser.add_argument("--timeout-rate", type=float, default=0.0,
                    help="Fraction of flaky failures that are timeouts vs HTTP 503 (0.0–1.0)")
parser.add_argument("--latency-min", type=float, default=0.0,
                    help="Minimum random latency added to successful responses (seconds)")
parser.add_argument("--latency-max", type=float, default=0.0,
                    help="Maximum random latency added to successful responses (seconds)")
parser.add_argument("--name", type=str, default=None,
                    help="Server name shown in responses (defaults to port)")
parser.add_argument("--models", type=str, default=None,
                    help="Comma-separated model IDs to advertise (default: mock-model,gpt-4)")
args = parser.parse_args()

SERVER_NAME = args.name or f"mock-{args.port}"
MODEL_IDS: list[str] = (
    [m.strip() for m in args.models.split(",") if m.strip()]
    if args.models
    else ["mock-model", "gpt-4"]
)

app = FastAPI(title=f"Mock LLM ({SERVER_NAME})")

_request_count = 0


def _make_response(model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        "x-served-by": SERVER_NAME,
    }


async def _stream_chunks(model: str, content: str) -> AsyncIterator[str]:
    """Yield SSE chunks simulating a streaming response."""
    words = content.split()
    for i, word in enumerate(words):
        chunk = {
            "id": f"chatcmpl-mock-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)

    # Final chunk
    done_chunk = {
        "id": f"chatcmpl-mock-{int(time.time())}",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    global _request_count
    _request_count += 1
    count = _request_count

    try:
        body = await request.json()
    except Exception as exc:
        print(f"[{SERVER_NAME}] #{count} Failed to parse body: {exc}")
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"Invalid JSON body: {exc}", "type": "invalid_request_error"}},
        )

    if not isinstance(body, dict):
        body = {}

    model = body.get("model", "mock-model")
    is_stream = body.get("stream", False)

    print(f"[{SERVER_NAME}] #{count} model={model!r} stream={is_stream} behavior={args.behavior}")

    # ── Apply behavior ──────────────────────────────────────────────────────
    if args.behavior == "timeout":
        print(f"[{SERVER_NAME}] Hanging forever...")
        await asyncio.sleep(9999)  # proxy will timeout first

    elif args.behavior == "error":
        print(f"[{SERVER_NAME}] Returning 500")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "server_error"}},
        )

    elif args.behavior == "flaky":
        if random.random() < args.fail_rate:
            if args.timeout_rate > 0 and random.random() < args.timeout_rate:
                print(f"[{SERVER_NAME}] Flaky: timeout (sleeping {args.delay}s)...")
                await asyncio.sleep(args.delay)
                # Still return after the long delay — proxy should have timed out by now
                return JSONResponse(
                    status_code=504,
                    content={"error": {"message": "Gateway timeout (mock)"}},
                )
            else:
                print(f"[{SERVER_NAME}] Flaky: returning 503")
                return JSONResponse(
                    status_code=503,
                    content={"error": {"message": "Service temporarily unavailable"}},
                )

    elif args.behavior == "slow":
        print(f"[{SERVER_NAME}] Sleeping {args.delay}s...")
        await asyncio.sleep(args.delay)

    # ── Random latency on successful responses ──────────────────────────────
    if args.latency_max > 0:
        lat = random.uniform(args.latency_min, args.latency_max)
        print(f"[{SERVER_NAME}] Random latency: {lat:.3f}s")
        await asyncio.sleep(lat)

    # ── Normal response ─────────────────────────────────────────────────────
    content = f"Hello from {SERVER_NAME}! (request #{count})"

    if is_stream:
        return StreamingResponse(
            _stream_chunks(model, content),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    return JSONResponse(_make_response(model, content))


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": int(time.time()), "owned_by": SERVER_NAME}
            for mid in MODEL_IDS
        ],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "server": SERVER_NAME, "requests": _request_count}


if __name__ == "__main__":
    parts = [f"Starting {SERVER_NAME} on port {args.port}",
             f"behavior={args.behavior}"]
    if args.behavior == "flaky":
        parts.append(f"fail_rate={args.fail_rate} timeout_rate={args.timeout_rate}")
    if args.latency_max > 0:
        parts.append(f"latency={args.latency_min:.2f}–{args.latency_max:.2f}s")
    parts.append(f"models={MODEL_IDS}")
    print(" | ".join(parts))
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
