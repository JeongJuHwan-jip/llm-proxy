"""Anthropic Messages API adapter.

Translates between Anthropic Messages API format and OpenAI Chat Completions
format, enabling clients that speak the Anthropic protocol to use OpenAI-
compatible upstream endpoints through the proxy's existing failover and
circuit-breaker infrastructure.

Client (Anthropic)  -->  translate_request()  -->  router.execute()  (OpenAI upstream)
                    <--  translate_response() <--
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

import httpx
from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..models import AttemptLog, RequestLog, RouteStep
from ..router import AllEndpointsFailedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error helper
# ---------------------------------------------------------------------------

def anthropic_error(type_str: str, message: str, status_code: int = 400) -> JSONResponse:
    """Return a JSONResponse in Anthropic error format."""
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": type_str, "message": message}},
    )


# ---------------------------------------------------------------------------
# Request translation  (Anthropic -> OpenAI)
# ---------------------------------------------------------------------------

_FINISH_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def translate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages API request body to OpenAI format.

    Pure function — no side effects.
    """
    oai: dict[str, Any] = {}

    # --- Pass-through fields ------------------------------------------------
    # ``thinking`` is forwarded as-is so OpenAI-compatible upstreams that
    # support extended thinking (LiteLLM, OpenRouter, Anthropic-via-OpenAI,
    # etc.) can honor it. Upstreams that don't recognize the field typically
    # ignore it.
    for key in ("model", "max_tokens", "temperature", "top_p", "stream", "thinking"):
        if key in body:
            oai[key] = body[key]

    # --- Renamed fields -----------------------------------------------------
    if "stop_sequences" in body:
        oai["stop"] = body["stop_sequences"]

    # --- System prompt → system message -------------------------------------
    messages: list[dict[str, Any]] = []

    system = body.get("system")
    if system is not None:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Array of text content blocks
            text = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # --- Messages -----------------------------------------------------------
    for msg in body.get("messages", []):
        converted = _translate_message(msg)
        if isinstance(converted, list):
            messages.extend(converted)
        else:
            messages.append(converted)

    oai["messages"] = messages

    # --- Tools --------------------------------------------------------------
    if "tools" in body:
        oai["tools"] = [_translate_tool_def(t) for t in body["tools"]]

    # --- Tool choice --------------------------------------------------------
    if "tool_choice" in body:
        oai["tool_choice"] = _translate_tool_choice(body["tool_choice"])

    return oai


def _translate_message(msg: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
    """Translate a single Anthropic message to OpenAI message(s).

    tool_result blocks expand to separate ``role: "tool"`` messages.
    tool_use blocks in assistant messages become ``tool_calls``.
    """
    role = msg.get("role", "user")
    content = msg.get("content")

    # Simple string content — pass through
    if isinstance(content, str):
        return {"role": role, "content": content}

    if not isinstance(content, list):
        return {"role": role, "content": content}

    # --- Assistant messages: extract tool_use blocks into tool_calls ---------
    if role == "assistant":
        return _translate_assistant_message(content)

    # --- User messages: expand tool_result blocks into tool messages ---------
    tool_results: list[dict[str, Any]] = []
    other_blocks: list[dict[str, Any]] = []

    for block in content:
        if block.get("type") == "tool_result":
            tool_results.append(block)
        else:
            other_blocks.append(block)

    result: list[dict[str, Any]] = []

    # Non-tool-result content stays as a user message
    if other_blocks:
        result.append({
            "role": "user",
            "content": _translate_content_blocks(other_blocks),
        })

    # Each tool_result → separate tool message
    for tr in tool_results:
        tc_content = tr.get("content", "")
        if isinstance(tc_content, list):
            # Extract text from content blocks
            tc_content = "\n".join(
                b.get("text", "") for b in tc_content if b.get("type") == "text"
            )
        result.append({
            "role": "tool",
            "tool_call_id": tr.get("tool_use_id", ""),
            "content": str(tc_content),
        })

    return result if result else {"role": "user", "content": ""}


def _translate_assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert an assistant message with content blocks to OpenAI format."""
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    msg: dict[str, Any] = {"role": "assistant"}
    msg["content"] = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _translate_content_blocks(blocks: list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Translate Anthropic content blocks to OpenAI content format.

    If all blocks are text, returns a plain string.
    If images are present, returns OpenAI multi-modal content array.
    """
    has_image = any(b.get("type") == "image" for b in blocks)

    if not has_image:
        # All text — collapse to string
        return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")

    # Multi-modal content array
    result: list[dict[str, Any]] = []
    for block in blocks:
        btype = block.get("type")
        if btype == "text":
            result.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media = source.get("media_type", "image/png")
                data = source.get("data", "")
                result.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media};base64,{data}"},
                })
            elif source.get("type") == "url":
                result.append({
                    "type": "image_url",
                    "image_url": {"url": source.get("url", "")},
                })
    return result


def _translate_tool_def(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic tool definition to OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _translate_tool_choice(tc: dict[str, Any] | str) -> str | dict[str, Any]:
    """Translate Anthropic tool_choice to OpenAI format."""
    if isinstance(tc, str):
        return tc
    tc_type = tc.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    if tc_type == "none":
        return "none"
    return "auto"


# ---------------------------------------------------------------------------
# Response translation  (OpenAI -> Anthropic)
# ---------------------------------------------------------------------------

def translate_response(oai_body: dict[str, Any], original_model: str) -> dict[str, Any]:
    """Translate an OpenAI chat completion response to Anthropic Messages format."""
    choice = {}
    if oai_body.get("choices"):
        choice = oai_body["choices"][0]

    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    # Build content blocks
    content: list[dict[str, Any]] = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_obj = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_obj = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": input_obj,
        })

    # Usage
    oai_usage = oai_body.get("usage", {})
    usage = {
        "input_tokens": oai_usage.get("prompt_tokens", 0),
        "output_tokens": oai_usage.get("completion_tokens", 0),
    }

    oai_id = oai_body.get("id", uuid.uuid4().hex[:24])
    return {
        "id": f"msg_{oai_id}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": original_model,
        "stop_reason": _FINISH_REASON_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Streaming SSE translator  (OpenAI SSE -> Anthropic SSE)
# ---------------------------------------------------------------------------

class SSEBuffer:
    """Accumulates raw bytes and yields complete SSE data lines."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, chunk: bytes) -> list[str]:
        """Feed raw bytes, return list of complete SSE event strings."""
        self._buf += chunk.decode("utf-8", errors="replace")
        events: list[str] = []
        while "\n\n" in self._buf:
            event_str, self._buf = self._buf.split("\n\n", 1)
            events.append(event_str.strip())
        return events


def _parse_sse_data(event_str: str) -> dict[str, Any] | None:
    """Extract JSON from a 'data: {...}' SSE line. Returns None for [DONE]."""
    for line in event_str.split("\n"):
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                return None
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
    return None


def _sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format a single Anthropic SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def anthropic_sse_generator(
    upstream_resp: httpx.Response,
    original_model: str,
) -> AsyncIterator[str]:
    """Wrap an OpenAI SSE byte stream, yielding Anthropic SSE events."""

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    buf = SSEBuffer()

    # State
    block_index = -1
    current_block_type: str | None = None
    output_tokens = 0
    finish_reason_str: str | None = None
    block_started = False

    # --- Preamble ---
    yield _sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": original_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    })
    yield _sse_event("ping", {"type": "ping"})

    try:
        async for raw_chunk in upstream_resp.aiter_bytes():
            for event_str in buf.feed(raw_chunk):
                chunk_data = _parse_sse_data(event_str)
                if chunk_data is None:
                    # [DONE] — finish up
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    # usage-only chunk (some providers send this)
                    usage = chunk_data.get("usage")
                    if usage:
                        output_tokens = usage.get("completion_tokens", output_tokens)
                    continue

                delta = choices[0].get("delta", {})
                fr = choices[0].get("finish_reason")

                # Capture usage from the chunk if present
                usage = chunk_data.get("usage")
                if usage:
                    output_tokens = usage.get("completion_tokens", output_tokens)

                # --- Text content delta ---
                text_content = delta.get("content")
                if text_content is not None:
                    if current_block_type != "text":
                        # Close previous block if any
                        if block_started:
                            yield _sse_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": block_index,
                            })
                        block_index += 1
                        current_block_type = "text"
                        block_started = True
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        })

                    if text_content:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "text_delta", "text": text_content},
                        })

                # --- Tool call deltas ---
                tool_calls = delta.get("tool_calls", [])
                for tc in tool_calls:
                    tc_index = tc.get("index", 0)
                    func = tc.get("function", {})
                    tc_name = func.get("name")
                    tc_args = func.get("arguments", "")

                    # New tool call → start a new content block
                    if tc_name is not None:
                        if block_started:
                            yield _sse_event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": block_index,
                            })
                        block_index += 1
                        current_block_type = "tool_use"
                        block_started = True
                        yield _sse_event("content_block_start", {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                "name": tc_name,
                                "input": {},
                            },
                        })

                    # Arguments chunk
                    if tc_args:
                        yield _sse_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "input_json_delta", "partial_json": tc_args},
                        })

                # --- Finish reason ---
                if fr is not None:
                    finish_reason_str = fr

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("Upstream SSE interrupted: %s: %s", type(exc).__name__, exc)
        if finish_reason_str is None:
            finish_reason_str = "stop"
    finally:
        # Close the last content block
        if block_started:
            try:
                yield _sse_event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": block_index,
                })
            except Exception:
                pass

        # message_delta with stop_reason
        stop_reason = _FINISH_REASON_MAP.get(finish_reason_str or "stop", "end_turn")
        try:
            yield _sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            })
            yield _sse_event("message_stop", {"type": "message_stop"})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

# Headers to strip (Anthropic-specific headers added)
_ANTHROPIC_STRIP_HEADERS = {
    "host", "content-length", "transfer-encoding", "connection",
    "authorization", "anthropic-version", "anthropic-beta",
    "x-api-key",
}


async def handle_anthropic_messages(request: Request) -> Response:
    """Top-level handler for ``POST /v1/messages``."""
    # Lazy imports to avoid circular dependency
    from ..router import Router, _should_failover
    from ..server import EventBroadcaster, _resolve_route, _STRIP_HEADERS, merge_headers
    from ..config import ProxyConfig, resolve_headers as _resolve

    clients: dict[bool, httpx.AsyncClient] = request.app.state.clients
    router: Router = request.app.state.router
    db = request.app.state.db
    cfg: ProxyConfig = request.app.state.config
    events: EventBroadcaster = request.app.state.events

    # Parse body
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        return anthropic_error("invalid_request_error", "Invalid JSON body")

    # Store original model before translation
    original_model = body.get("model", "unknown")

    # Translate Anthropic -> OpenAI
    try:
        oai_body = translate_request(body)
    except ValueError as exc:
        return anthropic_error("invalid_request_error", str(exc))

    # Resolve route
    raw_model = oai_body.get("model", "unknown")
    try:
        steps, fallback_model, is_direct = _resolve_route(raw_model, router)
    except Exception:
        return anthropic_error(
            "not_found_error",
            f"Unknown model {raw_model!r}. Use 'endpoint/model' or a named route.",
            404,
        )

    is_stream = bool(oai_body.get("stream", False))
    forward_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in _ANTHROPIC_STRIP_HEADERS
    }

    t_start = time.monotonic()
    log_body = body if cfg.logging.log_request_body else None

    if is_stream:
        return await _handle_anthropic_stream(
            clients, router, db, cfg, events,
            oai_body, forward_headers, original_model,
            log_body, t_start, steps, fallback_model, is_direct,
        )
    else:
        try:
            result = await _handle_anthropic_normal(
                clients, router, db, cfg,
                oai_body, forward_headers, original_model,
                log_body, t_start, steps, fallback_model, is_direct,
            )
        finally:
            events.publish()
        return result


async def _handle_anthropic_normal(
    clients: "dict[bool, httpx.AsyncClient]",
    router: "Router",
    db: Any,
    cfg: Any,
    oai_body: dict,
    forward_headers: dict[str, str],
    original_model: str,
    log_body: dict | None,
    t_start: float,
    steps: list[RouteStep],
    fallback_model: str,
    is_direct: bool,
) -> Response:
    """Non-streaming Anthropic handler — calls router.execute(), translates response."""
    loop = asyncio.get_event_loop()

    try:
        response, winning_step, attempts = await router.execute(
            clients, "/chat/completions", oai_body, forward_headers,
            steps, fallback_model, is_direct=is_direct,
        )
    except AllEndpointsFailedError as exc:
        total_ms = (time.monotonic() - t_start) * 1000
        log = RequestLog(
            timestamp=time.time(), model=original_model,
            selected_endpoint=None, attempts=exc.attempts,
            status="failure", total_latency_ms=total_ms,
            is_stream=False, request_body=log_body,
        )
        await loop.run_in_executor(None, db.insert_request_log, log)
        return anthropic_error("api_error", "All upstream endpoints failed", 502)

    total_ms = (time.monotonic() - t_start) * 1000
    log = RequestLog(
        timestamp=time.time(), model=original_model,
        selected_endpoint=winning_step.endpoint.name,
        attempts=attempts, status="success",
        total_latency_ms=total_ms, is_stream=False,
        request_body=log_body,
    )
    await loop.run_in_executor(None, db.insert_request_log, log)

    # Non-200 from upstream (direct request passthrough)
    if response.status_code != 200:
        try:
            err_body = response.json()
            err_msg = err_body.get("error", {}).get("message", f"HTTP {response.status_code}")
        except Exception:
            err_msg = f"Upstream returned HTTP {response.status_code}"
        return anthropic_error("api_error", err_msg, response.status_code)

    # Translate successful response
    try:
        oai_json = response.json()
    except Exception:
        return anthropic_error("api_error", "Invalid upstream response", 502)

    anthropic_body = translate_response(oai_json, original_model)
    return JSONResponse(content=anthropic_body, status_code=200)


async def _handle_anthropic_stream(
    clients: "dict[bool, httpx.AsyncClient]",
    router: "Router",
    db: Any,
    cfg: Any,
    events: Any,
    oai_body: dict,
    forward_headers: dict[str, str],
    original_model: str,
    log_body: dict | None,
    t_start: float,
    steps: list[RouteStep],
    fallback_model: str,
    is_direct: bool,
) -> StreamingResponse:
    """Streaming Anthropic handler — step iteration + SSE translation."""
    from ..config import resolve_headers as _resolve
    from ..router import _should_failover
    from ..server import merge_headers

    loop = asyncio.get_event_loop()

    if is_direct:
        eligible = steps
    else:
        eligible = router.filter_by_context(router.filter_steps(steps), oai_body)
    if not eligible:
        return anthropic_error("api_error", "No endpoints available", 502)

    max_attempts = min(len(eligible), router._failover.max_retries + 1)
    attempts: list[AttemptLog] = []

    # --- Step iteration (mirrors server._handle_stream._try_stream) ---
    upstream_resp: httpx.Response | None = None
    winning_step: RouteStep | None = None

    for step in eligible[:max_attempts]:
        ep = step.endpoint
        model_for_step = step.model or fallback_model
        body_for_step = {**oai_body, "model": model_for_step}

        headers = merge_headers(
            _resolve(ep.headers), forward_headers, cfg.proxy.header_priority,
        )
        url = f"{ep.url}/chat/completions"
        timeout = step.timeout_ms / 1000.0
        t0 = time.monotonic()

        try:
            _t = {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}
            _c = clients.get(ep.ssl_verify, clients[True])
            resp = await _c.send(
                _c.build_request(
                    "POST", url, json=body_for_step, headers=headers,
                    extensions={"timeout": _t},
                ),
                stream=True,
            )
            latency_ms = (time.monotonic() - t0) * 1000

            if _should_failover(resp.status_code):
                if not is_direct:
                    await resp.aclose()
                    router.record_failure(ep, is_timeout=False)
                attempts.append(AttemptLog(
                    endpoint_name=ep.name, latency_ms=latency_ms,
                    success=False, is_timeout=False,
                    error_message=f"HTTP {resp.status_code}",
                ))
                if is_direct:
                    # Return error in Anthropic format
                    await resp.aclose()
                    total_ms = (time.monotonic() - t_start) * 1000
                    log = RequestLog(
                        timestamp=time.time(), model=original_model,
                        selected_endpoint=ep.name, attempts=attempts,
                        status="failure", total_latency_ms=total_ms,
                        is_stream=True, request_body=log_body,
                    )
                    await loop.run_in_executor(None, db.insert_request_log, log)
                    events.publish()
                    return anthropic_error("api_error", f"Upstream returned HTTP {resp.status_code}", resp.status_code)
                continue

            if not is_direct:
                router.record_success(ep, latency_ms)
            attempts.append(AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=True, is_timeout=False,
            ))
            upstream_resp = resp
            winning_step = step
            break

        except httpx.TimeoutException as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            if not is_direct:
                router.record_failure(ep, is_timeout=True)
            attempts.append(AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=False, is_timeout=True, error_message=str(exc),
            ))
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            if not is_direct:
                router.record_failure(ep, is_timeout=False)
            attempts.append(AttemptLog(
                endpoint_name=ep.name, latency_ms=latency_ms,
                success=False, is_timeout=False, error_message=str(exc),
            ))

    # All steps failed
    if upstream_resp is None:
        total_ms = (time.monotonic() - t_start) * 1000
        log = RequestLog(
            timestamp=time.time(), model=original_model,
            selected_endpoint=None, attempts=attempts,
            status="failure", total_latency_ms=total_ms,
            is_stream=True, request_body=log_body,
        )
        await loop.run_in_executor(None, db.insert_request_log, log)
        events.publish()
        return anthropic_error("api_error", "All upstream endpoints failed", 502)

    # --- Wrap the upstream SSE stream with Anthropic translation ---
    final_attempts = attempts
    final_winning_step = winning_step

    async def wrapped_generator() -> AsyncIterator[str]:
        try:
            async for event_str in anthropic_sse_generator(upstream_resp, original_model):
                yield event_str
        finally:
            try:
                await upstream_resp.aclose()
            except Exception:
                pass
            total_ms = (time.monotonic() - t_start) * 1000
            log = RequestLog(
                timestamp=time.time(), model=original_model,
                selected_endpoint=final_winning_step.endpoint.name,
                attempts=final_attempts, status="success",
                total_latency_ms=total_ms, is_stream=True,
                request_body=log_body,
            )
            await loop.run_in_executor(None, db.insert_request_log, log)
            events.publish()

    return StreamingResponse(
        wrapped_generator(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},
    )
