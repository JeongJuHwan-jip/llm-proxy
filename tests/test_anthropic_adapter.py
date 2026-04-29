"""Unit tests for the Anthropic Messages API adapter.

Tests the pure translation functions (request, response, streaming SSE)
without any HTTP or app startup.
"""

from __future__ import annotations

import json

import pytest

from llm_proxy.adapters.anthropic import (
    SSEBuffer,
    _parse_sse_data,
    translate_request,
    translate_response,
)


# ---------------------------------------------------------------------------
# translate_request
# ---------------------------------------------------------------------------


class TestTranslateRequest:

    def test_basic_text_message(self):
        body = {
            "model": "test-route",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = translate_request(body)
        assert result["model"] == "test-route"
        assert result["max_tokens"] == 100
        assert result["messages"] == [{"role": "user", "content": "hello"}]

    def test_system_string_prepended(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_array_concatenated(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "system": [
                {"type": "text", "text": "Line one."},
                {"type": "text", "text": "Line two."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert result["messages"][0]["role"] == "system"
        assert "Line one." in result["messages"][0]["content"]
        assert "Line two." in result["messages"][0]["content"]

    def test_stop_sequences_renamed(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "stop_sequences": ["\n\nHuman:"],
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert result["stop"] == ["\n\nHuman:"]
        assert "stop_sequences" not in result

    def test_passthrough_fields(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9
        assert result["stream"] is True

    def test_top_k_dropped(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "top_k": 40,
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert "top_k" not in result

    def test_metadata_dropped(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "metadata": {"user_id": "abc"},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert "metadata" not in result

    def test_thinking_passthrough(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = translate_request(body)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    def test_tool_definitions_translated(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
        result = translate_request(body)
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_tool_choice_auto(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any_maps_to_required(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == "required"

    def test_tool_choice_specific_tool(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_use_in_assistant_message(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "get_weather",
                            "input": {"city": "Seoul"},
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_123"
        assert tc["function"]["name"] == "get_weather"
        assert json.loads(tc["function"]["arguments"]) == {"city": "Seoul"}

    def test_tool_result_expands_to_tool_messages(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "Sunny, 25C",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_456",
                            "content": "Rainy, 15C",
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        # Should expand to two tool messages
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["tool_call_id"] == "toolu_123"
        assert result["messages"][0]["content"] == "Sunny, 25C"
        assert result["messages"][1]["role"] == "tool"
        assert result["messages"][1]["tool_call_id"] == "toolu_456"

    def test_tool_result_with_content_blocks(self):
        """tool_result content can be an array of content blocks."""
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": [{"type": "text", "text": "Result text"}],
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        assert result["messages"][0]["content"] == "Result text"

    def test_mixed_user_content_with_tool_result(self):
        """User message with both text and tool_result blocks."""
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are the results:"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_123",
                            "content": "OK",
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        # First: user message with text, second: tool message
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Here are the results:"
        assert result["messages"][1]["role"] == "tool"

    def test_image_base64_translated(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBOR...",
                            },
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

    def test_image_url_passthrough(self):
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/img.png",
                            },
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["image_url"]["url"] == "https://example.com/img.png"

    def test_assistant_tool_use_only(self):
        """Assistant message with only tool_use blocks → content is None."""
        body = {
            "model": "m",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "fn",
                            "input": {},
                        },
                    ],
                }
            ],
        }
        result = translate_request(body)
        msg = result["messages"][0]
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# translate_response
# ---------------------------------------------------------------------------


class TestTranslateResponse:

    def test_text_content_wrapped_in_block(self):
        oai = {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = translate_response(oai, "claude-3")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello!"}]

    def test_tool_calls_become_tool_use_blocks(self):
        oai = {
            "id": "chatcmpl-abc",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Seoul"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        result = translate_response(oai, "claude-3")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        block = result["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "call_123"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Seoul"}

    def test_mixed_text_and_tool_calls(self):
        oai = {
            "id": "chatcmpl-abc",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Let me check.",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "fn", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        result = translate_response(oai, "m")
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_finish_reason_stop_maps_to_end_turn(self):
        oai = {
            "id": "x",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        assert translate_response(oai, "m")["stop_reason"] == "end_turn"

    def test_finish_reason_length_maps_to_max_tokens(self):
        oai = {
            "id": "x",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        assert translate_response(oai, "m")["stop_reason"] == "max_tokens"

    def test_finish_reason_tool_calls_maps_to_tool_use(self):
        oai = {
            "id": "x",
            "choices": [{"message": {"content": None, "tool_calls": []}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        assert translate_response(oai, "m")["stop_reason"] == "tool_use"

    def test_usage_fields_renamed(self):
        oai = {
            "id": "x",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }
        result = translate_response(oai, "m")
        assert result["usage"]["input_tokens"] == 100
        assert result["usage"]["output_tokens"] == 50

    def test_id_gets_msg_prefix(self):
        oai = {
            "id": "chatcmpl-abc123",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = translate_response(oai, "m")
        assert result["id"] == "msg_chatcmpl-abc123"

    def test_model_uses_original(self):
        oai = {
            "id": "x",
            "model": "gpt-4",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        result = translate_response(oai, "claude-3-sonnet")
        assert result["model"] == "claude-3-sonnet"

    def test_empty_choices(self):
        oai = {"id": "x", "choices": [], "usage": {}}
        result = translate_response(oai, "m")
        assert result["content"] == []
        assert result["stop_reason"] == "end_turn"


# ---------------------------------------------------------------------------
# SSE parsing helpers
# ---------------------------------------------------------------------------


class TestSSEBuffer:

    def test_single_complete_event(self):
        buf = SSEBuffer()
        events = buf.feed(b'data: {"test": 1}\n\n')
        assert len(events) == 1
        assert "test" in events[0]

    def test_split_across_chunks(self):
        buf = SSEBuffer()
        assert buf.feed(b'data: {"te') == []
        assert buf.feed(b'st": 1}\n') == []
        events = buf.feed(b"\n")
        assert len(events) == 1

    def test_multiple_events_in_one_chunk(self):
        buf = SSEBuffer()
        events = buf.feed(b'data: {"a": 1}\n\ndata: {"b": 2}\n\n')
        assert len(events) == 2

    def test_done_event(self):
        buf = SSEBuffer()
        events = buf.feed(b"data: [DONE]\n\n")
        assert len(events) == 1
        parsed = _parse_sse_data(events[0])
        assert parsed is None  # [DONE] returns None


class TestParseSSEData:

    def test_normal_json(self):
        result = _parse_sse_data('data: {"choices": []}')
        assert result == {"choices": []}

    def test_done(self):
        assert _parse_sse_data("data: [DONE]") is None

    def test_event_with_named_type(self):
        """SSE events may have 'event:' lines before 'data:'."""
        result = _parse_sse_data("event: message\ndata: {\"id\": 1}")
        assert result == {"id": 1}

    def test_invalid_json(self):
        assert _parse_sse_data("data: not-json") is None

    def test_no_data_line(self):
        assert _parse_sse_data("event: ping") is None
