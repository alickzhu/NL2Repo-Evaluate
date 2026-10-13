import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from litellm import ModelResponse
from openhands_research.model_profile import (
    GLM52_FP8_MODEL,
    GLM52_FP8_REVISION,
    MODEL_REVISIONS,
    QWEN38_FLASH_NEXT_MODEL,
    QWEN38_FLASH_NEXT_REVISION,
    ModelProfile,
)
from pydantic import SecretStr

from openhands.sdk import LLM
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.context.view import View
from openhands.sdk.event import ActionEvent, MessageEvent
from openhands.sdk.llm import Message, MessageToolCall, TextContent


def test_served_alias_preserves_qwen_profile(tmp_path):
    profile = ModelProfile("http://localhost/v1", api_model="qwen3.8-flash-next")
    for llm in (
        profile.build_agent_llm(tmp_path, "not-required"),
        profile.build_condenser(tmp_path, "not-required").llm,
    ):
        assert llm.model == "openai/qwen3.8-flash-next"
        assert llm.api_mode == "chat"
        assert llm.litellm_extra_body["chat_template_kwargs"]["preserve_thinking"]
        assert llm.capability_overrides["send_reasoning_content"]
        assert llm.num_retries == 8
        assert llm.retry_max_wait == 16
    assert profile.model == QWEN38_FLASH_NEXT_MODEL
    assert profile.public_dict()["api_model"] == "qwen3.8-flash-next"


@pytest.mark.parametrize("value", ["", "bad model", "bad\nmodel", 42])
def test_invalid_served_alias(value):
    with pytest.raises(ValueError, match="api_model"):
        ModelProfile("http://localhost/v1", api_model=value)


@pytest.mark.parametrize("model", [QWEN38_FLASH_NEXT_MODEL, GLM52_FP8_MODEL])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming, subscriber, usage", [
    (False, False, "solver"),
    (True, True, "solver"),
    (True, False, "solver"),
    (True, False, "condenser"),
])
@pytest.mark.parametrize("truncated_attempts", [0, 3])
@pytest.mark.asyncio
async def test_saved_input_matches_http_body_with_reasoning(
    tmp_path, model, asynchronous, streaming, subscriber, usage, truncated_attempts
):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            finish_reason = "length" if len(received) <= truncated_attempts else "stop"
            reasoning = (
                "DISCARDED_REASONING_CANARY"
                if finish_reason == "length"
                else "RESPONSE_REASONING_CANARY"
            )
            payload = {
                "id": f"http-response-{len(received)}",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": finish_reason,
                        "message": {
                            "role": "assistant",
                            "content": "OK",
                            "reasoning_content": reasoning,
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                    "total_tokens": 30,
                },
            }
            if body.get("stream"):
                payload["object"] = "chat.completion.chunk"
                choice = payload["choices"][0]
                choice["delta"] = choice.pop("message")
                choice["finish_reason"] = None
                first_chunk = "data: " + json.dumps(payload) + "\n\n"
                choice["delta"] = {}
                choice["finish_reason"] = finish_reason
                response = (
                    first_chunk
                    + "data: "
                    + json.dumps(payload)
                    + "\n\ndata: [DONE]\n\n"
                ).encode()
                content_type = "text/event-stream"
            else:
                response = json.dumps(payload).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format, *args):
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        profile = ModelProfile(
            f"http://127.0.0.1:{server.server_port}/v1",
            model=model,
            model_revision=MODEL_REVISIONS[model],
            reasoning_effort="high" if model == GLM52_FP8_MODEL else "xhigh",
        )
        llm = (profile.build_agent_llm(tmp_path, "not-needed") if usage == "solver"
               else profile.build_condenser(tmp_path, "not-needed").llm)
        assert llm.stream  # Do not hide profile/condenser validators with a copy.
        llm = llm.model_copy(update={"num_retries": 0})
        if not streaming:
            llm = llm.model_copy(update={"stream": False})
        messages = [
            Message(role="user", content=[TextContent(text="First task")]),
            Message(
                role="assistant",
                content=[TextContent(text="First answer")],
                reasoning_content="HTTP_HISTORY_REASONING_CANARY",
            ),
            Message(
                role="user",
                content=[TextContent(text="A later user or summary message")],
            ),
        ]
        chunks = []

        async def on_token(chunk):
            chunks.append(chunk)

        if asynchronous:
            response = await llm.acompletion(
                messages=messages, on_token=on_token if subscriber else None
            )
        else:
            response = llm.completion(
                messages=messages, on_token=chunks.append if subscriber else None
            )
        assert response.message.reasoning_content == "RESPONSE_REASONING_CANARY"
        assert len(received) == truncated_attempts + 1
        # Check the real HTTP body, including the production no-UI-callback path.
        assert all(bool(body.get("stream")) == streaming for body in received)
        assert [request["max_completion_tokens"] for request in received] == (
            [32768, 32768, 32768, 65536] if truncated_attempts else [32768]
        )
        assert [request["seed"] for request in received] == list(
            range(42, 43 + truncated_attempts)
        )
        assert all(
            request["messages"] == received[0]["messages"] for request in received
        )
        assert received[0]["messages"][1]["reasoning_content"] == (
            "HTTP_HISTORY_REASONING_CANARY"
        )
        template_kwargs = received[0]["chat_template_kwargs"]
        if model == GLM52_FP8_MODEL:
            assert template_kwargs["clear_thinking"] is False
            assert template_kwargs["reasoning_effort"] == "high"
            assert "preserve_thinking" not in template_kwargs
        else:
            assert template_kwargs["preserve_thinking"] is True
        logs = list(tmp_path.glob("*.json"))
        assert len(logs) == len(received)
        assert all(
            json.loads(path.read_text())["messages"] == received[0]["messages"]
            for path in logs
        )
        if subscriber:
            assert chunks
            assert all(chunk.id == response.id for chunk in chunks)
            assert "DISCARDED_REASONING_CANARY" not in json.dumps(
                [chunk.model_dump(mode="json") for chunk in chunks]
            )
        else:
            assert not chunks
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("model", [QWEN38_FLASH_NEXT_MODEL, GLM52_FP8_MODEL])
@pytest.mark.asyncio
async def test_solver_and_condenser_preserve_all_history_reasoning(tmp_path, model):
    profile = ModelProfile(
        "http://127.0.0.1:33202/v1",
        model=model,
        model_revision=MODEL_REVISIONS[model],
        reasoning_effort="high" if model == GLM52_FP8_MODEL else "xhigh",
    )
    messages = [
        Message(role="user", content=[TextContent(text="First task")]),
        Message(
            role="assistant",
            content=[TextContent(text="First answer")],
            reasoning_content="FIRST_REASONING_CANARY",
        ),
        Message(role="user", content=[TextContent(text="Continue")]),
        Message(
            role="assistant",
            content=[TextContent(text="Second answer")],
            reasoning_content="SECOND_REASONING_CANARY",
        ),
        Message(role="user", content=[TextContent(text="Continue again")]),
    ]
    for llm in (
        profile.build_agent_llm(tmp_path, "not-needed"),
        profile.build_condenser(tmp_path, "not-needed").llm,
    ):
        formatted = llm.format_messages_for_llm(messages)
        assert formatted == await llm.aformat_messages_for_llm(messages)
        assert [
            message["reasoning_content"]
            for message in formatted
            if message["role"] == "assistant"
        ] == ["FIRST_REASONING_CANARY", "SECOND_REASONING_CANARY"]


@pytest.mark.asyncio
async def test_summarizer_receives_full_reasoning_and_tool_arguments(
    tmp_path, monkeypatch
):
    captured = []

    async def complete(llm, *, messages, **kwargs):
        del llm, kwargs
        captured.extend(messages)
        return ModelResponse(
            id="summary-response",
            choices=[{"message": {"role": "assistant", "content": "Summary"}}],
        )

    monkeypatch.setattr(LLM, "_atransport_call", complete)
    condenser = ModelProfile("http://127.0.0.1:33202/v1").build_condenser(
        tmp_path, "not-needed"
    )
    event = ActionEvent(
        source="agent",
        thought=[],
        tool_name="terminal",
        tool_call_id="call-1",
        tool_call=MessageToolCall(
            id="call-1",
            name="terminal",
            arguments='{"command":"FULL_TOOL_ARGUMENT_CANARY"}',
            origin="completion",
        ),
        action=None,
        reasoning_content="FULL_REASONING_CANARY",
        llm_response_id="original-response",
    )
    result = await condenser._agenerate_condensation([event], 0)
    assert result.summary == "Summary"
    prompt = json.dumps(captured)
    assert "FULL_REASONING_CANARY" in prompt
    assert "FULL_TOOL_ARGUMENT_CANARY" in prompt
    assert "<think>" in prompt and "</think>" in prompt


def test_qwen_profile_builds_separate_solver_and_condenser(tmp_path):
    profile = ModelProfile(endpoint="http://127.0.0.1:33580/v1")
    assert profile.condenser_enabled is True
    solver = profile.build_agent_llm(tmp_path / "completions", "not-needed")
    condenser = profile.build_condenser(tmp_path / "completions", "not-needed")
    assert profile.model == QWEN38_FLASH_NEXT_MODEL
    assert profile.model_revision == QWEN38_FLASH_NEXT_REVISION
    assert solver.model == "openai/" + QWEN38_FLASH_NEXT_MODEL
    assert solver.usage_id == "solver" and solver.stream
    assert solver.max_input_tokens == 262144
    assert solver.max_output_tokens == 32768
    assert solver.api_mode == "chat" and solver.native_tool_calling
    assert solver.reasoning_effort is None
    assert solver.litellm_extra_body == {
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
            "reasoning_effort": "xhigh",
        }
    }
    assert condenser.llm.usage_id == "condenser"
    assert condenser.llm.stream
    assert condenser.llm.max_output_tokens == 32768
    assert condenser.max_size == 80 and condenser.keep_first == 4
    assert not condenser.trigger_on_events
    assert condenser.max_tokens == 196608
    assert profile.public_dict()["endpoint"] == "http://127.0.0.1:33580/v1"


def test_disabled_condenser_does_not_build_llm(tmp_path, monkeypatch):
    profile = ModelProfile(endpoint="http://127.0.0.1:33580/v1", condenser_enabled=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled condenser must not construct a summary LLM")

    monkeypatch.setattr(ModelProfile, "_llm", unexpected)
    assert profile.build_condenser(tmp_path, "not-needed") is None
    assert profile.public_dict()["condenser_enabled"] is False


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_condenser_toggle_requires_boolean(value):
    with pytest.raises(ValueError, match="condenser_enabled"):
        ModelProfile(endpoint="http://127.0.0.1:33580/v1", condenser_enabled=value)


@pytest.mark.parametrize("effort", ["high", "max"])
def test_glm_profile_builds_native_solver_and_condenser(tmp_path, effort):
    profile = ModelProfile(
        endpoint="http://127.0.0.1:33581/v1",
        model=GLM52_FP8_MODEL,
        model_revision=GLM52_FP8_REVISION,
        reasoning_effort=effort,
    )
    for llm in (
        profile.build_agent_llm(tmp_path, "not-needed"),
        profile.build_condenser(tmp_path, "not-needed").llm,
    ):
        assert llm.model == "openai/" + GLM52_FP8_MODEL
        assert llm.model_canonical_name == "glm-5.2"
        assert llm.max_input_tokens == 262144
        assert llm.max_output_tokens == 32768
        assert llm.litellm_extra_body == {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
                "reasoning_effort": effort,
            }
        }
    assert profile.public_dict()["model_revision"] == GLM52_FP8_REVISION


@pytest.mark.parametrize(
    "kwargs",
    [
        {"endpoint": "127.0.0.1:33580/v1"},
        {"endpoint": "http://127.0.0.1:33580"},
        {"endpoint": "http://127.0.0.1:33580/v1", "model": "other"},
        {
            "endpoint": "http://127.0.0.1:33580/v1",
            "model_revision": "mutable-main",
        },
        {
            "endpoint": "http://127.0.0.1:33580/v1",
            "model": GLM52_FP8_MODEL,
            "model_revision": QWEN38_FLASH_NEXT_REVISION,
        },
        {
            "endpoint": "http://127.0.0.1:33580/v1",
            "model": GLM52_FP8_MODEL,
            "model_revision": GLM52_FP8_REVISION,
            "reasoning_effort": "High",
        },
        {"endpoint": "http://127.0.0.1:33580/v1", "max_input_tokens": 131072},
    ],
)
def test_invalid_profile_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ModelProfile(**kwargs)


def test_profile_never_exports_api_key(tmp_path):
    profile = ModelProfile(endpoint="http://127.0.0.1:33580/v1")
    llm = profile.build_agent_llm(Path(tmp_path), "PRIVATE")
    assert "PRIVATE" not in str(profile.public_dict())
    assert isinstance(llm.api_key, SecretStr)
    assert llm.api_key.get_secret_value() == "PRIVATE"


@pytest.mark.parametrize("event_limit", [0, 80])
def test_event_trigger_is_optional_but_explicit_requests_still_work(
    tmp_path, event_limit
):
    condenser = ModelProfile(
        "http://127.0.0.1:33202/v1", condenser_max_events=event_limit
    ).build_condenser(tmp_path, "not-needed")
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user", content=[TextContent(text="Small message")]
                ),
            )
            for _index in range(81)
        ]
    )
    assert condenser.get_condensation_reasons(view) == (
        {Reason.EVENTS} if event_limit else set()
    )
    view.unhandled_condensation_request = True
    assert Reason.REQUEST in condenser.get_condensation_reasons(view)
