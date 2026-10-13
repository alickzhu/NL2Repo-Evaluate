import asyncio
import copy
import json

import pytest
from litellm import ModelResponse
from openhands_research.length_retry import LengthRetryLLM, OutputLengthLimitError
from openhands_research.model_profile import ModelProfile

from openhands.sdk import LLM
from openhands.sdk.context.view import View
from openhands.sdk.event import MessageEvent
from openhands.sdk.llm import Message, TextContent


@pytest.fixture
def provider(monkeypatch):
    requests = []
    responses = []

    def complete(llm, *, messages, **kwargs):
        del llm
        requests.append({"messages": copy.deepcopy(messages), **kwargs})
        finish = responses.pop(0) if responses else "length"
        return ModelResponse(
            id=f"response-{len(requests)}",
            choices=[
                {
                    "index": 0,
                    "finish_reason": finish,
                    "message": {
                        "role": "assistant",
                        "content": "Complete" if finish == "stop" else "",
                        "reasoning_content": f"REASONING_{len(requests)}",
                    },
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        )

    async def acomplete(llm, *, messages, **kwargs):
        return complete(llm, messages=messages, **kwargs)

    monkeypatch.setattr(LLM, "_transport_call", complete)
    monkeypatch.setattr(LLM, "_atransport_call", acomplete)
    return requests, responses


@pytest.fixture
def llm(tmp_path):
    return LengthRetryLLM(
        model="openai/Qwen/Qwen3.8-Flash-Next-FP8",
        api_key="not-needed",
        base_url="http://127.0.0.1:1/v1",
        api_mode="chat",
        capability_overrides={"send_reasoning_content": True},
        max_input_tokens=262144,
        max_output_tokens=32768,
        seed=42,
        num_retries=0,
        log_completions=True,
        log_completions_folder=str(tmp_path / "completions"),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_three_attempts_then_double_preserves_history_and_logs(
    llm, provider, tmp_path, asynchronous
):
    requests, responses = provider
    responses.extend(["length", "length", "length", "stop", "stop"])
    messages = [
        Message(role="user", content=[TextContent(text="Task")]),
        Message(
            role="assistant",
            content=[TextContent(text="Previous answer")],
            reasoning_content="PREVIOUS_REASONING",
        ),
        Message(role="user", content=[TextContent(text="Continue")]),
    ]
    original = copy.deepcopy(messages)
    if asynchronous:
        result = await llm.acompletion(messages)
        await llm.acompletion(messages)
    else:
        result = llm.completion(messages)
        llm.completion(messages)
    assert result.id == "response-4"
    assert messages == original
    assert [request["max_completion_tokens"] for request in requests] == [
        32768,
        32768,
        32768,
        65536,
        32768,
    ]
    assert [request["seed"] for request in requests] == [42, 43, 44, 45, 42]
    assert all(request["messages"] == requests[0]["messages"] for request in requests)
    assert requests[0]["messages"][1]["reasoning_content"] == "PREVIOUS_REASONING"
    assert len(list((tmp_path / "completions").glob("*.json"))) == 5
    decisions = [
        json.loads(line)["decision"]
        for line in (tmp_path / "output_retries.jsonl").read_text().splitlines()
    ]
    assert decisions == ["retry", "retry", "escalate", "accepted", "accepted"]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_terminal_length_failure_is_exactly_six_attempts(
    llm, provider, asynchronous
):
    requests, _responses = provider
    messages = [Message(role="user", content=[TextContent(text="Task")])]
    with pytest.raises(OutputLengthLimitError, match="after 6 attempts"):
        if asynchronous:
            await llm.acompletion(messages)
        else:
            llm.completion(messages)
    assert [request["max_completion_tokens"] for request in requests] == [
        32768,
        32768,
        32768,
        65536,
        65536,
        65536,
    ]


def test_escalation_respects_context_headroom(llm, provider, monkeypatch):
    requests, responses = provider
    monkeypatch.setattr(LLM, "get_token_count", lambda *_args, **_kwargs: 221632)
    responses.extend(["length", "length", "length", "stop"])
    llm.completion([Message(role="user", content=[TextContent(text="Task")])])
    assert requests[-1]["max_completion_tokens"] == 40000


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_fixed_cap_retries_past_old_limit_without_history_mutation(
    llm, provider, tmp_path, asynchronous
):
    requests, responses = provider
    llm = llm.model_copy(update={"length_retry_mode": "fixed_until_complete", "max_tokens_ceiling": 32768})
    responses.extend(["length"] * 10 + ["stop"])
    messages = [Message(role="user", content=[TextContent(text="Task")])]
    original = copy.deepcopy(messages)
    result = await llm.acompletion(messages) if asynchronous else llm.completion(messages)
    assert result.id == "response-11"
    assert len(requests) == 11 and messages == original
    assert {request["max_completion_tokens"] for request in requests} == {32768}
    assert [request["seed"] for request in requests] == list(range(42, 53))
    assert all(r["messages"] == requests[0]["messages"] for r in requests)
    decisions = [json.loads(line)["decision"] for line in (tmp_path / "output_retries.jsonl").read_text().splitlines()]
    assert decisions == ["retry"] * 10 + ["accepted"]


def test_fixed_retry_profile_reaches_solver_and_summary(tmp_path):
    profile = ModelProfile("http://localhost/v1", length_retry_mode="fixed_until_complete", max_tokens_ceiling=32768)
    for llm in (profile.build_agent_llm(tmp_path, "not-needed"), profile.build_condenser(tmp_path, "not-needed").llm):
        assert llm.length_retry_mode == "fixed_until_complete"
        assert llm.max_tokens_ceiling == llm.max_output_tokens == 32768
    assert ModelProfile("http://localhost/v1").length_retry_mode == "bounded"


def test_custom_initial_budget_ratio_and_ceiling(llm, provider):
    requests, responses = provider
    llm = llm.model_copy(
        update={
            "length_retries": 1,
            "length_growth_factor": 1.5,
            "max_tokens_ceiling": 50000,
        }
    )
    responses.extend(["length", "length", "stop"])
    llm.completion(
        [Message(role="user", content=[TextContent(text="Task")])],
        max_completion_tokens=24000,
    )
    assert [request["max_completion_tokens"] for request in requests] == [
        24000,
        36000,
        50000,
    ]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", ["explicit", "events", "hard_reset"])
@pytest.mark.asyncio
async def test_summary_exhaustion_cannot_restart_its_retry_quota(
    provider, tmp_path, asynchronous, mode
):
    requests, _responses = provider
    profile = ModelProfile(
        "http://127.0.0.1:1/v1", condenser_max_events=20, request_retries=0
    )
    condenser = profile.build_condenser(tmp_path / "completions", "not-needed")
    assert isinstance(condenser.llm, LengthRetryLLM)
    view = View(
        events=[
            MessageEvent(
                source="user",
                llm_message=Message(
                    role="user", content=[TextContent(text=f"Public event {index}")]
                ),
            )
            for index in range(30)
        ],
        unhandled_condensation_request=mode == "explicit",
    )
    with pytest.raises(OutputLengthLimitError, match="after 6 attempts"):
        if mode == "hard_reset":
            if asynchronous:
                await condenser.ahard_context_reset(view)
            else:
                condenser.hard_context_reset(view)
        elif asynchronous:
            await condenser.acondense(view)
        else:
            condenser.condense(view)
    assert len(requests) == 6


@pytest.mark.parametrize(
    "overrides",
    [
        {"length_retries": 0},
        {"length_retry_mode": "unknown"},
        {"length_growth_factor": 1},
        {"length_growth_factor": float("inf")},
        {"max_tokens_ceiling": 16000},
        {"context_safety_tokens": -1},
    ],
)
def test_invalid_output_retry_policy_fails_before_inference(overrides):
    with pytest.raises(ValueError):
        ModelProfile("http://127.0.0.1:1/v1", **overrides)


@pytest.mark.parametrize("prompt_tokens", [0, 261632, 262144])
def test_missing_token_count_or_context_headroom_stops_before_request(
    llm, provider, monkeypatch, prompt_tokens
):
    requests, _responses = provider
    monkeypatch.setattr(LLM, "get_token_count", lambda *_args, **_kwargs: prompt_tokens)
    with pytest.raises(OutputLengthLimitError):
        llm.completion([Message(role="user", content=[TextContent(text="Task")])])
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["bounded", "fixed_until_complete"])
async def test_cancellation_stops_retry_loop(llm, monkeypatch, mode):
    llm = llm.model_copy(update={"length_retry_mode": mode})
    started = asyncio.Event()
    blocked = asyncio.Event()
    requests = []

    async def transport(_llm, **kwargs):
        requests.append(kwargs)
        started.set()
        await blocked.wait()
        raise AssertionError("cancelled transport must not complete")

    monkeypatch.setattr(LLM, "_atransport_call", transport)
    task = asyncio.create_task(
        llm.acompletion([Message(role="user", content=[TextContent(text="Task")])])
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(requests) == 1
