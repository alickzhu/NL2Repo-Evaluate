"""Exercise the SDK retry loop without API calls or real backoff sleeps."""

import pytest
from litellm import RateLimitError

from openhands_research.model_profile import (
    GLM52_FP8_MODEL,
    GLM52_FP8_REVISION,
    ModelProfile,
)


@pytest.mark.parametrize("model_kind", ["qwen", "glm", "luna"])
@pytest.mark.parametrize("usage", ["solver", "condenser"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("succeed_on_last", [False, True])
@pytest.mark.asyncio
async def test_eight_api_attempts_with_capped_waits(
    tmp_path, model_kind, usage, asynchronous, succeed_on_last
):
    options = {}
    if model_kind == "glm":
        options = dict(
            model=GLM52_FP8_MODEL,
            model_revision=GLM52_FP8_REVISION,
            reasoning_effort="high",
        )
    elif model_kind == "luna":
        options = dict(
            model="gpt-5.6-luna",
            model_revision="endpoint-defined",
            api_mode="responses",
            temperature=None,
            seed=None,
            reasoning_effort="high",
        )
    profile = ModelProfile("http://localhost:1/v1", **options)
    llm = (
        profile.build_agent_llm(tmp_path, "test-key")
        if usage == "solver"
        else profile.build_condenser(tmp_path, "test-key").llm
    )
    assert llm.num_retries == 8
    assert llm.retry_min_wait == 8
    assert llm.retry_max_wait == 16
    assert llm.retry_multiplier == 8.0
    attempts = 0
    waits = []

    def request():
        nonlocal attempts
        attempts += 1
        if succeed_on_last and attempts == 8:
            return "ok"
        raise RateLimitError(
            message="too many requests", model=llm.model, llm_provider="openai"
        )

    async def async_request():
        return request()

    async def async_sleep(seconds):
        waits.append(seconds)

    wrapped = llm._make_retry_decorator()(async_request if asynchronous else request)
    # Exercise Tenacity's real stop/wait policy, replacing only the sleeping.
    wrapped.retry.sleep = async_sleep if asynchronous else waits.append

    async def invoke():
        return await wrapped() if asynchronous else wrapped()

    if succeed_on_last:
        assert await invoke() == "ok"
    else:
        with pytest.raises(RateLimitError, match="too many requests"):
            await invoke()
    assert attempts == 8  # Eight total attempts, not eight retries plus one.
    assert waits == [8, 16, 16, 16, 16, 16, 16]
