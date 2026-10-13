"""Fail closed on incomplete SGLang SSE before committing assistant/tool output."""

from __future__ import annotations

import inspect
import uuid
from typing import Any

from tenacity import wait_random_exponential

from ._checked_sse_client import async_client, client
from .length_retry import LengthRetryLLM

GUARDED_MODELS = frozenset(
    {
        "autonl2repo-qwen3.5-35b-a3b-base-v21-step288",
        "autonl2repo-qwen3.5-35b-a3b-base-v22-step400",
        "autonl2repo-qwen3.5-35b-a3b-base-v22-step650",
        "autonl2repo-qwen3.5-35b-a3b-base-v22-step1350",
        "autonl2repo-qwen3.5-35b-a3b-v17-step288",
    }
)


def identity_kwargs(options: dict[str, Any]) -> dict[str, Any]:
    options = dict(options)
    headers = dict(options.get("extra_headers") or {})
    headers.setdefault("X-Request-ID", "generation-" + uuid.uuid4().hex)
    options["extra_headers"] = headers
    return options


class SFTTransportLLM(LengthRetryLLM):
    """Retain length handling; validate raw endings and isolate retry callbacks.

    The SDK owns bounded transport retries. Each transport attempt receives its
    own private callback buffer, so rejected output never escapes into the agent
    or the condenser. This does not alter scores, prompts, sampling, or EOS.
    """

    def completion(self, *args: Any, **kwargs: Any):
        return super().completion(*args, **identity_kwargs(kwargs))

    def _make_retry_decorator(self):
        parent = super()._make_retry_decorator()

        def decorate(function):
            # Keep the SDK's error predicate, attempt limit, and retry logging.
            # Spread concurrent client retries without adding another retry layer.
            wrapped = parent(function)
            return wrapped.retry_with(
                wait=wait_random_exponential(
                    multiplier=self.retry_multiplier,
                    min=self.retry_min_wait,
                    max=self.retry_max_wait,
                )
            )

        return decorate

    async def acompletion(self, *args: Any, **kwargs: Any):
        return await super().acompletion(*args, **identity_kwargs(kwargs))

    def _transport_call(
        self, *, messages, enable_streaming=False, on_token=None, **kwargs
    ):
        if not enable_streaming:
            return super()._transport_call(
                messages=messages, enable_streaming=False, on_token=on_token, **kwargs
            )
        if kwargs.get("client") is not None:
            raise ValueError("SFTTransportLLM manages its own checked HTTP client")
        chunks = []
        assert self.base_url is not None
        with client(self.base_url) as checked:
            result = super()._transport_call(
                messages=messages,
                enable_streaming=True,
                on_token=chunks.append,
                **{**kwargs, "client": checked},
            )
        if on_token is not None:
            for chunk in chunks:
                on_token(chunk)
        return result

    async def _atransport_call(
        self, *, messages, enable_streaming=False, on_token=None, **kwargs
    ):
        if not enable_streaming:
            return await super()._atransport_call(
                messages=messages, enable_streaming=False, on_token=on_token, **kwargs
            )
        if kwargs.get("client") is not None:
            raise ValueError("SFTTransportLLM manages its own checked HTTP client")
        chunks = []
        assert self.base_url is not None
        async with async_client(self.base_url) as checked:
            result = await super()._atransport_call(
                messages=messages,
                enable_streaming=True,
                on_token=chunks.append,
                **{**kwargs, "client": checked},
            )
        if on_token is not None:
            for chunk in chunks:
                pending = on_token(chunk)
                if inspect.isawaitable(pending):
                    await pending
        return result
