"""Regenerate length-limited completions before event publication."""

from __future__ import annotations

import inspect
import math
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from litellm.types.llms.openai import ResponseIncompleteEvent
from litellm.types.utils import ModelResponseStream
from openhands.sdk.llm import LLM, LLMResponse, Message
from openhands.sdk.llm.llm import LLMCallContext
from openhands.sdk.llm.streaming import AnyTokenCallbackType, TokenCallbackType
from openhands.sdk.tool import ToolDefinition
from pydantic import Field

from .ledger import append_jsonl


class OutputLengthLimitError(RuntimeError):
    """No complete response was obtained within the configured output budgets."""


def _drain_token(_chunk):
    """Keep the SDK transport streaming without a UI token subscriber."""


@dataclass(frozen=True)
class _Attempt:
    number: int
    at_budget: int
    max_tokens: int
    next_max_tokens: int | None


class LengthRetryLLM(LLM):
    """Keep failed generations private; return only a non-truncated response."""

    length_retries: int = Field(default=3, ge=1)
    length_retry_mode: Literal["bounded", "fixed_until_complete"] = "bounded"
    max_tokens_ceiling: int = Field(default=65_536, gt=0)
    length_growth_factor: float = Field(default=2.0, gt=1, allow_inf_nan=False)
    context_safety_tokens: int = Field(default=512, ge=0)

    def _attempt_callback(self, on_token, chunks, options):
        # SDK disables streaming when on_token=None. LocalConversation and the
        # condenser normally have no UI callbacks, but still need SSE transport.
        # SDK assembles/logs the complete response independently of this sink.
        # Only buffer for real subscribers; never publish a rejected length
        # attempt (or keep a second copy of its chunks unnecessarily).
        if on_token is not None:
            return chunks.append
        if self.stream or options.get("stream", False):
            return _drain_token
        return None

    def _finalize_stream_response(self, completed_response, collected_output_items):
        # Preserve real incomplete streams for our bounded length retry policy.
        # The pinned SDK otherwise treats response.incomplete as transport error.
        if isinstance(completed_response, ResponseIncompleteEvent):
            response = completed_response.response
            if not response.output and collected_output_items:
                response.output = collected_output_items
            assert self._telemetry is not None
            self._telemetry.on_response(response, provider_info=self._provider_info)
            return response
        return super()._finalize_stream_response(
            completed_response, collected_output_items
        )

    def _length_attempts(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None,
        security: bool,
        options: dict[str, Any],
    ) -> Iterator[tuple[_Attempt, dict[str, Any]]]:
        options = dict(options)
        initial = options.pop("max_completion_tokens", None)
        legacy = options.pop("max_tokens", None)
        if initial is not None and legacy is not None and initial != legacy:
            raise ValueError("conflicting output token budgets")
        initial = initial if initial is not None else legacy
        initial = initial if initial is not None else self.effective_max_output_tokens
        if type(initial) is not int or not 0 < initial <= self.max_tokens_ceiling:
            raise ValueError("initial output budget must be positive and <= ceiling")
        context = self.effective_max_input_tokens
        if context is None:
            raise ValueError("length retries require an explicit context limit")
        prompt_tokens = self.get_token_count(
            messages, tools=tools, add_security_risk_prediction=security
        )
        if prompt_tokens <= 0:
            raise OutputLengthLimitError("Unable to establish input token count")
        ceiling = min(
            self.max_tokens_ceiling,
            context - prompt_tokens - self.context_safety_tokens,
        )
        if ceiling <= 0:
            raise OutputLengthLimitError("No output headroom remains in the context")
        seed = options.pop("seed", self.seed)
        if seed is not None and type(seed) is not int:
            raise ValueError("seed must be an integer or None")
        budget = min(initial, ceiling)
        number = 0
        while True:
            if self.length_retry_mode == "fixed_until_complete":
                number += 1
                yield (
                    _Attempt(number, number, budget, budget),
                    {**options, "max_completion_tokens": budget,
                     "seed": seed + number - 1 if seed is not None else None},
                )
                # No length backoff, escalation or attempt limit. Cancellation
                # and the owning task's wall-clock deadline still apply.
                continue
            grown = min(math.ceil(budget * self.length_growth_factor), ceiling)
            for at_budget in range(1, self.length_retries + 1):
                number += 1
                next_budget = (
                    budget
                    if at_budget < self.length_retries
                    else grown
                    if grown > budget
                    else None
                )
                attempt = _Attempt(number, at_budget, budget, next_budget)
                yield (
                    attempt,
                    {
                        **options,
                        "max_completion_tokens": budget,
                        "seed": seed + number - 1 if seed is not None else None,
                    },
                )
            if grown <= budget:
                raise OutputLengthLimitError(
                    f"{self.usage_id}: output-length retries exhausted after "
                    f"{number} attempts; final cap={budget}, context headroom={ceiling}"
                )
            budget = grown

    def _accept_length_attempt(
        self, response: LLMResponse, attempt: _Attempt, generation_id: str
    ) -> bool:
        raw = response.raw_response.model_dump(mode="json")
        if "choices" in raw:
            finish_reason = raw["choices"][0]["finish_reason"]
        elif raw.get("status") == "completed":
            finish_reason = "stop"
        elif (
            raw.get("status") == "incomplete"
            and (raw.get("incomplete_details") or {}).get("reason")
            == "max_output_tokens"
        ):
            finish_reason = "length"
        else:
            raise RuntimeError(
                f"Responses request did not complete: {raw.get('status')}"
            )
        accepted = finish_reason != "length"
        if accepted:
            decision = "accepted"
        elif attempt.next_max_tokens is None:
            decision = "exhausted"
        elif attempt.next_max_tokens > attempt.max_tokens:
            decision = "escalate"
        else:
            decision = "retry"
        if self.log_completions:
            append_jsonl(
                Path(self.log_completions_folder).parent / "output_retries.jsonl",
                {
                    "generation_id": generation_id,
                    "usage_id": self.usage_id,
                    "response_id": response.id,
                    "attempt": attempt.number,
                    "attempt_at_budget": attempt.at_budget,
                    "max_tokens": attempt.max_tokens,
                    "finish_reason": finish_reason,
                    "decision": decision,
                    "next_max_tokens": None if accepted else attempt.next_max_tokens,
                },
            )
        return accepted

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: TokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        generation_id = uuid.uuid4().hex
        for attempt, options in self._length_attempts(
            messages, tools, add_security_risk_prediction, kwargs
        ):
            chunks: list[ModelResponseStream] = []
            response = super().completion(
                messages,
                tools,
                add_security_risk_prediction,
                self._attempt_callback(on_token, chunks, options),
                call_context,
                **options,
            )
            if self._accept_length_attempt(response, attempt, generation_id):
                if on_token is not None:
                    for chunk in chunks:
                        on_token(chunk)
                return response
        raise AssertionError("length-attempt iterator must return or raise")

    def _responses_attempts(self, messages, tools, security, options):
        options = dict(options)
        if "max_output_tokens" in options:
            options["max_completion_tokens"] = options.pop("max_output_tokens")
        for attempt, values in self._length_attempts(
            messages, tools, security, options
        ):
            values["max_output_tokens"] = values.pop("max_completion_tokens")
            values.pop("seed", None)
            yield attempt, values

    def responses(
        self,
        messages,
        tools=None,
        include=None,
        store=None,
        add_security_risk_prediction=False,
        on_token=None,
        call_context=None,
        **kwargs,
    ):
        generation_id = uuid.uuid4().hex
        for attempt, options in self._responses_attempts(
            messages, tools, add_security_risk_prediction, kwargs
        ):
            chunks = []
            response = super().responses(
                messages,
                tools,
                include,
                store,
                add_security_risk_prediction,
                self._attempt_callback(on_token, chunks, options),
                call_context,
                **options,
            )
            if self._accept_length_attempt(response, attempt, generation_id):
                if on_token is not None:
                    for chunk in chunks:
                        on_token(chunk)
                return response
        raise AssertionError("length-attempt iterator must return or raise")

    async def aresponses(
        self,
        messages,
        tools=None,
        include=None,
        store=None,
        add_security_risk_prediction=False,
        on_token=None,
        call_context=None,
        **kwargs,
    ):
        generation_id = uuid.uuid4().hex
        for attempt, options in self._responses_attempts(
            messages, tools, add_security_risk_prediction, kwargs
        ):
            chunks = []
            response = await super().aresponses(
                messages,
                tools,
                include,
                store,
                add_security_risk_prediction,
                self._attempt_callback(on_token, chunks, options),
                call_context,
                **options,
            )
            if self._accept_length_attempt(response, attempt, generation_id):
                if on_token is not None:
                    for chunk in chunks:
                        result = on_token(chunk)
                        if inspect.isawaitable(result):
                            await result
                return response
        raise AssertionError("length-attempt iterator must return or raise")

    async def acompletion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        add_security_risk_prediction: bool = False,
        on_token: AnyTokenCallbackType | None = None,
        call_context: LLMCallContext | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        generation_id = uuid.uuid4().hex
        for attempt, options in self._length_attempts(
            messages, tools, add_security_risk_prediction, kwargs
        ):
            chunks: list[ModelResponseStream] = []
            response = await super().acompletion(
                messages,
                tools,
                add_security_risk_prediction,
                self._attempt_callback(on_token, chunks, options),
                call_context,
                **options,
            )
            if self._accept_length_attempt(response, attempt, generation_id):
                if on_token is not None:
                    for chunk in chunks:
                        result = on_token(chunk)
                        if inspect.isawaitable(result):
                            await result
                return response
        raise AssertionError("length-attempt iterator must return or raise")
