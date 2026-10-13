"""Pinned OpenAI-compatible model profiles for research collection and eval."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar

from openhands.sdk import LLM
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.sdk.context.condenser.llm_summarizing_condenser import Reason
from openhands.sdk.context.view import View
from openhands.sdk.event.base import LLMConvertibleEvent
from openhands.sdk.utils import maybe_truncate
from pydantic import SecretStr, model_validator

from .length_retry import LengthRetryLLM, OutputLengthLimitError
from .responses_records import messages as responses_messages
from .sft_transport import GUARDED_MODELS, SFTTransportLLM
from .trajectories import think_message

QWEN38_FLASH_NEXT_MODEL = "Qwen/Qwen3.8-Flash-Next-FP8"
QWEN38_FLASH_NEXT_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
GLM52_FP8_MODEL = "zai-org/GLM-5.2-FP8"
GLM52_FP8_REVISION = "ba978f7d347eaf65d22f1a86833408afdb953541"
QWEN35_SFT_MODEL = "qwen35-27b-sft-5208"
MODEL_REVISIONS = {
    QWEN38_FLASH_NEXT_MODEL: QWEN38_FLASH_NEXT_REVISION,
    GLM52_FP8_MODEL: GLM52_FP8_REVISION,
    QWEN35_SFT_MODEL: "endpoint-defined",
}


class FullHistoryCondenser(LLMSummarizingCondenser):
    """Summarize complete public messages, including reasoning and tool payloads."""

    _terminal_exceptions: ClassVar[tuple[type[Exception], ...]] = (
        OutputLengthLimitError,
    )
    trigger_on_events: bool = True

    @model_validator(mode="after")
    def _disable_streaming_for_summary(self):
        # The SDK base validator disables SSE because summaries lack callbacks.
        # Our wrapper supplies an internal sink, so preserve its transport mode.
        if isinstance(self.llm, LengthRetryLLM):
            return self
        return super()._disable_streaming_for_summary()

    def get_condensation_reasons(
        self, view: View, agent_llm: LLM | None = None
    ) -> set[Reason]:
        reasons = super().get_condensation_reasons(view, agent_llm)
        if not self.trigger_on_events:
            reasons.discard(Reason.EVENTS)
        return reasons

    def _event_strings(
        self,
        events: Sequence[LLMConvertibleEvent],
        max_event_str_length: int | None = None,
    ) -> list[str]:
        messages = LLMConvertibleEvent.events_to_messages(list(events))
        native = self.llm.api_mode == "responses"
        if native:
            instructions, items = self.llm.format_messages_for_responses(messages)
            # Only provider-visible summary text is readable; never place an
            # encrypted reasoning blob into a summarizer's text prompt.
            formatted = responses_messages(items, instructions)
        else:
            formatted = self.llm.format_messages_for_llm(messages)
        return [
            maybe_truncate(
                json.dumps(
                    message if native else think_message(message), ensure_ascii=False
                ),
                truncate_after=max_event_str_length,
            )
            for message in formatted
        ]


@dataclass(frozen=True)
class ModelProfile:
    endpoint: str
    model: str = QWEN38_FLASH_NEXT_MODEL
    model_revision: str = QWEN38_FLASH_NEXT_REVISION
    temperature: float | None = 1.0
    seed: int | None = 42
    max_input_tokens: int = 262_144
    max_output_tokens: int = 32_768
    custom_tokenizer: str | None = None
    reasoning_effort: str = "xhigh"
    request_timeout: int = 7_200
    request_retries: int = 8  # Total API attempts, including the initial request.
    length_retries: int = 3
    length_retry_mode: str = "bounded"
    max_tokens_ceiling: int = 65_536
    length_growth_factor: float = 2.0
    context_safety_tokens: int = 512
    condenser_enabled: bool = True
    condenser_max_events: int = 0
    condenser_keep_first: int = 4
    condenser_max_tokens: int = 196_608
    condenser_max_output_tokens: int = 32_768
    api_mode: str = "chat"
    api_model: str | None = None  # Optional served name; keep family capabilities.

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint, str) or not self.endpoint.startswith(
            ("http://", "https://")
        ):
            raise ValueError("endpoint must be an HTTP(S) URL")
        if not self.endpoint.rstrip("/").endswith("/v1"):
            raise ValueError("endpoint must end in /v1")
        if self.api_mode not in {"chat", "responses"}:
            raise ValueError("api_mode must be chat or responses")
        if self.api_model is not None and (
            not isinstance(self.api_model, str)
            or not self.api_model
            or any(c.isspace() for c in self.api_model)
        ):
            raise ValueError("api_model must be a nonempty model ID without whitespace")
        if self.api_mode == "responses":
            if not self.model or "/" in self.model or self.model.strip() != self.model:
                raise ValueError("Responses model must be the bare endpoint model ID")
            if self.temperature is not None or self.seed is not None:
                raise ValueError(
                    "Responses reasoning profiles omit temperature and seed"
                )
        elif self.model not in MODEL_REVISIONS:
            raise ValueError("unsupported pinned model profile")
        if (
            self.api_mode == "chat"
            and self.model_revision != MODEL_REVISIONS[self.model]
        ):
            raise ValueError("unexpected pinned model revision")
        if self.model == GLM52_FP8_MODEL and self.reasoning_effort not in {
            "high",
            "max",
        }:
            raise ValueError("GLM-5.2 reasoning effort must be high or max")
        if self.temperature is not None and not 0 <= self.temperature:
            raise ValueError("temperature must be nonnegative")
        if self.api_mode == "chat" and self.max_input_tokens != 262_144:
            raise ValueError("the validated model context is 262144 tokens")
        if not 0 < self.max_output_tokens < self.max_input_tokens:
            raise ValueError("invalid output token budget")
        if type(self.condenser_enabled) is not bool:
            raise ValueError("condenser_enabled must be a boolean")
        if not 0 < self.condenser_max_tokens < self.max_input_tokens:
            raise ValueError("invalid condenser trigger budget")
        if type(self.condenser_max_events) is not int or self.condenser_max_events < 0:
            raise ValueError("condenser_max_events must be nonnegative; 0 disables it")
        if not 0 < self.condenser_max_output_tokens < self.max_input_tokens:
            raise ValueError("invalid condenser output budget")
        if type(self.length_retries) is not int or self.length_retries < 1:
            raise ValueError("length_retries must be a positive attempt count per cap")
        if self.length_retry_mode not in {"bounded", "fixed_until_complete"}:
            raise ValueError("invalid length retry mode")
        if (
            type(self.max_tokens_ceiling) is not int
            or self.max_tokens_ceiling
            < max(self.max_output_tokens, self.condenser_max_output_tokens)
            or self.max_tokens_ceiling >= self.max_input_tokens
        ):
            raise ValueError("output ceiling must cover initial caps and fit context")
        if (
            type(self.length_growth_factor) not in (int, float)
            or not math.isfinite(self.length_growth_factor)
            or self.length_growth_factor <= 1
        ):
            raise ValueError("length_growth_factor must be finite and greater than 1")
        if type(self.context_safety_tokens) is not int or not (
            0 <= self.context_safety_tokens < self.max_input_tokens
        ):
            raise ValueError("invalid context safety-token reserve")
        if (
            self.custom_tokenizer is not None
            and not Path(self.custom_tokenizer).is_dir()
        ):
            raise ValueError("custom_tokenizer must be a local tokenizer snapshot")

    @property
    def base_url(self) -> str:
        return self.endpoint.rstrip("/")

    @property
    def litellm_model(self) -> str:
        return f"openai/{self.api_model or self.model}"

    def public_dict(self) -> dict[str, object]:
        return asdict(self)

    def _llm(
        self,
        *,
        usage_id: str,
        output_tokens: int,
        completion_dir: Path,
        api_key: str,
        stream: bool,
    ) -> LLM:
        if self.api_mode == "responses":
            return LengthRetryLLM(
                model=self.litellm_model,
                model_canonical_name=self.model,
                base_url=self.base_url,
                api_key=SecretStr(api_key),
                api_mode="responses",
                capability_overrides={
                    "supports_responses_api": True,
                    "supports_reasoning_effort": True,
                    "supports_sampling_params": False,
                },
                temperature=None,
                seed=None,
                reasoning_effort=self.reasoning_effort,
                max_input_tokens=self.max_input_tokens,
                max_output_tokens=output_tokens,
                custom_tokenizer=self.custom_tokenizer,
                max_message_chars=1_000_000,
                stream=stream,
                timeout=self.request_timeout,
                num_retries=self.request_retries,
                retry_min_wait=8,
                retry_max_wait=16,
                retry_multiplier=8.0,
                length_retries=self.length_retries,
                length_retry_mode=self.length_retry_mode,
                max_tokens_ceiling=self.max_tokens_ceiling,
                length_growth_factor=self.length_growth_factor,
                context_safety_tokens=self.context_safety_tokens,
                drop_params=False,
                log_completions=True,
                log_completions_folder=str(completion_dir),
                usage_id=usage_id,
            )
        template_kwargs = {
            "enable_thinking": True,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.model in {QWEN38_FLASH_NEXT_MODEL, QWEN35_SFT_MODEL}:
            template_kwargs["preserve_thinking"] = True
        else:
            template_kwargs["clear_thinking"] = False
        llm_type = SFTTransportLLM if self.api_model in GUARDED_MODELS else LengthRetryLLM
        return llm_type(
            model=self.litellm_model,
            model_canonical_name=(
                "qwen3.8-flash" if self.model == QWEN38_FLASH_NEXT_MODEL
                else "qwen3.5-27b" if self.model == QWEN35_SFT_MODEL else "glm-5.2"
            ),
            base_url=self.base_url,
            api_key=SecretStr(api_key),
            api_mode="chat",
            capability_overrides={
                "supports_reasoning_effort": False,
                "supports_responses_api": False,
                "supports_sampling_params": True,
                "thinking_mode": "unknown",
                "send_reasoning_content": True,
            },
            native_tool_calling=True,
            temperature=self.temperature,
            seed=self.seed,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=output_tokens,
            custom_tokenizer=self.custom_tokenizer,
            max_message_chars=1_000_000,
            reasoning_effort=None,
            litellm_extra_body={"chat_template_kwargs": template_kwargs},
            stream=stream,
            timeout=self.request_timeout,
            num_retries=self.request_retries,
            retry_min_wait=8,
            retry_max_wait=16,
            retry_multiplier=8.0,
            length_retries=self.length_retries,
            length_retry_mode=self.length_retry_mode,
            max_tokens_ceiling=self.max_tokens_ceiling,
            length_growth_factor=self.length_growth_factor,
            context_safety_tokens=self.context_safety_tokens,
            drop_params=False,
            log_completions=True,
            log_completions_folder=str(completion_dir),
            usage_id=usage_id,
        )

    def build_agent_llm(self, completion_dir: Path, api_key: str) -> LLM:
        return self._llm(
            usage_id="solver",
            output_tokens=self.max_output_tokens,
            completion_dir=completion_dir,
            api_key=api_key,
            stream=True,
        )

    def build_condenser(
        self, completion_dir: Path, api_key: str
    ) -> FullHistoryCondenser | None:
        if not self.condenser_enabled:
            return None
        llm = self._llm(
            usage_id="condenser",
            output_tokens=self.condenser_max_output_tokens,
            completion_dir=completion_dir,
            api_key=api_key,
            stream=True,
        )
        return FullHistoryCondenser(
            llm=llm,
            max_size=self.condenser_max_events or 80,
            trigger_on_events=self.condenser_max_events > 0,
            keep_first=self.condenser_keep_first,
            max_tokens=self.condenser_max_tokens,
        )
