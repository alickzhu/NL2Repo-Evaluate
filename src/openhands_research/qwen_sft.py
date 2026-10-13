"""Render Qwen SFT with preserved reasoning and completion-only token labels."""

from __future__ import annotations

import copy
import json
from typing import Any

from jinja2 import nodes
from jinja2.visitor import NodeTransformer
from transformers import PreTrainedTokenizerBase
from transformers.utils.chat_template_utils import _compile_jinja_template

from .trajectories import content_text, reasoning_text


class _PreserveReasoning(NodeTransformer):
    replacements = 0

    def visit_Compare(self, node: nodes.Compare, *args, **kwargs):
        self.generic_visit(node, *args, **kwargs)
        if (
            isinstance(node.expr, nodes.Getattr)
            and isinstance(node.expr.node, nodes.Name)
            and node.expr.node.name == "loop"
            and node.expr.attr == "index0"
            and len(node.ops) == 1
            and node.ops[0].op == "gt"
            and isinstance(node.ops[0].expr, nodes.Getattr)
            and isinstance(node.ops[0].expr.node, nodes.Name)
            and node.ops[0].expr.node.name == "ns"
            and node.ops[0].expr.attr == "last_query_index"
        ):
            self.replacements += 1
            return nodes.Const(True).set_lineno(node.lineno)
        return node


def render_qwen(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    source = tokenizer.get_chat_template()
    environment = _compile_jinja_template(source).environment
    transformer = _PreserveReasoning()
    tree = transformer.visit(environment.parse(source))
    if transformer.replacements != 1:
        raise ValueError("unrecognized Qwen history-reasoning template")
    normalized = copy.deepcopy(messages)
    for message in normalized:
        content = content_text(message)
        if message["role"] == "assistant":
            message["reasoning_content"] = reasoning_text(message)
            if content.startswith("<think>\n") and "</think>" in content:
                content = content.split("</think>", 1)[1].lstrip("\n")
        message["content"] = content
        for call in message.get("tool_calls") or []:
            function = call["function"]
            if isinstance(function.get("arguments"), str):
                function["arguments"] = json.loads(function["arguments"])
    template = environment.from_string(tree)
    return template.render(
        messages=normalized,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=True,
        preserve_thinking=True,
        **tokenizer.special_tokens_map,
    )


def tokenize_sft(
    tokenizer: PreTrainedTokenizerBase, record: dict[str, Any]
) -> dict[str, Any]:
    if record.get("api_mode") == "responses":
        raise ValueError(
            "Responses records require a provider-specific tokenizer, not Qwen SFT"
        )
    messages = record["messages"]
    if record["message_loss_mask"] != [0] * (len(messages) - 1) + [1]:
        raise ValueError("expected one supervised terminal assistant response")
    if messages[-1]["role"] != "assistant":
        raise ValueError("SFT target must be an assistant response")
    tools = record.get("tools") or []
    prompt = render_qwen(tokenizer, messages[:-1], tools, add_generation_prompt=True)
    full = render_qwen(tokenizer, messages, tools, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("target rendering changed the inference prefix")
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = tokenizer.encode(full, add_special_tokens=False)
    if input_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenization changed the inference prefix")
    labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids) :]
    if not any(label != -100 for label in labels):
        raise ValueError("empty supervised response")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "prompt_length": len(prompt_ids),
    }
