import copy

import pytest
from openhands_research.qwen_sft import render_qwen, tokenize_sft
from openhands_research.trajectories import audit_records, sft_records, think_message
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def trace():
    answer = {
        "role": "assistant",
        "content": "Answer",
        "reasoning_content": "HISTORY_REASONING_CANARY",
    }
    summary = {
        "role": "assistant",
        "content": "The task is partly done.",
        "reasoning_content": "SUMMARY_REASONING_CANARY",
    }
    records = []
    for sequence, (usage, messages, target) in enumerate(
        [
            ("solver", [{"role": "user", "content": "Task"}], answer),
            (
                "condenser",
                [{"role": "user", "content": "Summarize the public history"}],
                summary,
            ),
            (
                "solver",
                [
                    {"role": "user", "content": "Task"},
                    answer,
                    {"role": "user", "content": summary["content"]},
                ],
                {
                    "role": "assistant",
                    "content": "Done",
                    "reasoning_content": "FINAL_REASONING_CANARY",
                },
            ),
        ]
    ):
        records.append(
            {
                "sequence": sequence,
                "usage_id": usage,
                "source_log": f"call-{sequence}.json",
                "request_sha256": "fixture",
                "messages": copy.deepcopy(messages),
                "response": {
                    "id": f"response-{sequence}",
                    "choices": [{"message": target}],
                },
            }
        )
    events = [
        {"event_type": "MessageEvent", "event": {"llm_response_id": "response-0"}},
        {
            "event_type": "Condensation",
            "event": {
                "llm_response_id": "response-1",
                "summary": summary["content"],
            },
        },
        {"event_type": "MessageEvent", "event": {"llm_response_id": "response-2"}},
    ]
    return records, events


def test_summary_is_one_terminal_target_then_masked_history(trace):
    records, events = trace
    audit = audit_records(records, events)
    assert audit["ok"] and audit["history_reasoning_checked"] == 1
    assert audit["post_summary_calls_checked"] == 1
    rows = sft_records(records, events)
    assert [row["segment_index"] for row in rows] == [0, 0, 1]
    assert rows[1]["target_kind"] == "summary" and rows[1]["segment_terminal"]
    assert rows[2]["message_loss_mask"] == [0, 0, 0, 1]
    assert rows[2]["messages"][1]["content"] == (
        "<think>\nHISTORY_REASONING_CANARY\n</think>\n\nAnswer"
    )
    assert rows[2]["messages"][2]["content"] == "The task is partly done."
    assert sum(sum(row["message_loss_mask"]) for row in rows) == 3


def test_audit_detects_lost_reasoning_after_a_user_summary(trace):
    records, events = trace
    del records[2]["messages"][1]["reasoning_content"]
    audit = audit_records(records, events)
    assert not audit["ok"] and audit["violation_count"] == 1


def test_audit_detects_missing_accepted_call(trace):
    records, events = trace
    audit = audit_records(records[:2], events)
    assert not audit["ok"]
    assert "no request record" in audit["violations"][0]


def test_truncated_response_remains_history_but_is_not_an_sft_target(trace):
    records, events = trace
    records[0]["response"]["choices"][0]["finish_reason"] = "length"
    audit = audit_records(records, events)
    assert audit["ok"] and audit["truncated_responses"] == 1
    rows = sft_records(records, events)
    assert [row["response_id"] for row in rows] == ["response-1", "response-2"]
    assert "HISTORY_REASONING_CANARY" in rows[-1]["messages"][1]["content"]


def test_repeated_reasoning_only_responses_match_in_order(trace):
    records, events = trace
    records[0]["response"]["choices"][0]["message"]["content"] = ""
    records[1]["usage_id"] = "solver"
    records[1]["response"]["choices"][0]["message"]["content"] = ""
    first = copy.deepcopy(records[0]["response"]["choices"][0]["message"])
    second = copy.deepcopy(records[1]["response"]["choices"][0]["message"])
    records[1]["messages"] = [{"role": "user", "content": "Task"}, first]
    records[2]["messages"] = [
        {"role": "user", "content": "Task"},
        first,
        second,
        {"role": "user", "content": "Continue"},
    ]
    events[1] = {
        "event_type": "MessageEvent",
        "event": {"llm_response_id": "response-1"},
    }
    assert audit_records(records, events)["ok"]
    records[2]["messages"][1:3] = [second, first]
    assert not audit_records(records, events)["ok"]


def test_parallel_tool_response_matches_ordered_split_history():
    first = {
        "role": "assistant",
        "reasoning_content": "PARALLEL_TOOL_REASONING_CANARY",
        "tool_calls": [
            {"id": "call-a", "function": {"name": "file_editor"}},
            {"id": "call-b", "function": {"name": "file_editor"}},
        ],
    }
    split = [
        {
            "role": "assistant",
            "reasoning_content": "PARALLEL_TOOL_REASONING_CANARY",
            "tool_calls": [first["tool_calls"][0]],
        },
        {
            "role": "assistant",
            "tool_calls": [first["tool_calls"][1]],
        },
    ]
    records = [
        {
            "sequence": 0,
            "usage_id": "solver",
            "messages": [{"role": "user", "content": "Task"}],
            "response": {"id": "response-0", "choices": [{"message": first}]},
        },
        {
            "sequence": 1,
            "usage_id": "solver",
            "messages": [
                {"role": "user", "content": "Task"},
                *split,
                {"role": "user", "content": "Continue"},
            ],
            "response": {
                "id": "response-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done",
                            "reasoning_content": "FINAL_REASONING_CANARY",
                        }
                    }
                ],
            },
        },
    ]
    events = [
        {"event_type": "ActionEvent", "event": {"llm_response_id": "response-0"}},
        {"event_type": "MessageEvent", "event": {"llm_response_id": "response-1"}},
    ]

    audit = audit_records(records, events)
    assert audit["ok"] and audit["history_reasoning_checked"] == 1

    missing = copy.deepcopy(records)
    missing[1]["messages"][1].pop("reasoning_content")
    assert not audit_records(missing, events)["ok"]

    reordered = copy.deepcopy(records)
    reordered[1]["messages"][1:3] = reversed(reordered[1]["messages"][1:3])
    assert not audit_records(reordered, events)["ok"]


def test_parallel_tool_history_requires_a_complete_nonoverlapping_partition():
    tool_calls = [
        {"id": f"call-{name}", "function": {"name": "file_editor"}}
        for name in ("a", "b", "c")
    ]
    target = {
        "role": "assistant",
        "reasoning_content": "PARALLEL_TOOL_REASONING_CANARY",
        "tool_calls": tool_calls,
    }
    records = [
        {
            "sequence": 0,
            "usage_id": "solver",
            "messages": [{"role": "user", "content": "Task"}],
            "response": {"id": "response-0", "choices": [{"message": target}]},
        },
        {
            "sequence": 1,
            "usage_id": "solver",
            "messages": [{"role": "user", "content": "Task"}],
            "response": {
                "id": "response-1",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Done",
                            "reasoning_content": "FINAL_REASONING_CANARY",
                        }
                    }
                ],
            },
        },
    ]
    events = [
        {"event_type": "ActionEvent", "event": {"llm_response_id": "response-0"}},
        {"event_type": "MessageEvent", "event": {"llm_response_id": "response-1"}},
    ]

    def projected(start: int, end: int, *, reasoning: bool = False):
        message = {"role": "assistant", "tool_calls": tool_calls[start:end]}
        if reasoning:
            message["reasoning_content"] = "PARALLEL_TOOL_REASONING_CANARY"
        return message

    valid = copy.deepcopy(records)
    valid[1]["messages"].extend(
        [projected(0, 1, reasoning=True), projected(1, 3)]
    )
    assert audit_records(valid, events)["ok"]

    omitted_middle = copy.deepcopy(records)
    omitted_middle[1]["messages"].extend(
        [projected(0, 1, reasoning=True), projected(2, 3)]
    )
    assert not audit_records(omitted_middle, events)["ok"]

    omitted_first = copy.deepcopy(records)
    omitted_first[1]["messages"].extend([projected(1, 2), projected(2, 3)])
    assert not audit_records(omitted_first, events)["ok"]

    overlap = copy.deepcopy(records)
    overlap[1]["messages"].extend(
        [projected(0, 2, reasoning=True), projected(1, 3)]
    )
    assert not audit_records(overlap, events)["ok"]


def test_think_format_preserves_tools_without_duplicate_delimiters():
    message = {
        "role": "assistant",
        "content": "<think>\nReason\n</think>\n\nAnswer",
        "reasoning_content": "Reason",
        "tool_calls": [{"id": "call-1", "function": {"name": "terminal"}}],
    }
    formatted = think_message(message)
    assert formatted["content"] == message["content"]
    assert "reasoning_content" not in formatted
    assert formatted["tool_calls"] == message["tool_calls"]
    assert "reasoning_content" in message


def test_qwen_rendering_keeps_old_reasoning_and_masks_prompt_tokens(trace):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template=(
            "{% set ns = namespace(last_query_index=2) %}"
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message.role + '\\n' }}"
            "{% if message.role == 'assistant' %}"
            "{% if loop.index0 > ns.last_query_index %}"
            "{{ '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}"
            "{% endif %}{% endif %}"
            "{{ message.content + '<|im_end|>\\n' }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}"
            "{% endif %}"
        ),
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<think>"]}
    )
    record = sft_records(*trace)[-1]
    rendered = render_qwen(
        tokenizer, record["messages"], [], add_generation_prompt=False
    )
    assert "<think>\nHISTORY_REASONING_CANARY\n</think>" in rendered
    assert rendered.count("<think>") == 2
    encoded = tokenize_sft(tokenizer, record)
    assert all(label == -100 for label in encoded["labels"][: encoded["prompt_length"]])
    assert all(label != -100 for label in encoded["labels"][encoded["prompt_length"] :])
