"""Exact SDK request records, reasoning audits, and completion-level SFT export."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from .contracts import DatasetTask
from .ledger import AttemptPaths, append_jsonl, atomic_json, atomic_jsonl
from .responses_records import audit_native_history, projection


def content_text(message: dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(
        not isinstance(block, dict) or block.get("type") != "text" for block in content
    ):
        raise ValueError("SFT export currently requires text-only messages")
    return "".join(block["text"] for block in content)


def reasoning_text(message: dict[str, Any]) -> str:
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, str):
            raise ValueError("reasoning must be text")
        return reasoning
    content = content_text(message)
    if content.startswith("<think>\n") and "</think>" in content:
        return content[len("<think>\n") : content.index("</think>")].rstrip("\n")
    return ""


def think_message(message: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(message)
    content = content_text(message)
    if message["role"] == "assistant":
        reasoning = reasoning_text(message).strip("\n")
        if content.startswith("<think>\n") and "</think>" in content:
            content = content.split("</think>", 1)[1].lstrip("\n")
        result["content"] = f"<think>\n{reasoning}\n</think>\n\n{content}"
        result.pop("reasoning_content", None)
        result.pop("reasoning", None)
    else:
        result["content"] = content
    return result


class TrajectoryRecorder:
    """One recorder per attempt; solver and condenser callbacks share its lock."""

    def __init__(self, paths: AttemptPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self._sequence = 0

    def record(self, usage_id: str, filename: str, log_data: str) -> None:
        if Path(filename).name != filename:
            raise ValueError("completion filename must be a basename")
        payload = json.loads(log_data)
        keys = (
            ("instructions", "input", "tools", "kwargs")
            if payload.get("llm_path") == "responses"
            else ("messages", "tools", "kwargs")
        )
        request = {key: payload.get(key) for key in keys}
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        with self._lock:
            record = {
                **payload,
                "schema_version": 1,
                "sequence": self._sequence,
                "usage_id": usage_id,
                "source_log": filename,
                "request_sha256": digest,
            }
            atomic_json(self.paths.completions / filename, record)
            append_jsonl(self.paths.root / "trajectory.jsonl", record)
            self._sequence += 1


def _history_key(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if calls:
        return "tools:" + json.dumps([call["id"] for call in calls])
    return "content:" + content_text(message)


def _history_projections(
    message: dict[str, Any], sequence: int
) -> list[tuple[str, tuple[int, int, int, int], str]]:
    calls = message.get("tool_calls") or []
    reasoning = reasoning_text(message)
    if not calls:
        return [(_history_key(message), (sequence, 0, 1, 1), reasoning)]

    call_ids = [call["id"] for call in calls]
    projections = []
    for start in range(len(call_ids)):
        projected_reasoning = reasoning if start == 0 else ""
        for end in range(start + 1, len(call_ids) + 1):
            key = "tools:" + json.dumps(call_ids[start:end])
            projections.append(
                (
                    key,
                    (sequence, start, end, len(call_ids)),
                    projected_reasoning,
                )
            )
    return projections


def _continues_history_partition(
    previous: tuple[int, int, int, int] | None,
    candidate: tuple[int, int, int, int],
) -> bool:
    sequence, start, _end, total = candidate
    if previous is None:
        return start == 0
    previous_sequence, _previous_start, previous_end, previous_total = previous
    if sequence == previous_sequence:
        return total == previous_total and start == previous_end
    return (
        sequence > previous_sequence and previous_end == previous_total and start == 0
    )


def audit_records(
    records: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    native_checked, native_violations = audit_native_history(records)
    records = [projection(record) for record in records]
    accepted = {
        row["event"]["llm_response_id"]
        for row in events
        if row["event"].get("llm_response_id")
    }
    summaries = {
        row["event"]["llm_response_id"]: row["event"].get("summary") or ""
        for row in events
        if row["event_type"] == "Condensation"
    }
    expected_reasoning: dict[str, list[tuple[tuple[int, int, int, int], str]]] = {}
    observed_ids: set[str] = set()
    violations = list(native_violations)
    history_checked = 0
    reasoning_responses = 0
    truncated_responses = 0
    post_summary_calls = 0
    pending_summary = ""
    for record in records:
        if "response" not in record:
            continue
        response = record["response"]
        response_id = response["id"]
        observed_ids.add(response_id)
        messages = record["messages"]
        if record["usage_id"] == "solver":
            previous_position = None
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                candidates = expected_reasoning.get(_history_key(message), [])
                actual = reasoning_text(message)
                match = next(
                    (
                        position
                        for position, expected in candidates
                        if _continues_history_partition(previous_position, position)
                        and actual == expected
                    ),
                    None,
                )
                if match is None:
                    violations.append(
                        f"call {record['sequence']} history {index}: reasoning mismatch"
                    )
                else:
                    previous_position = match
                    history_checked += bool(actual)
            if (
                previous_position is not None
                and previous_position[2] != previous_position[3]
            ):
                violations.append(
                    f"call {record['sequence']}: incomplete parallel tool history"
                )
            if pending_summary:
                if not any(
                    message["role"] == "user"
                    and pending_summary.strip() in content_text(message)
                    for message in messages
                ):
                    violations.append(
                        f"call {record['sequence']}: missing summary history"
                    )
                post_summary_calls += 1
                pending_summary = ""
        target = response["choices"][0]["message"]
        truncated_responses += response["choices"][0].get("finish_reason") == "length"
        reasoning_responses += bool(reasoning_text(target))
        if record["usage_id"] == "solver":
            for key, position, expected in _history_projections(
                target, record["sequence"]
            ):
                expected_reasoning.setdefault(key, []).append((position, expected))
        elif response_id in summaries:
            pending_summary = summaries[response_id]
    missing = accepted - observed_ids
    if missing:
        violations.append(f"{len(missing)} accepted responses have no request record")
    return {
        "schema_version": 1,
        "ok": not violations and bool(observed_ids),
        "calls": len(records),
        "responses": len(observed_ids),
        "reasoning_responses": reasoning_responses,
        "truncated_responses": truncated_responses,
        "history_reasoning_checked": history_checked,
        "native_history_items_checked": native_checked,
        "condensations": len(summaries),
        "post_summary_calls_checked": post_summary_calls,
        "violation_count": len(violations),
        "violations": violations[:20],
    }


def sft_records(
    records: list[dict[str, Any]], events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    records = [projection(record) for record in records]
    accepted = {
        row["event"]["llm_response_id"]
        for row in events
        if row["event"].get("llm_response_id")
    }
    summaries = {
        row["event"]["llm_response_id"]
        for row in events
        if row["event_type"] == "Condensation"
    }
    rows = []
    segment = 0
    seen = set()
    for record in records:
        response = record.get("response")
        if not response or response["id"] not in accepted:
            continue
        response_id = response["id"]
        if response_id in seen:
            raise ValueError("duplicate supervised response")
        seen.add(response_id)
        is_summary = response_id in summaries
        finish_reason = response["choices"][0].get("finish_reason")
        if finish_reason in {"length", "content_filter"}:
            segment += int(is_summary)
            continue
        native = record.get("llm_path") == "responses"
        convert = copy.deepcopy if native else think_message
        messages = [convert(message) for message in record["messages"]]
        target = convert(response["choices"][0]["message"])
        rows.append(
            {
                "schema_version": 1,
                "response_id": response_id,
                "source_log": record["source_log"],
                "request_sha256": record["request_sha256"],
                "segment_index": segment,
                "target_kind": "summary" if is_summary else "solver",
                "finish_reason": finish_reason,
                "segment_terminal": is_summary,
                "messages": [*messages, target],
                "message_loss_mask": [0] * len(messages) + [1],
                "tools": record.get("tools") or [],
                "chat_template_kwargs": {} if native else {"preserve_thinking": True},
                **(
                    {
                        "api_mode": "responses",
                        "reasoning_visibility": "provider_exposed_only",
                        "native_input": record.get("input"),
                        "native_output": record["native_response"].get("output"),
                    }
                    if native
                    else {}
                ),
            }
        )
        if is_summary:
            segment += 1
    return rows


def export_trajectory(paths: AttemptPaths, task: DatasetTask) -> dict[str, Any]:
    trace_path = paths.root / "trajectory.jsonl"
    records = (
        [json.loads(line) for line in trace_path.read_text().splitlines()]
        if trace_path.exists()
        else []
    )
    events = (
        [json.loads(line) for line in paths.events.read_text().splitlines()]
        if paths.events.exists()
        else []
    )
    audit = audit_records(records, events)
    split = task.public.metadata.get("split", "unspecified")
    identity = {
        "run_id": paths.run_id,
        "task_key": paths.task_key,
        "task_id": paths.task_id,
        "attempt_index": paths.attempt_index,
        "dataset_id": task.public.dataset_id,
        "dataset_revision": task.public.dataset_revision,
        "split": split,
        "training_eligible": split == "train",
    }
    rows = sft_records(records, events) if audit["ok"] else []
    atomic_jsonl(paths.root / "sft.jsonl", ({**identity, **row} for row in rows))
    audit["sft_rows"] = len(rows)
    atomic_json(paths.root / "reasoning_audit.json", {**identity, **audit})
    return audit
