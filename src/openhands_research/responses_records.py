"""Audit projections only; native Responses request/output items stay untouched."""

from __future__ import annotations

import copy
import json


def _arguments(value):
    # The SDK reserializes validated tool arguments. Compare JSON values, not
    # insignificant whitespace/key order, while rejecting ambiguous JSON.
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate tool argument")
            result[key] = item
        return result
    parsed = json.loads(value, object_pairs_hook=unique)
    return json.dumps(parsed, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _same_native_item(item, expected):
    if expected is None or item.get("type") != expected.get("type"):
        return False
    if item["type"] == "reasoning":
        # Optional empty content is [] in response models, omitted in requests.
        sequence = lambda obj, key: [] if obj.get(key) is None else obj[key]
        return (item.get("id") == expected.get("id")
                and item.get("encrypted_content") == expected.get("encrypted_content")
                and sequence(item, "summary") == sequence(expected, "summary")
                and sequence(item, "content") == sequence(expected, "content"))
    try:
        return (item.get("call_id") == expected.get("call_id")
                and item.get("name") == expected.get("name")
                and _arguments(item.get("arguments")) == _arguments(expected.get("arguments")))
    except (TypeError, ValueError):
        return False


def audit_native_history(records):
    known = {}
    violations = []
    checked = 0
    for record in records:
        if record.get("llm_path") != "responses" or record.get("usage_id") != "solver":
            continue
        inputs = record.get("input")
        for item in inputs if isinstance(inputs, list) else []:
            if item.get("type") not in {"reasoning", "function_call"}:
                continue
            identifier = item.get("id") or item.get("call_id")
            expected = known.get(identifier)
            if not _same_native_item(item, expected):
                violations.append(
                    f"call {record.get('sequence')}: native {item['type']} history mismatch"
                )
            else:
                checked += 1
        for item in (record.get("response") or {}).get("output") or []:
            if item.get("type") in {"reasoning", "function_call"}:
                known[item.get("id") or item.get("call_id")] = item
    return checked, violations


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "")
        for part in content or []
        if part.get("type") in {"text", "input_text", "output_text", "summary_text"}
    )


def messages(items, instructions=None):
    result = []
    if instructions:
        result.append({"role": "system", "content": instructions})
    if isinstance(items, str):
        return [*result, {"role": "user", "content": items}]
    pending = None
    for item in items or []:
        kind = item.get("type", "message")
        if kind in {"reasoning", "function_call"} or item.get("role") == "assistant":
            if pending is None:
                pending = {"role": "assistant", "content": ""}
                result.append(pending)
            if kind == "reasoning":
                pending["reasoning_content"] = _text(item.get("summary"))
            elif kind == "function_call":
                pending.setdefault("tool_calls", []).append(
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"],
                        },
                    }
                )
            else:
                pending["content"] += _text(item.get("content"))
        else:
            pending = None
            if kind == "function_call_output":
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": item["call_id"],
                        "content": _text(item.get("output")),
                    }
                )
            elif kind == "message" and item.get("role") in {
                "user",
                "system",
                "developer",
            }:
                result.append(
                    {"role": item["role"], "content": _text(item.get("content"))}
                )
            else:
                raise ValueError(f"unsupported Responses audit item: {kind}")
    return result


def projection(record):
    if record.get("llm_path") != "responses":
        return record
    result = copy.deepcopy(record)
    result["messages"] = messages(record.get("input"), record.get("instructions"))
    response = record.get("response")
    if not response:
        return result
    output = messages(response.get("output"))
    if not output and response.get("status") != "completed":
        output = [{"role": "assistant", "content": ""}]
    if len(output) != 1 or output[0]["role"] != "assistant":
        raise ValueError("Responses output is not one assistant turn")
    if response.get("status") == "completed":
        reason = "tool_calls" if output[0].get("tool_calls") else "stop"
    elif (response.get("incomplete_details") or {}).get(
        "reason"
    ) == "max_output_tokens":
        reason = "length"
    else:
        reason = "content_filter"
    result["response"] = {
        "id": response["id"],
        "choices": [{"message": output[0], "finish_reason": reason}],
    }
    result["native_response"] = response
    return result
