"""Run TaskForge gold-path validation inside the final outer container.

The module is copied/executed *inside* the candidate image by OpenPAI.  It does
not launch Docker.  The original checkout is treated as a perfect agent output
and must score every recorded ``passed_ptp`` after the real clean/patch/install
sequence.
"""

from __future__ import annotations

import argparse
import hashlib
import errno
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

if __package__:
    from ._submission import materialize_implementation
else:
    from _submission import materialize_implementation


SKIP_COPY = frozenset({
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".nox", "build", "dist",
})

# Every final task image contains the exact BuildSpec used to construct its H3
# environment.  This is more durable than the historical Blob checkpoints,
# which may be compacted after image publication.
FROZEN_IMAGE_BUILDSPEC = Path("/opt/taskforge_environment_expected.json")


CONTEXT_PLUGIN_SOURCE = r'''"""Bind coverage.py contexts to pytest node IDs."""
from __future__ import annotations

import coverage


def _switch(item, phase):
    current = coverage.Coverage.current()
    if current is not None:
        current.switch_context(f"{item.nodeid}|{phase}")


def pytest_runtest_setup(item):
    _switch(item, "setup")


def pytest_runtest_call(item):
    _switch(item, "run")


def pytest_runtest_teardown(item):
    _switch(item, "teardown")
'''


OUTCOME_PLUGIN_SOURCE = r'''"""Record canonical pytest outcomes for TaskForge."""
from __future__ import annotations

import json
import os

import pytest


_OUTCOMES = {}


def _merge(previous, current):
    priority = {
        "passed": 0,
        "skipped": 1,
        "xfailed": 1,
        "failed": 2,
        "error": 3,
    }
    if previous is None:
        return current
    return max((previous, current), key=lambda value: priority.get(value, 3))


def _canonical_nodeid(item):
    raw = str(item.nodeid).strip().replace("\\", "/")
    _raw_path, separator, selector = raw.partition("::")
    # Test suites sometimes monkeypatch pathlib.Path/os.name while checking
    # platform-specific behavior.  The validation reporter is infrastructure,
    # so it must not instantiate Path after user tests have started.
    workspace = os.path.abspath(
        os.environ.get("TASKFORGE_PYTEST_WORKSPACE") or os.getcwd()
    )
    try:
        absolute = os.path.abspath(os.fspath(item.path))
        relative = os.path.relpath(absolute, workspace)
        path = relative.replace("\\", "/")
        if path == ".." or path.startswith("../"):
            path = _raw_path.lstrip("./")
    except (AttributeError, OSError, TypeError, ValueError):
        path = _raw_path.lstrip("./")
    return path + (("::" + selector) if separator else "")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    value = None
    if report.failed:
        value = "failed" if report.when == "call" else "error"
    elif report.skipped:
        value = "xfailed" if getattr(report, "wasxfail", False) else "skipped"
    elif report.when == "call" and report.passed:
        value = "passed"
    if value is not None:
        nodeid = _canonical_nodeid(item)
        _OUTCOMES[nodeid] = _merge(_OUTCOMES.get(nodeid), value)


def pytest_sessionfinish(session, exitstatus):
    destination = os.environ.get("TASKFORGE_PYTEST_OUTCOME_FILE")
    if not destination:
        return
    parent = os.path.dirname(destination)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write(
        json.dumps({
            "schema_version": 1,
            "exitstatus": int(exitstatus),
            "outcomes": _OUTCOMES,
        }, sort_keys=True),
        )
'''


CONTEXT_RUNNER_SOURCE = r'''"""Run one existing TaskForge pytest chunk under context coverage."""
from __future__ import annotations

import json
import sys

import pytest
import taskforge_validation_context_plugin
import taskforge_validation_outcome_plugin

plugins = [
    taskforge_validation_context_plugin,
    taskforge_validation_outcome_plugin,
]
if __import__("os").environ.get("TASKFORGE_RUNTIME_BEHAVIOR_CONFIG"):
    import taskforge_runtime_behavior_plugin
    plugins.append(taskforge_runtime_behavior_plugin)


def main():
    arguments = json.load(open(sys.argv[1], encoding="utf-8"))
    return pytest.main(arguments, plugins=plugins)


if __name__ == "__main__":
    raise SystemExit(main())
'''


DIRECT_RUNNER_SOURCE = r'''"""Run pytest arguments from JSON, outside the OS argv limit."""
from __future__ import annotations

import json
import os
import sys

import pytest
import taskforge_validation_outcome_plugin

plugins = [taskforge_validation_outcome_plugin]
if os.environ.get("TASKFORGE_RUNTIME_BEHAVIOR_CONFIG"):
    import taskforge_runtime_behavior_plugin
    plugins.append(taskforge_runtime_behavior_plugin)

def main():
    arguments = json.load(open(sys.argv[1], encoding="utf-8"))
    return pytest.main(arguments, plugins=plugins)


if __name__ == "__main__":
    raise SystemExit(main())
'''


RUNTIME_BEHAVIOR_PLUGIN_SOURCE = r'''"""Capture private, bounded runtime evidence for Specification compilation.

This file is generated inside a validated task image.  The payload is a
private construction artifact: the public Writer never receives it directly.
Values are depth/width/byte bounded and credential-shaped fields are redacted,
but ordinary test inputs and outputs are deliberately retained because they
are often the only executable description of a public API.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import inspect
import json
import math
import os
import sys
import threading

import pytest


with open(os.environ["TASKFORGE_RUNTIME_BEHAVIOR_CONFIG"], encoding="utf-8") as _handle:
    _config = json.load(_handle)
_output = str(_config["output"])
_roots = tuple(sorted({str(item) for item in _config.get("import_roots") or [] if item}))
_root_discovery = dict(_config.get("root_discovery") or {})
_identity = dict(_config.get("identity") or {})
_scope = str(_config.get("scope") or "targeted")
_trace_mode = str(_config.get("trace_mode") or "bounded_value")
if _trace_mode not in {"bounded_value", "minimal"}:
    _trace_mode = "bounded_value"
_selected = set(str(item) for item in _config.get("node_ids") or [])
_trace_all = bool(_config.get("trace_all", "node_ids" not in _config))
_current = ""
_symbols = defaultdict(Counter)
_fixture_types = defaultdict(set)
_fixtures = defaultdict(list)
_parameters = defaultdict(dict)
_events = defaultdict(list)
_reports = defaultdict(dict)
_active_frames = {}
_event_symbol_counts = defaultdict(Counter)
_sampled_nodes = set()
_detail_unavailable = defaultdict(dict)
_executed = set()
_truncated_nodes = set()
_lock = threading.RLock()
_max_symbols = max(1, int(_config.get("max_symbols_per_node") or 128))
_max_events = max(1, int(_config.get("max_events_per_node") or 512))
_max_events_per_symbol = max(
    1, int(_config.get("max_events_per_symbol") or 2)
)
_max_value_chars = max(32, int(_config.get("max_value_chars") or 1000))
_max_collection_chars = max(1, int(_config.get("max_collection_items") or 20))
_max_depth = max(1, int(_config.get("max_value_depth") or 3))
_max_event_bytes = max(256, int(_config.get("max_event_bytes") or 2000))
_redacted_names = {
    "authorization", "cookie", "credential", "github_token", "password",
    "private_key", "registry_token", "secret", "token",
}
_public_protocols = {
    "__aenter__", "__aexit__", "__call__", "__dir__", "__enter__",
    "__exit__", "__getattr__", "__getitem__", "__init__", "__iter__",
    "__len__", "__next__",
}


def _node_parts(value):
    value = str(value or "").strip().replace("\\", "/")
    workspace = os.path.abspath(os.getcwd()).replace("\\", "/").rstrip("/")
    value = value.replace(workspace + "/", "/workspace/")
    path, separator, selector = value.partition("::")
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/"), selector if separator else ""


def _selected_nodeid(item):
    """Map pytest's nested-root node ID back to the authoritative plan ID."""
    raw_path, raw_selector = _node_parts(getattr(item, "nodeid", "") or "")
    raw = raw_path + (("::" + raw_selector) if raw_selector else "")
    if raw in _selected:
        return raw
    actual_path, actual_selector = _node_parts(raw)
    candidates = []
    for expected in _selected:
        expected_path, expected_selector = _node_parts(expected)
        if expected_selector != actual_selector:
            continue
        if (
            expected_path == actual_path
            or expected_path.endswith("/" + actual_path)
            or actual_path.endswith("/" + expected_path)
        ):
            candidates.append(expected)
    return candidates[0] if len(candidates) == 1 else raw


def _inside(module):
    return bool(module) and any(module == root or module.startswith(root + ".") for root in _roots)


def _clean_qualname(value):
    parts = [part for part in str(value or "").split(".") if part and part != "<locals>"]
    if any(part.startswith("<") for part in parts):
        return ""
    return ".".join(parts)


def _is_public_qualname(value):
    return not any(
        part.startswith("_") and part not in _public_protocols
        for part in str(value or "").split(".") if part
    )


def _add(node_id, module, qualname):
    if not node_id or not _inside(module):
        return ""
    clean = _clean_qualname(qualname)
    if not _is_public_qualname(clean):
        return ""
    qualified = ".".join(filter(None, (module, clean)))
    if not qualified or qualified.endswith(".<module>"):
        return ""
    with _lock:
        bucket = _symbols[node_id]
        if qualified not in bucket and len(bucket) >= _max_symbols:
            _truncated_nodes.add(node_id)
            return ""
        bucket[qualified] += 1
    return qualified


def _redacted(name):
    lowered = str(name or "").lower()
    return any(part in lowered for part in _redacted_names)


def _bounded_text(value, limit=None):
    limit = _max_value_chars if limit is None else max(8, int(limit))
    try:
        text = str(value)
    except BaseException:
        text = f"<{type(value).__module__}.{type(value).__qualname__}>"
    if len(text) > limit:
        return text[:limit] + f"...<truncated:{len(text) - limit}>"
    return text


def _safe_value(value, *, name="", depth=0, seen=None):
    """Serialize behavior evidence without invoking user descriptors."""
    if _redacted(name):
        return {"type": type(value).__name__, "redacted": True}
    if _trace_mode == "minimal":
        value_type = type(value)
        return {
            "type": f"{value_type.__module__}.{value_type.__qualname__}",
            "detail_mode": "type_only",
        }
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return {"type": type(value).__name__, "cycle": True}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _bounded_text(value)
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, bytes):
        return {
            "type": "bytes", "length": len(value),
            "preview_hex": value[: min(64, len(value))].hex(),
            "truncated": len(value) > 64,
        }
    value_type = type(value)
    type_name = f"{value_type.__module__}.{value_type.__qualname__}"
    if depth >= _max_depth:
        return {"type": type_name, "summary": _bounded_text(value, 200)}
    seen = set(seen)
    seen.add(identity)
    if isinstance(value, (list, tuple)):
        items = [
            _safe_value(item, depth=depth + 1, seen=seen)
            for item in value[:_max_collection_chars]
        ]
        return {
            "type": type_name, "length": len(value), "items": items,
            "truncated": len(value) > _max_collection_chars,
        }
    if isinstance(value, (set, frozenset)):
        try:
            values = sorted(value, key=lambda item: _bounded_text(item, 120))
        except BaseException:
            values = list(value)
        items = [
            _safe_value(item, depth=depth + 1, seen=seen)
            for item in values[:_max_collection_chars]
        ]
        return {
            "type": type_name, "length": len(value), "items": items,
            "truncated": len(value) > _max_collection_chars,
        }
    if isinstance(value, dict):
        items = []
        try:
            pairs = list(value.items())
        except BaseException:
            pairs = []
        for key, item in pairs[:_max_collection_chars]:
            key_text = _bounded_text(key, 200)
            items.append({
                "key": key_text,
                "value": _safe_value(
                    item, name=key_text, depth=depth + 1, seen=seen,
                ),
            })
        return {
            "type": type_name, "length": len(pairs), "items": items,
            "truncated": len(pairs) > _max_collection_chars,
        }
    try:
        namespace = vars(value)
    except BaseException:
        namespace = None
    public_state = []
    if isinstance(namespace, dict):
        for key in sorted(namespace, key=str)[:_max_collection_chars]:
            if str(key).startswith("_"):
                continue
            public_state.append({
                "name": str(key),
                "value": _safe_value(
                    namespace[key], name=str(key), depth=depth + 1, seen=seen,
                ),
            })
    result = {"type": type_name, "summary": _bounded_text(value, 300)}
    if public_state:
        result["public_state"] = public_state
    return result


def _call_arguments(frame, *, shape_only=False):
    code = frame.f_code
    positional = int(code.co_argcount) + int(code.co_kwonlyargcount)
    names = list(code.co_varnames[:positional])
    cursor = positional
    if code.co_flags & inspect.CO_VARARGS:
        names.append(code.co_varnames[cursor])
        cursor += 1
    if code.co_flags & inspect.CO_VARKEYWORDS:
        names.append(code.co_varnames[cursor])
    result = {}
    for name in names:
        if name not in frame.f_locals:
            continue
        value = _safe_value(frame.f_locals.get(name), name=name)
        result[name] = _compact_value(value) if shape_only else value
    return result


def _compact_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            "type": value.get("type", "object"),
            "length": value.get("length"),
            "summary": _bounded_text(value.get("summary", ""), 240),
            "truncated": True,
        }
    return {"type": type(value).__name__, "summary": _bounded_text(value, 240)}


def _shape_only_detail(qualname):
    # Pytest hook arguments contain very large Item/Session graphs.  Do not
    # discard the call: its parameter names, types, return shape, and ordering
    # are essential for plugin Specifications.  Bound only the value detail.
    name = str(qualname or "").rsplit(".", 1)[-1]
    return name.startswith(("pytest_", "pytest_runtest_"))


def _bound_event(event):
    try:
        size = len(json.dumps(event, sort_keys=True, default=str).encode())
    except BaseException:
        size = _max_event_bytes + 1
    if size > _max_event_bytes:
        event["arguments"] = {
            str(name): _compact_value(value)
            for name, value in (event.get("arguments") or {}).items()
        }
        if event.get("receiver_before") is not None:
            event["receiver_before"] = _compact_value(event["receiver_before"])
        if event.get("receiver_after") is not None:
            event["receiver_after"] = _compact_value(event["receiver_after"])
        if "return" in event:
            event["return"] = _compact_value(event["return"])
        event["event_payload_truncated"] = True


def _append_event(node_id, event):
    with _lock:
        bucket = _events[node_id]
        symbol = str(event.get("symbol") or "")
        if _event_symbol_counts[node_id][symbol] >= _max_events_per_symbol:
            # Repeated calls are represented by boundary_call_counts. Preserve
            # representative cases per symbol instead of the first calls only.
            _sampled_nodes.add(node_id)
            return -1
        if len(bucket) >= _max_events:
            _truncated_nodes.add(node_id)
            return -1
        _event_symbol_counts[node_id][symbol] += 1
        event["order"] = len(bucket)
        event["thread_id"] = threading.get_ident()
        _bound_event(event)
        bucket.append(event)
        return len(bucket) - 1


def _frame_qualname(frame):
    value = getattr(frame.f_code, "co_qualname", "")
    if value:
        return value
    # Python 3.8-3.10 do not expose code.co_qualname. Recover only the public
    # receiver type; no argument value is serialized or retained.
    receiver = frame.f_locals.get("self")
    receiver_type = type(receiver) if receiver is not None else None
    if receiver_type is None:
        candidate = frame.f_locals.get("cls")
        receiver_type = candidate if isinstance(candidate, type) else None
    if receiver_type is not None and str(getattr(receiver_type, "__module__", "")) == str(frame.f_globals.get("__name__") or ""):
        return f"{receiver_type.__qualname__}.{frame.f_code.co_name}"
    return frame.f_code.co_name


def _profile(frame, event, argument):
    if not _current:
        return _profile
    if event == "call":
        module = str(frame.f_globals.get("__name__") or "")
        caller = frame.f_back
        caller_module = str(caller.f_globals.get("__name__") or "") if caller else ""
        if _inside(module) and not _inside(caller_module):
            qualname = _frame_qualname(frame)
            qualified = _add(_current, module, qualname)
            if not qualified:
                return _profile
            shape_only = _shape_only_detail(qualname)
            receiver = frame.f_locals.get("self")
            index = _append_event(_current, {
                "event": "call", "symbol": qualified,
                "arguments": _call_arguments(frame, shape_only=shape_only),
                "receiver_before": (
                    _compact_value(_safe_value(receiver, name="self"))
                    if shape_only else _safe_value(receiver, name="self")
                ) if receiver is not None else None,
                "detail_mode": "shape_only" if shape_only else "bounded_value",
            })
            if index >= 0:
                _active_frames[id(frame)] = (
                    _current, index, receiver, shape_only,
                )
    elif event == "return":
        active = _active_frames.pop(id(frame), None)
        if active is not None:
            node_id, index, receiver, shape_only = active
            if index < len(_events[node_id]):
                returned = _safe_value(argument, name="return")
                _events[node_id][index]["return"] = (
                    _compact_value(returned) if shape_only else returned
                )
                if receiver is not None:
                    after = _safe_value(receiver, name="self")
                    _events[node_id][index]["receiver_after"] = (
                        _compact_value(after) if shape_only else after
                    )
                _bound_event(_events[node_id][index])
    elif event == "c_call":
        module = str(getattr(argument, "__module__", "") or "")
        caller_module = str(frame.f_globals.get("__name__") or "")
        if _inside(module) and not _inside(caller_module):
            qualified = _add(
                _current, module,
                getattr(argument, "__qualname__", getattr(argument, "__name__", "")),
            )
            if qualified:
                _detail_unavailable[_current][qualified] = "c_call_has_no_python_frame"
    return _profile


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    global _current
    selected_node = _selected_nodeid(item)
    if _trace_all or selected_node in _selected:
        _executed.add(selected_node)
        _current = selected_node
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            _parameters[selected_node] = {
                str(name): _safe_value(value, name=str(name))
                for name, value in dict(getattr(callspec, "params", {}) or {}).items()
            }
        sys.setprofile(_profile)
        # ``sys.setprofile`` only affects the current thread. Public API calls
        # made by thread pools would otherwise disappear from the sidecar.
        threading.setprofile(_profile)
    try:
        yield
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
        _current = ""


@pytest.hookimpl(hookwrapper=True)
def pytest_fixture_setup(fixturedef, request):
    outcome = yield
    node = getattr(request, "node", None)
    node_id = _selected_nodeid(node) if node is not None else ""
    if not node_id or (not _trace_all and node_id not in _selected):
        return
    try:
        value = outcome.get_result()
    except BaseException:
        return
    value_type = type(value)
    module = str(getattr(value_type, "__module__", "") or "")
    qualname = _clean_qualname(getattr(value_type, "__qualname__", ""))
    if _inside(module) and qualname and _is_public_qualname(qualname):
        _fixture_types[node_id].add(f"{module}.{qualname}")
    if len(_fixtures[node_id]) < _max_collection_chars:
        _fixtures[node_id].append({
            "name": str(getattr(fixturedef, "argname", "") or ""),
            "value": _safe_value(
                value, name=str(getattr(fixturedef, "argname", "") or "fixture")
            ),
        })


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    node_id = _selected_nodeid(item)
    if not node_id or (not _trace_all and node_id not in _selected):
        return
    row = _reports[node_id]
    row[str(report.when)] = {
        "outcome": str(report.outcome),
        "duration": float(getattr(report, "duration", 0.0) or 0.0),
    }
    if report.failed:
        row[str(report.when)]["longrepr"] = _bounded_text(
            getattr(report, "longreprtext", "") or getattr(report, "longrepr", ""),
            8000,
        )
    stdout = str(getattr(report, "capstdout", "") or "")
    stderr = str(getattr(report, "capstderr", "") or "")
    if stdout:
        row[str(report.when)]["stdout"] = _bounded_text(stdout, 4000)
    if stderr:
        row[str(report.when)]["stderr"] = _bounded_text(stderr, 4000)


def pytest_sessionfinish(session, exitstatus):
    prior = {}
    if os.path.isfile(_output):
        try:
            with open(_output, encoding="utf-8") as handle:
                prior = json.load(handle)
        except (OSError, ValueError):
            prior = {}
    prior_rows = {
        str(row.get("node_id") or ""): row
        for row in prior.get("nodes") or [] if row.get("node_id")
    }
    for node_id, row in prior_rows.items():
        _symbols[node_id].update({
            str(key): int(value)
            for key, value in (row.get("boundary_call_counts") or {}).items()
        })
        _fixture_types[node_id].update(row.get("fixture_public_types") or [])
        if row.get("truncated"):
            _truncated_nodes.add(node_id)
        if row.get("events_sampled"):
            _sampled_nodes.add(node_id)
    _executed.update(str(item) for item in prior.get("executed_node_ids") or [])
    rows = []
    for node_id in sorted(
        set(_executed) | set(_symbols) | set(_fixture_types) | set(_events)
    ):
        event_symbols = {
            str(event.get("symbol") or "") for event in _events[node_id]
            if event.get("symbol")
        }
        unavailable = {
            symbol: reason
            for symbol, reason in _detail_unavailable[node_id].items()
            if symbol not in event_symbols
        }
        rows.append({
            "node_id": node_id,
            "boundary_symbols": sorted(_symbols[node_id]),
            "boundary_call_counts": dict(sorted(_symbols[node_id].items())),
            "fixture_public_types": sorted(_fixture_types[node_id]),
            "fixture_values": _fixtures[node_id],
            "parameters": _parameters[node_id],
            "events": _events[node_id],
            "events_sampled": node_id in _sampled_nodes,
            "event_symbol_count": len(event_symbols),
            "detail_unavailable_symbols": unavailable,
            "reports": _reports[node_id],
            "truncated": node_id in _truncated_nodes,
        })
    boundary_node_ids = sorted(
        node_id for node_id in _executed
        if _symbols[node_id] or _fixture_types[node_id]
    )
    event_node_ids = sorted(
        node_id for node_id in _executed if _events[node_id]
    )
    payload = {
        "schema_version": 3,
        "status": "captured_complete" if (
            not _selected or len(_executed & _selected) == len(_selected)
        ) else "incomplete_execution",
        "identity": _identity,
        "scope": _scope,
        "import_roots": list(_roots),
        "root_discovery": _root_discovery,
        "selected_node_count": len(_selected),
        "executed_node_count": len(_executed),
        "executed_selected_node_count": len(_executed & _selected),
        "executed_node_ids": sorted(_executed),
        "row_count": len(rows),
        "observed_node_count": len(boundary_node_ids),
        "boundary_node_count": len(boundary_node_ids),
        "boundary_node_ids": boundary_node_ids,
        "event_node_count": len(event_node_ids),
        "event_node_ids": event_node_ids,
        "truncated_node_count": len(_truncated_nodes),
        "sampled_node_count": len(_sampled_nodes),
        "nodes": rows,
        "evidence_policy": (
            "private bounded construction evidence; public rendering requires "
            "subject attribution and semantic leak validation"
        ),
    }
    parent = os.path.dirname(_output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = _output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, _output)
'''


class ValidationError(RuntimeError):
    pass


def _buildspec_install_targets(buildspec: dict[str, Any]) -> list[str]:
    if not bool(buildspec.get("install_project", True)):
        return []
    values = [
        *(buildspec.get("local_projects") or []),
        str(buildspec.get("project_subdir") or "."),
    ]
    targets: list[str] = []
    for value in values:
        text = str(value or ".")
        if text != ".":
            text = _safe_relative(text).as_posix()
        if text not in targets:
            targets.append(text)
    return targets


def _resolve_h3_install_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Resolve the exact H3 install roots, including deferred Blob ledgers."""
    contract = dict(spec.get("h3_install_contract") or {})
    status = str(contract.get("status") or "")
    frozen_buildspec: dict[str, Any] | None = None
    if FROZEN_IMAGE_BUILDSPEC.is_file():
        try:
            candidate = json.loads(FROZEN_IMAGE_BUILDSPEC.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError("frozen image BuildSpec is unreadable") from exc
        if not isinstance(candidate, dict):
            raise ValidationError("frozen image BuildSpec is not a mapping")
        frozen_buildspec = candidate

    # The immutable scoring image is the primary executable record. Prefer its
    # content-bound BuildSpec whenever present; fall back to a historical Blob
    # checkpoint only for older images that predate the frozen artifact.
    if frozen_buildspec is not None:
        actual_digest = hashlib.sha256(json.dumps(
            frozen_buildspec, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        expected_digest = str(contract.get("buildspec_sha256") or "")
        # Some compacted H3 ledgers point at an earlier build attempt while
        # the published immutable image contains the BuildSpec from the
        # successful attempt.  The image digest is already verified before
        # this resolver runs and therefore content-binds the frozen file.
        # Treat the in-image artifact as executable truth and retain a
        # disagreement with the advisory ledger as provenance, not as a
        # validation failure.
        digest_disagreement = bool(
            expected_digest and actual_digest != expected_digest
        )
        contract.update({
            "status": "resolved_from_image",
            "install_targets": _buildspec_install_targets(frozen_buildspec),
            "resolved_image_buildspec_path": str(FROZEN_IMAGE_BUILDSPEC),
            "ledger_buildspec_sha256": expected_digest,
            "buildspec_digest_disagreement": digest_disagreement,
            "buildspec_sha256": actual_digest,
            "buildspec": frozen_buildspec,
        })
    elif status == "checkpoint":
        raw_path = str(contract.get("checkpoint_blob_path") or "")
        source_path = Path(raw_path)
        collapsed_parts: list[str] = []
        for part in source_path.parts:
            if not collapsed_parts or part != collapsed_parts[-1]:
                collapsed_parts.append(part)
        collapsed_path = Path(*collapsed_parts)
        relative_candidates = [source_path]
        if collapsed_path != source_path:
            # Some historical ledgers accidentally repeated the generation
            # directory (``.../family/family/shard...``). Exact lookup stays
            # authoritative; this fallback is identity-checked below against
            # both snapshot_id and BuildSpec SHA256.
            relative_candidates.append(collapsed_path)
        candidates = (
            relative_candidates if source_path.is_absolute() else [
                Path("/mnt/blob-qinglinzhu-shuwangmain-out") / path
                for path in relative_candidates
            ]
        )
        checkpoint_path = next((path for path in candidates if path.is_file()), None)
        if checkpoint_path is None:
            raise ValidationError(
                f"H3 install-contract checkpoint is unavailable: {raw_path}"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_snapshot = str(contract.get("source_h3_snapshot_id") or "")
        if str(checkpoint.get("snapshot_id") or "") != expected_snapshot:
            raise ValidationError("H3 install-contract checkpoint identity mismatch")
        buildspec = checkpoint.get("buildspec")
        if not isinstance(buildspec, dict):
            raise ValidationError("H3 install-contract checkpoint has no BuildSpec")
        expected_digest = str(contract.get("buildspec_sha256") or "")
        actual_digest = hashlib.sha256(json.dumps(
            buildspec, sort_keys=True, separators=(",", ":"), default=str,
        ).encode()).hexdigest()
        if expected_digest and actual_digest != expected_digest:
            raise ValidationError("H3 BuildSpec digest mismatch")
        contract.update({
            "status": "resolved_from_checkpoint",
            "install_targets": _buildspec_install_targets(buildspec),
            "resolved_checkpoint_path": str(checkpoint_path),
            # Installation, pytest and environment settings are one H3
            # runtime contract.  Keeping only install_targets silently lost
            # Django/doctest/plugin settings for deferred ledgers.
            "buildspec": buildspec,
        })
    targets = contract.get("install_targets")
    if not isinstance(targets, list):
        targets = list(spec.get("install_targets") or [])
    normalized: list[str] = []
    for value in targets:
        text = str(value or ".")
        if text != ".":
            text = _safe_relative(text).as_posix()
        if text not in normalized:
            normalized.append(text)
    contract["install_targets"] = normalized
    contract.setdefault("install_mode", "no_build_isolation")
    contract.setdefault("status", "legacy")
    return contract


_H3_FORBIDDEN_ENV_KEY = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|API_KEY|PRIVATE_KEY|"
    r"ACCESS_KEY|SESSION_KEY|AUTH|COOKIE)",
    re.IGNORECASE,
)
_H3_CREDENTIAL_URL = re.compile(r"^[a-z][a-z0-9+.-]*://[^/@\s]+:[^/@\s]+@", re.I)
_H3_FORBIDDEN_ENV_EXACT = frozenset({
    "HOME", "PATH", "PYTHONHOME", "VIRTUAL_ENV", "LD_PRELOAD",
    "GIT_CONFIG", "GIT_CONFIG_GLOBAL", "GIT_CONFIG_SYSTEM",
    "PIP_CONFIG_FILE", "PYTHONSTARTUP",
})


def _safe_h3_environment(
    buildspec: dict[str, Any], *, workspace: Path,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Restore deterministic H3 variables without importing credentials.

    H3 BuildSpecs are historical, model-produced inputs.  Values are never
    interpreted by a shell, but process-control and secret-shaped variables
    still fail closed.  Relative PYTHONPATH entries are anchored inside the
    mutable scorer workspace so they keep the same meaning after packaging.
    """
    accepted: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    raw_environment = buildspec.get("env") or {}
    if not isinstance(raw_environment, dict):
        return accepted, [{"key": "<env>", "reason": "not_a_mapping"}]
    for raw_key, raw_value in raw_environment.items():
        key = str(raw_key)
        value = str(raw_value)
        reason = ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            reason = "invalid_key"
        elif key in _H3_FORBIDDEN_ENV_EXACT or _H3_FORBIDDEN_ENV_KEY.search(key):
            reason = "security_sensitive_key"
        elif "\x00" in value or len(value) > 16_384:
            reason = "invalid_value"
        elif _H3_CREDENTIAL_URL.search(value):
            reason = "credential_bearing_value"
        if reason:
            rejected.append({"key": key, "reason": reason})
            continue
        if key == "PYTHONPATH":
            normalized: list[str] = []
            for item in value.split(os.pathsep):
                item = item.strip()
                if not item:
                    continue
                candidate = Path(item)
                if not candidate.is_absolute():
                    candidate = (workspace / _safe_relative(item)).resolve()
                else:
                    candidate = candidate.resolve()
                workspace_root = workspace.resolve()
                system_path = str(candidate).startswith((
                    "/usr/lib/", "/usr/local/lib/", "/opt/conda/lib/",
                ))
                if candidate != workspace_root and workspace_root not in candidate.parents \
                        and not system_path:
                    rejected.append({
                        "key": key, "reason": "pythonpath_outside_workspace",
                    })
                    continue
                text = str(candidate)
                if text not in normalized:
                    normalized.append(text)
            value = os.pathsep.join(normalized)
        accepted[key] = value
    return accepted, rejected


_PYTEST_DROP_WITH_VALUE = frozenset({
    "-k", "-m", "--ignore", "--ignore-glob", "--deselect", "--maxfail",
    "-n", "--numprocesses", "--dist", "--json-report-file", "--junitxml",
    "--cov-report", "--cov-config", "--cov-context",
})
_PYTEST_DROP_FLAGS = frozenset({
    "-x", "--exitfirst", "--lf", "--last-failed", "--ff", "--failed-first",
    "--sw", "--stepwise", "--stepwise-skip", "--json-report", "--cov",
    "--no-cov", "--pyargs",
})
_PYTEST_PRESENTATION_FLAGS = frozenset({
    "-q", "--quiet", "-v", "--verbose", "-vv", "-vvv", "-s",
    "--showlocals", "--disable-warnings", "--no-header", "--color=yes",
    "--color=no", "-ra", "-rs", "--tb=short", "--tb=long", "--tb=line",
    "--tb=no", "--durations=5", "--durations=10", "--durations=25",
})
_PYTEST_SEMANTIC_WITH_VALUE = frozenset({
    "-p", "-o", "-W", "-c", "--ds", "--rootdir", "--confcutdir",
    "--basetemp", "--import-mode", "--asyncio-mode", "--doctest-glob",
})


def _looks_like_pytest_selector(value: str) -> bool:
    return bool(
        "::" in value or "/" in value or "\\" in value
        or value in {".", "tests", "test", "src"}
        or value.endswith((".py", ".rst", ".ipynb"))
    )


def _h3_runtime_pytest_args(
    buildspec: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    """Keep H3 semantic pytest options while freezing the scored nodes.

    Selection, xdist, early-exit, coverage and reporting options are owned by
    the evaluator.  Plugin/framework/doctest options remain in their original
    order.  Positional roots are discarded because ``passed_ptp`` is the
    immutable denominator.
    """
    tokens = [str(value) for value in (buildspec.get("addopts") or [])]
    command = [str(value) for value in (buildspec.get("test_command") or [])]
    if command[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
        command = command[3:]
    elif command[:1] in (["pytest"], ["py.test"]):
        command = command[1:]
    tokens.extend(command)

    kept: list[str] = []
    dropped: list[dict[str, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        option, equals, _value = token.partition("=")
        if option in _PYTEST_DROP_WITH_VALUE:
            dropped.append({"argument": token, "reason": "evaluator_owned"})
            if not equals and index + 1 < len(tokens):
                index += 1
                dropped.append({
                    "argument": tokens[index], "reason": "option_value",
                })
        elif option in _PYTEST_DROP_FLAGS or token in _PYTEST_PRESENTATION_FLAGS:
            dropped.append({"argument": token, "reason": "evaluator_owned"})
        elif token.startswith(("--ignore=", "--ignore-glob=", "--deselect=", "--maxfail=")):
            dropped.append({"argument": token, "reason": "evaluator_owned"})
        elif token.startswith(("--cov=", "--cov-report=", "--json-report-file=", "--junitxml=")):
            dropped.append({"argument": token, "reason": "evaluator_owned"})
        elif token == "-o" and index + 1 < len(tokens) \
                and tokens[index + 1].startswith("addopts="):
            index += 1
            override = tokens[index]
            nested, nested_dropped = _h3_runtime_pytest_args({
                "addopts": shlex.split(override.split("=", 1)[1]),
            })
            for value in nested:
                if value not in kept:
                    kept.append(value)
            dropped.append({"argument": "-o", "reason": "flattened_addopts"})
            dropped.append({"argument": override, "reason": "flattened_addopts"})
            dropped.extend(nested_dropped)
        elif token.startswith("-"):
            if token not in kept:
                kept.append(token)
            # Preserve the operand of semantic/custom options.  Known
            # selection options were consumed above.
            if (
                not equals and index + 1 < len(tokens)
                and not tokens[index + 1].startswith("-")
                and (
                    option in _PYTEST_SEMANTIC_WITH_VALUE
                    or not _looks_like_pytest_selector(tokens[index + 1])
                )
            ):
                index += 1
                if tokens[index] not in kept:
                    kept.append(tokens[index])
        else:
            dropped.append({"argument": token, "reason": "positional_selector"})
        index += 1
    return kept, dropped


def _cleanup_roots(workspace: Path, install_targets: Iterable[str]) -> list[Path]:
    # Clean nested editable distributions first.  Cleaning the repository
    # root removes all Python sources recursively; running the legacy clean
    # script on a nested root afterwards can then exit 1 at its empty
    # ``find | grep`` pipeline before emitting its completion marker.
    roots: list[Path] = []
    for target in install_targets:
        if target in {"", "."}:
            continue
        candidate = workspace / _safe_relative(str(target))
        if candidate not in roots:
            roots.append(candidate)
    roots.append(workspace)
    return roots


def _load_collector_identity(path: Path | None) -> dict[str, Any]:
    """Verify extracted collector files against their pinned manifest."""
    if path is None:
        return {
            "collector_version": "image_baked_legacy",
            "collector_schema_version": 0,
            "collector_bundle_sha256": "",
            "collector_contract_sha256": "",
        }
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = dict(manifest.get("files") or {})
    for name, expected in files.items():
        candidate = path.parent / str(name)
        if not candidate.is_file():
            raise ValidationError(f"collector file missing: {name}")
        actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual != str((expected or {}).get("sha256") or ""):
            raise ValidationError(f"collector file digest mismatch: {name}")
    contract = {
        "collector_schema_version": manifest.get("collector_schema_version"),
        "collector_version": manifest.get("collector_version"),
        "files": files,
    }
    contract_sha = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if contract_sha != str(manifest.get("collector_contract_sha256") or ""):
        raise ValidationError("collector contract digest mismatch")
    return {
        "collector_version": str(manifest.get("collector_version") or ""),
        "collector_schema_version": int(
            manifest.get("collector_schema_version") or 0
        ),
        "collector_git_revision": str(
            manifest.get("collector_git_revision") or ""
        ),
        "collector_bundle_sha256": str(
            os.environ.get("TASKFORGE_COLLECTOR_BUNDLE_SHA256", "")
        ),
        "collector_contract_sha256": contract_sha,
    }


_PASSING_PYTEST_OUTCOMES = frozenset({
    "passed",
    # pytest-subtests reports the parent item this way when every subtest
    # passed. Treating it as an error parked otherwise healthy repositories.
    "subtests passed",
    # An XPASS executed successfully. It is not a reason to reject an
    # environment whose H3 stable ground truth already contains the node.
    "xpassed",
})


def normalize_pytest_outcome(value: Any) -> str:
    """Map pytest-json-report/plugin outcomes to TaskForge scoring states."""
    outcome = str(value or "error").strip().lower()
    if outcome in _PASSING_PYTEST_OUTCOMES:
        return "passed"
    if outcome in {"failed", "skipped", "xfailed", "error"}:
        return outcome
    return outcome or "error"


def merge_pytest_outcome(previous: str | None, current: str) -> str:
    """Combine duplicate records conservatively (for example subtests)."""
    if previous is None:
        return current
    priority = {
        "passed": 0,
        "skipped": 1,
        "xfailed": 1,
        "failed": 2,
        "error": 3,
    }
    return max((previous, current), key=lambda item: priority.get(item, 3))


def reconcile_pytest_nodeids(
    requested: list[str], observed: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Map pytest's rootdir-relative node IDs back to fixed ground truth.

    In monorepos pytest can choose a nested ``rootdir``.  A node requested as
    ``packages/api/tests/test_x.py::test_a`` is then reported by
    pytest-json-report as ``tests/test_x.py::test_a``.  Both identify the same
    executed item, but an exact string join manufactured a 0% pass rate even
    when pytest printed ``123 passed``.  Match only unique path-suffix plus
    exact selector pairs; ambiguous matches remain unclaimed and therefore
    fail closed.
    """
    normalized_observed: dict[str, str] = {}
    for node, outcome in observed.items():
        clean = str(node).strip()
        if not clean:
            continue
        normalized_observed[clean] = merge_pytest_outcome(
            normalized_observed.get(clean), outcome,
        )
    observed = normalized_observed
    requested_set = set(requested)
    matched = {
        node: outcome for node, outcome in observed.items() if node in requested_set
    }
    remapped: dict[str, str] = {}
    unmatched_requested = [node for node in requested if node not in matched]
    unmatched_observed = {
        node: outcome for node, outcome in observed.items() if node not in requested_set
    }

    def parts(node: str) -> tuple[str, str]:
        path, separator, selector = node.replace("\\", "/").partition("::")
        while path.startswith("./"):
            path = path[2:]
        return path.rstrip("/"), selector if separator else ""

    candidate_map: dict[str, list[str]] = {}
    reverse_candidates: dict[str, list[str]] = {}
    for expected in unmatched_requested:
        expected_path, expected_selector = parts(expected)
        candidates: list[str] = []
        for actual in unmatched_observed:
            actual_path, actual_selector = parts(actual)
            if expected_selector != actual_selector:
                continue
            if (
                expected_path == actual_path
                or expected_path.endswith("/" + actual_path)
                or actual_path.endswith("/" + expected_path)
            ):
                candidates.append(actual)
                reverse_candidates.setdefault(actual, []).append(expected)
        candidate_map[expected] = candidates
    for expected in unmatched_requested:
        candidates = candidate_map[expected]
        if len(candidates) != 1 or len(reverse_candidates[candidates[0]]) != 1:
            continue
        actual = candidates[0]
        outcome = unmatched_observed[actual]
        matched[expected] = outcome
        remapped[actual] = expected
    return matched, remapped


def remap_workspace_paths_in_nodeid(node_id: str, workspace: Path) -> str:
    """Rewrite absolute ``/workspace`` parameters for a pristine copy.

    Pytest includes ``Path`` parameter values in parametrized node IDs. The
    final image records those values under ``/workspace``; behavior capture
    deliberately executes a copied tree under ``/tmp``. Passing the old
    selector makes pytest exit 4 before running any item in the shard.
    """
    raw_path, separator, selector = str(node_id).partition("::")
    value = raw_path.replace("\\", "/")
    target = workspace.resolve().as_posix().rstrip("/")
    path = value.replace("/workspace/", target + "/")
    selector = selector.replace("/workspace/", target + "/")
    return path + (("::" + selector) if separator else "")


def pytest_selection_nodeid(node_id: str) -> str:
    """Return pytest's recorded node ID in its CLI-selection spelling.

    Pytest deliberately renders non-ASCII and control characters in parameter
    IDs as ``\\uXXXX``/``\\xXX`` sequences, and its CLI selector expects that
    exact escaped spelling.  Decoding it is incorrect and can even create an
    embedded NUL that cannot be passed in ``argv``.  Path remapping is handled
    separately and must never normalize backslashes in the selector suffix.
    """
    return str(node_id)


def canonicalize_workspace_paths_in_nodeid(
    node_id: str, workspace: Path,
) -> str:
    """Map a pristine-copy node ID back to its fixed ``/workspace`` form."""
    raw_path, separator, selector = str(node_id).partition("::")
    value = raw_path.replace("\\", "/")
    target = workspace.resolve().as_posix().rstrip("/")
    path = value.replace(target + "/", "/workspace/")
    selector = selector.replace(target + "/", "/workspace/")
    return path + (("::" + selector) if separator else "")


_PYTEST_NOT_FOUND_RE = re.compile(
    r"(?m)^ERROR:\s+not found:\s+(.+?)\s*$",
)


def pytest_not_found_nodeids(
    output: str, requested: list[str], *, workspace: Path,
) -> list[str]:
    """Return exact requested selectors that pytest explicitly rejected.

    A single stale parametrized node ID makes pytest exit 4 before executing
    *any* of the other selectors in the same invocation. Only consume the
    unambiguous ``ERROR: not found`` diagnostics and only when they map back
    to an exact requested node. Import/collection errors therefore remain
    fail-closed instead of being mistaken for stale ground truth.
    """
    workspace_text = workspace.resolve().as_posix().rstrip("/")

    def normalize(value: str) -> str:
        value = str(value).strip().strip("\"'").replace("\\", "/")
        path, separator, selector = value.partition("::")
        if path == workspace_text:
            path = ""
        elif path.startswith(workspace_text + "/"):
            path = path[len(workspace_text) + 1:]
        while path.startswith("./"):
            path = path[2:]
        return path + (("::" + selector) if separator else "")

    by_normalized = {normalize(node): node for node in requested}
    rejected: set[str] = set()
    for match in _PYTEST_NOT_FOUND_RE.finditer(output or ""):
        node = by_normalized.get(normalize(match.group(1)))
        if node is not None:
            rejected.add(node)
    return [node for node in requested if node in rejected]


def pytest_diagnostic(
    *, payload: dict[str, Any], process: subprocess.CompletedProcess[str],
    nodes: list[str], observed: dict[str, str], chunk_index: int,
    raw_observed: dict[str, str] | None = None,
    remapped: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Retain bounded evidence for collection/execution repair decisions.

    The old validator reduced every unreported node to ``missing`` and threw
    away the pytest output and collector payload. That made an environment
    failure indistinguishable from thousands of real test failures.
    """
    expected = set(nodes)
    missing = sorted(expected - set(observed))
    collectors: list[dict[str, Any]] = []
    for row in payload.get("collectors") or []:
        if not isinstance(row, dict):
            continue
        outcome = str(row.get("outcome") or "")
        if outcome.lower() in {"passed", ""}:
            continue
        rendered = json.dumps(row, sort_keys=True, default=str)
        collectors.append({
            "nodeid": str(row.get("nodeid") or ""),
            "outcome": outcome,
            "detail": rendered[-4000:],
        })
        if len(collectors) >= 8:
            break
    return {
        "chunk_index": chunk_index,
        "requested_count": len(nodes),
        "observed_count": len(expected & set(observed)),
        "raw_observed_count": len(raw_observed or observed),
        "raw_observed_sample": sorted(raw_observed or observed)[:20],
        "rootdir_remapped_count": len(remapped or {}),
        "rootdir_remapped_sample": [
            {"reported": actual, "expected": expected_node}
            for actual, expected_node in sorted((remapped or {}).items())[:20]
        ],
        "missing_count": len(missing),
        "missing_sample": missing[:20],
        "returncode": process.returncode,
        "summary": payload.get("summary") or {},
        "collector_errors": collectors,
        "output_tail": (process.stdout + "\n" + process.stderr)[-12000:],
    }


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or "\0" in value:
        raise ValidationError(f"unsafe relative path: {value!r}")
    return path


def patch_new_files(test_patch: str) -> list[str]:
    """Return paths introduced by a git patch, including unusual filenames.

    The old ``rsplit(' b/', 1)`` parser corrupted unquoted paths containing
    `` b/`` and skipped Git-quoted paths. Prefer the authoritative ``+++``
    path once available, while retaining the diff header for empty new files.
    """
    result: list[str] = []
    current: str | None = None
    is_new = False

    def diff_path(header: str) -> str | None:
        if header.startswith("diff --git ") and '"' not in header:
            parts = header.rsplit(" b/", 1)
            if len(parts) == 2:
                return parts[1]
        try:
            parts = shlex.split(header)
        except ValueError:
            return None
        if len(parts) != 4 or parts[:2] != ["diff", "--git"]:
            return None
        value = parts[3]
        return value[2:] if value.startswith("b/") else None

    def finish() -> None:
        if current and is_new:
            result.append(current)

    for line in test_patch.splitlines():
        if line.startswith("diff --git "):
            finish()
            current = diff_path(line)
            is_new = False
        elif line.startswith("new file mode "):
            is_new = True
        elif line.startswith("+++ ") and is_new:
            value = line[4:].strip()
            if value.startswith('"'):
                try:
                    value = shlex.split(value)[0]
                except (ValueError, IndexError):
                    pass
            if value.startswith("b/"):
                current = value[2:]
    finish()
    for value in result:
        _safe_relative(value)
    return result


def safe_extract_binary(archive: Path, destination: Path, expected: Iterable[str]) -> None:
    expected_set = set(expected)
    with tarfile.open(archive, "r:gz") as handle:
        names: set[str] = set()
        for member in handle.getmembers():
            _safe_relative(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise ValidationError(f"binary archive contains a non-file: {member.name}")
            names.add(member.name)
        if names != expected_set:
            raise ValidationError("binary archive member list does not match draft")
        handle.extractall(destination, filter="data")


def _copy_gold(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name in SKIP_COPY or name.endswith((".pyc", ".pyo"))}

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def _remove_path(path: Path) -> None:
    """Remove one payload path without following a final symlink."""
    if path.is_dir() and not path.is_symlink():
        # Some repositories contain nested Git metadata which can be touched
        # briefly by an install hook while the gold payload is separated.  A
        # single rmtree can then lose a race and report ENOTEMPTY even though
        # every target is still below the already validated payload root.
        # Retry the exact path only; never widen the cleanup target.
        for attempt in range(4):
            try:
                shutil.rmtree(path)
                return
            except FileNotFoundError:
                return
            except OSError as exc:
                if exc.errno != errno.ENOTEMPTY or attempt == 3:
                    raise
                time.sleep(0.05 * (attempt + 1))
    else:
        path.unlink(missing_ok=True)


def _prepare_gold_implementation(
    source: Path,
    destination: Path,
    *,
    declared_test_files: Iterable[str],
    patch_targets: Iterable[str],
    binary_targets: Iterable[str],
) -> dict[str, Any]:
    """Copy the parent checkout as an implementation-only gold payload.

    Gold substitutes for the solver's output, so it must not carry any file
    that the hidden-test bundle later materializes.  ``test_files`` is the
    public, authoritative bundle inventory; the text patch and binary archive
    must partition it exactly.  Failing closed here catches both accidental
    gold-test leakage and publication damage before pytest can manufacture a
    reassuring but invalid score.

    Other, unscored repository tests are not an authority for this task and
    are harmless because pytest receives the frozen node denominator.  What
    matters for scorer equivalence is that every scored test/config/fixture
    byte comes only from ``test_patch`` or ``binfix``.
    """
    declared = [_safe_relative(str(value)).as_posix() for value in declared_test_files]
    text = [_safe_relative(str(value)).as_posix() for value in patch_targets]
    binary = [_safe_relative(str(value)).as_posix() for value in binary_targets]
    for name, values in (
        ("test_files", declared), ("patch targets", text),
        ("binary targets", binary),
    ):
        if len(values) != len(set(values)):
            raise ValidationError(f"{name} contain duplicate paths")
    overlap = sorted(set(text) & set(binary))
    if overlap:
        raise ValidationError(
            "text patch and binary archive overlap: " + ", ".join(overlap[:20])
        )
    materialized = set(text) | set(binary)
    if set(declared) != materialized:
        missing = sorted(set(declared) - materialized)
        undeclared = sorted(materialized - set(declared))
        raise ValidationError(
            "test material inventory mismatch: "
            f"missing_from_payload={missing[:20]!r} "
            f"undeclared_payload={undeclared[:20]!r}"
        )

    _copy_gold(source, destination)
    copied_before = sum(1 for path in destination.rglob("*") if path.is_file())
    present_before = 0
    for relative in declared:
        target = destination / _safe_relative(relative)
        if target.exists() or target.is_symlink():
            present_before += 1
        _remove_path(target)
    residual = [
        relative for relative in declared
        if (destination / _safe_relative(relative)).exists()
        or (destination / _safe_relative(relative)).is_symlink()
    ]
    if residual:
        raise ValidationError(
            "gold implementation still contains declared test material: "
            + ", ".join(residual[:20])
        )
    copied_after = sum(1 for path in destination.rglob("*") if path.is_file())
    path_identity = hashlib.sha256(
        ("\n".join(sorted(
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*") if path.is_file()
        )) + "\n").encode()
    ).hexdigest()
    return {
        "policy": "implementation_only_v1",
        "authoritative_test_source": "test_patch_plus_binary_fixtures",
        "declared_test_file_count": len(declared),
        "text_patch_file_count": len(text),
        "binary_fixture_file_count": len(binary),
        "declared_files_present_in_parent_checkout": present_before,
        "gold_files_before_exclusion": copied_before,
        "gold_implementation_file_count": copied_after,
        "gold_implementation_paths_sha256": path_identity,
        "residual_declared_test_file_count": 0,
        "status": "separated",
    }


def _copy_runtime_pristine(source: Path, destination: Path) -> None:
    """Snapshot the exact pre-test tree for the instrumented comparison pass."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        ["cp", "-a", "--reflink=auto", f"{source}/.", str(destination)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if process.returncode:
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source, destination, symlinks=True)


def _overlay(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to(os.readlink(path))
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def _run(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv, cwd=cwd, env=env, text=True, encoding="utf-8",
            errors="replace", capture_output=True,
            timeout=max(1, timeout), check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValidationError(f"timeout after {timeout}s: {shlex.join(argv[:5])}") from exc


_INVALID_UNTAGGED_GIT_VERSION = re.compile(
    r"Invalid version:\s*['\"]unknown\+g([0-9a-f]+)['\"]",
    re.IGNORECASE,
)


def _invalid_untagged_git_version_sha(
    attempts: Iterable[dict[str, Any]],
) -> str:
    for attempt in attempts:
        match = _INVALID_UNTAGGED_GIT_VERSION.search(
            str(attempt.get("output_tail") or "")
        )
        if match:
            return match.group(1).lower()
    return ""


def _needs_git_version_anchor(attempts: Iterable[dict[str, Any]]) -> bool:
    """Recognize the narrow legacy-version failure repaired by a local tag.

    Some historical ``miniver``-style backends turn an untagged checkout into
    ``unknown+g<sha>``, which modern packaging correctly rejects. Do not use
    the repair for arbitrary editable-install failures: it is intentionally
    gated on the exact invalid-version diagnostic.
    """
    return bool(_invalid_untagged_git_version_sha(attempts))


def _ensure_git_version_anchor(
    workspace: Path, *, parent_commit: str, env: dict[str, str],
    attempts: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Create an auditable PEP 440 tag without changing checkout contents.

    The tag exists only inside the disposable validation container. Its name
    is deterministic for the verified checkout (the declared commit or the
    trusted clean root), and every identity check is recorded so promotion can
    distinguish this repair from source mutation.
    """
    expected = parent_commit.lower()
    # Keep the synthetic release free of a PEP 440 local ``+`` component.
    # Legacy backends may append ``+dirty`` after TaskForge overlays the test
    # patch; a tag that already contained ``+`` would then become invalid.
    # Every task has an isolated Git database, so this fixed local tag cannot
    # collide across snapshots; the audited commit provides its identity.
    tag = "0.0.0"
    head = _require_success(
        ["git", "rev-parse", "HEAD"], cwd=workspace, env=env, timeout=30,
        phase="version anchor HEAD verification",
    ).stdout.strip().lower()
    trigger_sha = _invalid_untagged_git_version_sha(attempts)
    if not trigger_sha or not head.startswith(trigger_sha):
        raise ValidationError(
            "cannot anchor version: invalid-version SHA does not match current HEAD"
        )

    head_kind = "declared_parent"
    if head != expected:
        # ``taskforge_clean.sh`` deliberately destroys the source-bearing Git
        # history and creates one auditable root commit before the gold overlay
        # is restored.  Legacy miniver-style backends inspect that synthetic
        # commit, not the original parent.  Accept precisely that trusted shape
        # and nothing broader: a root commit with the fixed clean identity and
        # message whose SHA is also named by the packaging error.
        metadata = _require_success(
            ["git", "show", "-s", "--format=%P%x00%s%x00%an%x00%ae", "HEAD"],
            cwd=workspace, env=env, timeout=30,
            phase="synthetic clean commit verification",
        ).stdout.rstrip("\n").split("\x00")
        if metadata != [
            "", "cleaned workspace", "DeNovoSWE-Clean", "clean@denovoswe.local",
        ]:
            raise ValidationError(
                f"cannot anchor version: HEAD {head} is not the trusted clean root"
            )
        head_kind = "trusted_clean_root"

    tree_before = _require_success(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=workspace, env=env, timeout=30,
        phase="version anchor tree verification",
    ).stdout.strip().lower()
    status_before = _require_success(
        ["git", "status", "--porcelain=v1", "-z"], cwd=workspace, env=env,
        timeout=30, phase="version anchor worktree verification",
    ).stdout

    reference = f"refs/tags/{tag}^{{commit}}"
    tag_reference = f"refs/tags/{tag}"
    existing = _run(
        ["git", "rev-parse", "-q", "--verify", reference],
        cwd=workspace, env=env, timeout=30,
    )
    if existing.returncode == 0:
        anchored_commit = existing.stdout.strip().lower()
        if anchored_commit != head:
            raise ValidationError(
                f"version anchor {tag} points to {anchored_commit}, expected {head}"
            )
        tag_kind = _require_success(
            ["git", "cat-file", "-t", tag_reference], cwd=workspace, env=env,
            timeout=30, phase="version anchor type verification",
        ).stdout.strip()
        if tag_kind != "tag":
            raise ValidationError("existing version anchor is not an annotated tag")
        action = "reused"
    else:
        tag_env = dict(env)
        tag_env.update({
            "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
        })
        _require_success(
            [
                "git", "-c", "user.name=TaskForge-Version-Anchor", "-c",
                "user.email=version-anchor@taskforge.invalid", "tag", "-a", tag,
                head, "-m", "TaskForge validation version anchor",
            ], cwd=workspace, env=tag_env,
            timeout=30, phase="version anchor creation",
        )
        anchored_commit = _require_success(
            ["git", "rev-parse", "--verify", reference],
            cwd=workspace, env=env, timeout=30,
            phase="version anchor verification",
        ).stdout.strip().lower()
        if anchored_commit != head:
            raise ValidationError(
                f"created version anchor {tag} points to {anchored_commit}, expected {head}"
            )
        tag_kind = _require_success(
            ["git", "cat-file", "-t", tag_reference], cwd=workspace, env=env,
            timeout=30, phase="version anchor type verification",
        ).stdout.strip()
        if tag_kind != "tag":
            raise ValidationError("created version anchor is not an annotated tag")
        action = "created"

    described = _require_success(
        ["git", "describe", "--long", "--always", "--first-parent"],
        cwd=workspace, env=env, timeout=30, phase="version anchor describe verification",
    ).stdout.strip()
    if not (described == f"{tag}-0-g{head[:7]}" or described.startswith(f"{tag}-0-g")):
        raise ValidationError(f"version anchor is not visible to git describe: {described}")
    tag_object = _require_success(
        ["git", "rev-parse", "--verify", tag_reference], cwd=workspace, env=env,
        timeout=30, phase="version anchor object verification",
    ).stdout.strip().lower()

    head_after = _require_success(
        ["git", "rev-parse", "HEAD"], cwd=workspace, env=env, timeout=30,
        phase="version anchor post-check",
    ).stdout.strip().lower()
    if head_after != head:
        raise ValidationError("version anchor creation changed HEAD")
    tree_after = _require_success(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=workspace, env=env, timeout=30,
        phase="version anchor tree post-check",
    ).stdout.strip().lower()
    status_after = _require_success(
        ["git", "status", "--porcelain=v1", "-z"], cwd=workspace, env=env,
        timeout=30, phase="version anchor worktree post-check",
    ).stdout
    if tree_after != tree_before or status_after != status_before:
        raise ValidationError("version anchor creation changed source tree or worktree")
    return {
        "status": action,
        "trigger": "invalid_pep440_unknown_git_version",
        "tag": tag,
        "tag_kind": tag_kind,
        "tag_object": tag_object,
        "git_describe": described,
        "commit": anchored_commit,
        "declared_parent_commit": expected,
        "head_kind": head_kind,
        "trigger_git_sha": trigger_sha,
        "head_after": head_after,
        "head_tree": tree_after,
        "source_tree_mutated": "false",
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _node_ids_sha256(nodes: Iterable[str]) -> str:
    payload = "".join(f"{node}\n" for node in sorted(set(nodes))).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_success(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: int, phase: str,
) -> subprocess.CompletedProcess[str]:
    result = _run(argv, cwd=cwd, env=env, timeout=timeout)
    if result.returncode:
        tail = (result.stdout + "\n" + result.stderr)[-8000:]
        raise ValidationError(f"{phase} failed with exit {result.returncode}: {tail}")
    return result


_BENIGN_CLEAN_SUMMARY_LINE = re.compile(
    r"(?:Files|Directories) removed:\s*0(?:\s+\(0\s+bytes\))?\s*"
)
_PIP_UNINSTALL_FOUND_LINE = re.compile(r"Found existing installation:\s*\S.*")
_PIP_UNINSTALL_START_LINE = re.compile(r"Uninstalling\s+\S.*:\s*")
_PIP_UNINSTALL_SUCCESS_LINE = re.compile(r"Successfully uninstalled\s+\S.*")


def _is_benign_clean_empty_exit(stdout: str, stderr: str) -> bool:
    """Recognize only the cleaner's legacy empty-pipeline exit contract.

    Some scoring images first uninstall one or more editable distributions and
    then execute three removal pipelines.  GNU ``xargs``/``grep`` variants in
    those legacy cleaners can return 1 when every pipeline is empty, even
    though cleanup succeeded.  Pip's successful-uninstall preamble is benign;
    arbitrary warnings, ``find`` errors, non-zero removal counts, incomplete
    uninstall blocks, and all stderr remain fatal.
    """
    if stderr.strip():
        return False
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return False

    index = 0
    while index < len(lines) and _PIP_UNINSTALL_FOUND_LINE.fullmatch(lines[index]):
        if index + 2 >= len(lines):
            return False
        if not _PIP_UNINSTALL_START_LINE.fullmatch(lines[index + 1]):
            return False
        if not _PIP_UNINSTALL_SUCCESS_LINE.fullmatch(lines[index + 2]):
            return False
        index += 3

    summaries = lines[index:]
    return bool(
        summaries
        and any(line.startswith("Files removed:") for line in summaries)
        and all(_BENIGN_CLEAN_SUMMARY_LINE.fullmatch(line) for line in summaries)
    )


def _run_clean_script(
    clean_script: Path, cleanup_root: Path, *, workspace: Path,
    env: dict[str, str], timeout: int = 600,
) -> dict[str, Any]:
    """Run the cleaner, tolerating only its known empty-pipeline exit."""
    if not cleanup_root.exists() and not cleanup_root.is_symlink():
        # Nested install roots may already have been removed by an outer clean.
        # Cleaning is intentionally idempotent; absence is the desired state.
        return {
            "root": str(cleanup_root),
            "exit_code": 0,
            "root_missing_tolerated": True,
            "legacy_empty_pipeline_tolerated": False,
            "stdout_tail": "",
        }
    process = _run(
        ["bash", str(clean_script), str(cleanup_root)], cwd=workspace,
        env=env, timeout=timeout,
    )
    empty_pipeline_tolerated = bool(
        process.returncode == 1
        and _is_benign_clean_empty_exit(process.stdout, process.stderr)
    )
    # Some legacy cleaners remove a source/test directory and then invoke more
    # ``find`` pipelines beneath the deleted path.  Those calls report ENOENT
    # and return 1 even though the requested post-condition has been reached.
    # Accept only that exact, root-bound failure shape.  Arbitrary stderr,
    # paths outside the cleanup root, and non-benign stdout remain fatal.
    stderr_lines = [
        line.strip() for line in process.stderr.splitlines() if line.strip()
    ]
    missing_paths_tolerated = bool(
        process.returncode == 1
        and stderr_lines
        and all(
            line.startswith("find: ")
            and str(cleanup_root) in line
            and line.endswith("No such file or directory")
            for line in stderr_lines
        )
        and _is_benign_clean_empty_exit(process.stdout, "")
    )
    tolerated = empty_pipeline_tolerated or missing_paths_tolerated
    if process.returncode and not tolerated:
        tail = (process.stdout + "\n" + process.stderr)[-8000:]
        raise ValidationError(
            f"TaskForge clean {cleanup_root} failed with exit "
            f"{process.returncode}: {tail}"
        )
    return {
        "root": str(cleanup_root),
        "exit_code": process.returncode,
        "legacy_empty_pipeline_tolerated": empty_pipeline_tolerated,
        "missing_paths_during_clean_tolerated": missing_paths_tolerated,
        "root_removed_during_clean_tolerated": bool(
            missing_paths_tolerated
            and not cleanup_root.exists()
            and not cleanup_root.is_symlink()
        ),
        "stdout_tail": process.stdout[-1000:],
    }


def _chunks(
    nodes: list[str], size: int = 200, max_argument_chars: int = 64_000,
) -> Iterable[list[str]]:
    grouped: dict[str, list[str]] = {}
    for node in nodes:
        grouped.setdefault(node.split("::", 1)[0], []).append(node)
    for path in sorted(grouped):
        values = grouped[path]
        chunk: list[str] = []
        characters = 0
        for node in values:
            if chunk and (len(chunk) >= size or characters + len(node) + 1 > max_argument_chars):
                yield chunk
                chunk = []
                characters = 0
            chunk.append(node)
            characters += len(node) + 1
        if chunk:
            yield chunk


def _workspace_pythonpath(workspace: Path, previous: str = "") -> str:
    """Prefer the mutable checkout over any package baked into the H3 image."""
    values = [str(workspace), str(workspace / "src")]
    if previous:
        values.append(previous)
    return os.pathsep.join(values)


def _published_pytest_pythonpath(
    workspace: Path, plugin_root: Path, previous: str = "",
) -> str:
    """Mirror ``python -m pytest`` from the scorer workspace.

    The JSON argv runner lives under ``/tmp``, so Python would otherwise omit
    its process working directory from ``sys.path``.  Add only the workspace
    root that the formal scorer naturally exposes; do not add ``workspace/src``
    as an import escape hatch for a broken editable installation.
    """
    values = [str(plugin_root), str(workspace)]
    if previous:
        values.append(previous)
    return os.pathsep.join(values)


def _install_environment(workspace: Path, env: dict[str, str]) -> dict[str, str]:
    """Keep project modules from shadowing pip and Python's standard library."""
    result = env.copy()
    # Some historical task images export TMPDIR/TMP/TEMP as /workspace.  PEP
    # 517 then creates a temporary metadata directory beside the project and
    # setuptools mistakes it for a second top-level package.  The public
    # evaluator must keep its own build scratch outside the submitted tree.
    result.update({"TMPDIR": "/tmp", "TMP": "/tmp", "TEMP": "/tmp"})
    blocked = {str(workspace.resolve()), str((workspace / "src").resolve())}
    result["PYTHONPATH"] = os.pathsep.join(
        value for value in result.get("PYTHONPATH", "").split(os.pathsep)
        if value and str(Path(value).resolve()) not in blocked
    )
    return result


_SOURCE_METADATA_IMPORT_FAILURE = re.compile(
    r"(?:setup\.py|build_meta\.py|metadata).*?"
    r"ModuleNotFoundError:\s*No module named",
    re.IGNORECASE | re.DOTALL,
)
_SETUPTOOLS_SCM_METADATA_FAILURE = re.compile(
    r"(?:setuptools_scm|unable to detect version|"
    r"canonicalize_version\(\).*strip_trailing_zero|"
    r"canonicalize_version\(\) got an unexpected keyword argument)",
    re.IGNORECASE | re.DOTALL,
)


def _install_failure_output(attempts: Iterable[dict[str, Any]]) -> str:
    return "\n".join(str(attempt.get("output_tail") or "") for attempt in attempts)


def _needs_source_metadata_path(attempts: Iterable[dict[str, Any]]) -> bool:
    """Whether project metadata imports its own not-yet-installed package."""
    return bool(_SOURCE_METADATA_IMPORT_FAILURE.search(_install_failure_output(attempts)))


def _needs_setuptools_scm_override(attempts: Iterable[dict[str, Any]]) -> bool:
    """Recognize metadata failures caused by the scorer's intentionally fresh Git repo."""
    return bool(_SETUPTOOLS_SCM_METADATA_FAILURE.search(_install_failure_output(attempts)))


def capture_runtime_environment(spec: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Capture the actual outer-container environment without local paths or secrets."""
    distributions = sorted({
        (
            str(distribution.metadata.get("Name") or distribution.name).lower().replace("_", "-"),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    })
    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os_release[key] = value.strip().strip('"')
    except OSError:
        pass
    value: dict[str, Any] = {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable_name": Path(sys.executable).name,
        },
        "platform": {
            "machine": platform.machine(),
            "system": platform.system(),
            "release": platform.release(),
            "libc": list(platform.libc_ver()),
        },
        "os_release": os_release,
        "installed_distributions": [
            {"name": name, "version": version} for name, version in distributions
        ],
        "expected_environment_sha256": str(spec.get("expected_environment_sha256") or ""),
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return value, hashlib.sha256(payload).hexdigest()


def _native_surface_declared(workspace: Path) -> bool:
    """Detect native build intent without mistaking vendored C for an API."""
    native_suffixes = {".pyx", ".pxd", ".so", ".pyd", ".rs"}
    declarations = {
        "setup.py", "setup.cfg", "pyproject.toml", "CMakeLists.txt", "meson.build",
    }
    pattern = re.compile(
        r"Extension\s*\(|ext_modules|cffi_modules|\[tool\.maturin\]|"
        r"\[tool\.scikit-build\]|scikit[_-]build|pybind11|cythonize\s*\(|"
        r"maturin|setuptools-rust|cmake",
        re.I,
    )
    for path in workspace.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        if path.name == "Cargo.toml" or path.suffix.lower() in native_suffixes:
            return True
        if path.name in declarations:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")[:1_000_000]
            except OSError:
                continue
            if pattern.search(text):
                return True
    return False


def capture_runtime_api_inventory(
    draft: dict[str, Any], *, workspace: Path, env: dict[str, str], timeout: int,
    script: Path = Path("/opt/taskforge_runtime_surface.py"),
) -> dict[str, Any]:
    """Reflect native Python APIs in an isolated, bounded subprocess.

    Reflection is Specification evidence, never a reason to invalidate an
    otherwise healthy task environment. Failures remain explicit in the
    validation record while the normal test validation continues.
    """
    import_names = sorted({
        str(name).replace("-", "_").split(".", 1)[0]
        for name in draft.get("import_names") or [] if name
    })
    empty = {
        "schema_version": 1, "import_names": import_names, "symbols": [],
        "symbol_count": 0, "import_errors": [],
    }
    source_declared = _native_surface_declared(workspace)
    empty["source_declared"] = source_declared
    if not script.is_file():
        return {**empty, "status": "unavailable", "reason": "runtime_surface_tool_missing"}
    with tempfile.TemporaryDirectory(prefix="taskforge-runtime-surface-") as temp:
        manifest = Path(temp) / "task.json"
        destination = Path(temp) / "runtime_api_inventory.json"
        # Root discovery needs more than the legacy ``import_names`` list.  In
        # particular, distribution metadata, nested package setup files,
        # stable/scoring-test imports and the repository identity rescue
        # projects whose PyPI and import names differ.  Pass the exact task
        # manifest used by the already-running validation container.
        manifest.write_text(
            json.dumps(draft, sort_keys=True, default=str) + "\n", encoding="utf-8",
        )
        argv = [
            sys.executable, str(script),
            "--workspace", str(workspace),
            "--import-names-file", str(manifest),
            "--snapshot-id", str(draft.get("snapshot_id") or ""),
            "--output", str(destination),
        ]
        try:
            process = _run(
                argv, cwd=workspace, env=env,
                # The reflector enforces its own 120-second wall budget.  Give
                # the subprocess enough grace to serialize a large inventory.
                timeout=min(max(30, timeout), 180),
            )
        except ValidationError as exc:
            return {**empty, "status": "unavailable", "reason": str(exc)}
        if not destination.is_file():
            return {
                **empty, "status": "unavailable",
                "reason": f"runtime_surface_exit_{process.returncode}",
            }
        try:
            value = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                **empty, "status": "unavailable",
                "reason": f"runtime_surface_result_invalid: {exc}",
            }
        if process.returncode != 0:
            value["status"] = "incomplete"
        else:
            value.setdefault("status", "captured")
        value["source_declared"] = source_declared
        return value


def capture_delivery_evidence(
    draft: dict[str, Any], *, workspace: Path, env: dict[str, str], timeout: int,
    script: Path = Path("/opt/taskforge_delivery_evidence.py"),
) -> dict[str, Any]:
    """Capture import ownership/exports after the exact editable install."""
    if not script.is_file():
        return {
            "schema_version": 1, "status": "unavailable",
            "reason": "delivery_evidence_tool_missing",
        }
    requested = sorted({
        str(name).split(".", 1)[0]
        for name in draft.get("import_names") or [] if name
    })
    distributions = sorted({
        str(value) for value in (
            draft.get("pypi_name"), draft.get("dist_name"),
        ) if value
    } | {
        str(item.get("name") if isinstance(item, dict) else item)
        for item in draft.get("pypi_name_candidates") or [] if item
    })
    test_paths = sorted({
        str(node).split("::", 1)[0]
        for node in draft.get("passed_ptp") or [] if node
    })
    with tempfile.TemporaryDirectory(prefix="taskforge-delivery-") as temporary:
        destination = Path(temporary) / "delivery_evidence.json"
        request_file = Path(temporary) / "request.json"
        request_file.write_text(json.dumps({
            "requested_roots": requested,
            "target_distributions": distributions,
            "test_paths": [
                str(workspace / relative)
                for relative in test_paths
                if (workspace / relative).exists()
            ],
            "selected_node_ids": [
                str(node_id)
                for node_id in draft.get("passed_ptp") or [] if node_id
            ],
        }), encoding="utf-8")
        argv = [
            sys.executable, str(script), "--workspace", str(workspace),
            "--output", str(destination), "--request-file", str(request_file),
        ]
        try:
            process = _run(
                argv, cwd=workspace, env=env,
                timeout=min(max(30, timeout), 180),
            )
        except ValidationError as exc:
            return {
                "schema_version": 1, "status": "unavailable",
                "reason": str(exc),
            }
        # Native libraries can terminate the evidence interpreter even though
        # the same final image has just passed its authoritative pytest nodes.
        # Retry only signal deaths with the delivery collector's import-free
        # mode. Ordinary Python errors remain visible and are not papered over.
        if process.returncode < 0 and not destination.is_file():
            fallback_argv = [*argv, "--validated-test-import-fallback"]
            try:
                fallback = _run(
                    fallback_argv, cwd=workspace, env=env,
                    timeout=min(max(30, timeout), 180),
                )
            except ValidationError as exc:
                return {
                    "schema_version": 1, "status": "unavailable",
                    "reason": (
                        f"delivery_evidence_native_signal_{-process.returncode}; "
                        f"validated_test_fallback_failed: {exc}"
                    ),
                }
            if fallback.returncode == 0 and destination.is_file():
                try:
                    value = json.loads(destination.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    return {
                        "schema_version": 1, "status": "unavailable",
                        "reason": f"delivery_evidence_fallback_invalid: {exc}",
                    }
                value["native_signal_recovered"] = -process.returncode
                return value
            process = fallback
        if process.returncode or not destination.is_file():
            return {
                "schema_version": 1, "status": "unavailable",
                "reason": f"delivery_evidence_exit_{process.returncode}: "
                + (process.stdout + process.stderr)[-2000:],
            }
        try:
            return json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "schema_version": 1, "status": "unavailable",
                "reason": f"delivery_evidence_invalid: {exc}",
            }


def discover_runtime_behavior_roots(
    draft: dict[str, Any], *, workspace: Path,
    script: Path = Path("/opt/taskforge_runtime_surface.py"),
) -> tuple[list[str], dict[str, Any]]:
    """Reuse runtime reflection's bounded import-root discovery.

    Historical tasks sometimes store a distribution name where an import root
    was expected (for example ``cache_simulator`` versus ``cachesimulator``).
    Reflection already reconciles workspace packages, stable-test imports and
    installed distribution ownership.  Behavior profiling must use the same
    roots or it can silently produce an empty but apparently valid sidecar.
    """
    requested = sorted({
        str(name).replace("-", "_").split(".", 1)[0]
        for name in draft.get("import_names") or [] if name
    })
    fallback = {
        "schema_version": 1,
        "status": "legacy_import_names_fallback",
        "requested_import_names": requested,
        "selected_import_names": requested,
    }
    if not script.is_file():
        return requested, fallback
    try:
        module_spec = importlib.util.spec_from_file_location(
            "taskforge_runtime_surface_discovery", script,
        )
        if module_spec is None or module_spec.loader is None:
            raise ImportError("runtime surface module has no loader")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        discovery = dict(module.discover_import_roots(draft, workspace=workspace))
        selected = sorted({
            str(name).split(".", 1)[0]
            for name in discovery.get("selected_import_names") or [] if name
        })
        discovery["status"] = "discovered" if selected else "empty"
        return selected, discovery
    except Exception as exc:  # evidence discovery must not invalidate tests
        fallback["reason"] = f"root_discovery_failed: {exc}"
        return requested, fallback


def final_ground_truth_accepted(
    *, passed: int, expected: int, min_pass_rate: float = 0.90,
    min_passed_tests: int = 10,
) -> bool:
    if expected < 1 or passed < 0 or passed > expected:
        return False
    return passed >= min_passed_tests and passed / expected >= min_pass_rate


def runtime_behavior_covers_denominator(
    selected_nodes: list[str], accepted_nodes: list[str], *, requested: bool,
) -> bool:
    """Prove a behavior sidecar covers the full accepted scoring denominator.

    A targeted repair fragment can be complete for its small selection while
    still being incomplete for the task.  Such fragments must go through the
    generation merger/promotion gate instead of being committed directly.
    """
    if not requested:
        return True
    return (
        len(selected_nodes) == len(accepted_nodes)
        and set(selected_nodes) == set(accepted_nodes)
    )


def _context_runtime_available() -> bool:
    """Return whether this final image can instrument pytest without installation."""
    return importlib.util.find_spec("coverage") is not None


def _write_context_runner(directory: Path) -> Path:
    plugin = directory / "taskforge_validation_context_plugin.py"
    outcome_plugin = directory / "taskforge_validation_outcome_plugin.py"
    runner = directory / "taskforge_validation_context_runner.py"
    plugin.write_text(CONTEXT_PLUGIN_SOURCE, encoding="utf-8")
    outcome_plugin.write_text(OUTCOME_PLUGIN_SOURCE, encoding="utf-8")
    runner.write_text(CONTEXT_RUNNER_SOURCE, encoding="utf-8")
    return runner


def _write_direct_runner(directory: Path) -> Path:
    plugin = directory / "taskforge_validation_outcome_plugin.py"
    runner = directory / "taskforge_validation_direct_runner.py"
    plugin.write_text(OUTCOME_PLUGIN_SOURCE, encoding="utf-8")
    runner.write_text(DIRECT_RUNNER_SOURCE, encoding="utf-8")
    return runner


def _write_runtime_behavior_plugin(directory: Path) -> Path:
    plugin = directory / "taskforge_runtime_behavior_plugin.py"
    plugin.write_text(RUNTIME_BEHAVIOR_PLUGIN_SOURCE, encoding="utf-8")
    return plugin


def _capture_runtime_behavior_shard(
    *,
    output: Path,
    workspace: Path,
    env: dict[str, str],
    draft: dict[str, Any],
    identity: dict[str, str],
    selected_nodes: list[str],
    baseline_outcomes: dict[str, str],
    pytest_command: list[str],
    addopts: list[str],
    timeout: int,
    scope: str = "targeted",
    import_roots: list[str] | None = None,
    root_discovery: dict[str, Any] | None = None,
    max_symbols_per_node: int = 256,
    max_events_per_node: int = 512,
    max_events_per_symbol: int = 2,
    max_event_bytes: int = 2000,
    trace_mode: str = "bounded_value",
) -> dict[str, Any]:
    """Capture one bounded node shard and prove it matches the baseline.

    The first validation run is deliberately uninstrumented.  Runtime evidence
    is accepted only when the second run executes every selected node and
    reports the same normalized outcome for each one.  This prevents the
    profiler from silently changing the benchmark it is meant to describe.
    """
    roots = list(import_roots or [])
    discovery = dict(root_discovery or {})
    if import_roots is None:
        roots, discovery = discover_runtime_behavior_roots(
            draft, workspace=workspace,
        )
    if scope not in {"targeted", "all_stable"}:
        raise ValueError("runtime behavior scope must be targeted or all_stable")
    if not selected_nodes:
        raise ValueError("runtime behavior collection requires non-empty node IDs")
    if not roots:
        raise ValidationError(
            "runtime behavior collection has no distribution-owned import roots"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="taskforge-behavior-", dir="/tmp",
    ) as temporary:
        root = Path(temporary)
        config = root / "config.json"
        report = root / "report.json"
        baseline_report = root / "baseline-report.json"
        arguments = root / "arguments.json"
        baseline_arguments = root / "baseline-arguments.json"
        runner = _write_direct_runner(root)
        _write_runtime_behavior_plugin(root)
        config.write_text(json.dumps({
            "output": str(output),
            "import_roots": roots,
            "root_discovery": discovery,
            "identity": identity,
            "scope": scope,
            "node_ids": selected_nodes,
            "max_symbols_per_node": max(1, int(max_symbols_per_node)),
            "max_events_per_node": max(1, int(max_events_per_node)),
            "max_events_per_symbol": max(1, int(max_events_per_symbol)),
            "max_event_bytes": max(256, int(max_event_bytes)),
            "max_value_chars": 1000,
            "max_collection_items": 20,
            "max_value_depth": 3,
            "trace_mode": (
                trace_mode if trace_mode in {"bounded_value", "minimal"}
                else "bounded_value"
            ),
        }, sort_keys=True), encoding="utf-8")
        # A plugin object installed in the controller process is not
        # automatically installed in xdist workers. Force this bounded
        # evidence pass into one process so an apparently empty sidecar cannot
        # be caused by worker instrumentation loss.
        evidence_addopts: list[str] = []
        skip_next = False
        for option in addopts:
            if skip_next:
                skip_next = False
                continue
            if option in {"-n", "--numprocesses", "--dist"}:
                skip_next = True
                continue
            if option.startswith(("-n=", "--numprocesses=", "--dist=")):
                continue
            evidence_addopts.append(option)
        pytest_prefix = [
            "-q", "--no-header", "-o", "addopts=" + shlex.join(evidence_addopts),
        ]
        instrumented_workspace = root / "instrumented-workspace"
        _copy_runtime_pristine(workspace, instrumented_workspace)
        baseline_selected_nodes = [
            remap_workspace_paths_in_nodeid(
                pytest_selection_nodeid(node), workspace
            )
            for node in selected_nodes
        ]
        instrumented_selected_nodes = [
            remap_workspace_paths_in_nodeid(
                pytest_selection_nodeid(node), instrumented_workspace
            )
            for node in selected_nodes
        ]
        pytest_arguments = [
            *pytest_prefix, "--json-report", f"--json-report-file={report}",
            *pytest_command, *instrumented_selected_nodes,
        ]
        baseline_pytest_arguments = [
            *pytest_prefix, "--json-report",
            f"--json-report-file={baseline_report}",
            *pytest_command, *baseline_selected_nodes,
        ]
        arguments.write_text(json.dumps(pytest_arguments), encoding="utf-8")
        baseline_arguments.write_text(
            json.dumps(baseline_pytest_arguments), encoding="utf-8",
        )
        baseline_env = env.copy()
        baseline_env.pop("TASKFORGE_RUNTIME_BEHAVIOR_CONFIG", None)
        baseline_env["PYTHONPATH"] = _workspace_pythonpath(
            workspace, baseline_env.get("PYTHONPATH", ""),
        )
        baseline_process = _run(
            _direct_pytest_argv(
                runner=runner, arguments_file=baseline_arguments,
            ),
            cwd=workspace, env=baseline_env, timeout=max(1, timeout),
        )
        run_env = env.copy()
        run_env["TASKFORGE_RUNTIME_BEHAVIOR_CONFIG"] = str(config)
        run_env["PYTHONPATH"] = os.pathsep.join((
            str(root),
            _workspace_pythonpath(
                instrumented_workspace, run_env.get("PYTHONPATH", ""),
            ),
        )).rstrip(os.pathsep)
        process = _run(
            _direct_pytest_argv(runner=runner, arguments_file=arguments),
            cwd=instrumented_workspace, env=run_env, timeout=max(1, timeout),
        )
        if not baseline_report.is_file() or not report.is_file() or not output.is_file():
            raise ValidationError(
                "instrumented runtime pass did not create complete artifacts: "
                + (process.stdout + process.stderr)[-4000:]
            )
        report_payload = json.loads(report.read_text(encoding="utf-8"))
        baseline_report_payload = json.loads(
            baseline_report.read_text(encoding="utf-8")
        )

        def outcomes(payload: dict[str, Any]) -> dict[str, str]:
            values: dict[str, str] = {}
            for item in payload.get("tests") or []:
                node_id = str(item.get("nodeid") or "")
                if node_id:
                    outcome = normalize_pytest_outcome(item.get("outcome"))
                    values[node_id] = merge_pytest_outcome(
                        values.get(node_id), outcome,
                    )
            return values

        raw = {
            canonicalize_workspace_paths_in_nodeid(node, instrumented_workspace): outcome
            for node, outcome in outcomes(report_payload).items()
        }
        raw_baseline = {
            canonicalize_workspace_paths_in_nodeid(node, workspace): outcome
            for node, outcome in outcomes(baseline_report_payload).items()
        }
        instrumented, remapped = reconcile_pytest_nodeids(selected_nodes, raw)
        serial_baseline, baseline_remapped = reconcile_pytest_nodeids(
            selected_nodes, raw_baseline,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        payload["selected_node_ids"] = sorted(selected_nodes)
        payload["selected_node_ids_sha256"] = _node_ids_sha256(selected_nodes)
        executed_node_ids = sorted(
            str(node) for node in payload.get("executed_node_ids") or []
        )
        payload["executed_node_ids_sha256"] = _node_ids_sha256(
            executed_node_ids
        )
        payload["plan_sha256"] = _node_ids_sha256(selected_nodes)
        baseline_state_mismatches = [
            {
                "node_id": node,
                "authoritative": baseline_outcomes.get(node, "missing"),
                "serial_baseline": serial_baseline.get(node, "missing"),
            }
            for node in selected_nodes
            if baseline_outcomes.get(node, "missing")
            != serial_baseline.get(node, "missing")
        ]
        mismatches = [
            {
                "node_id": node,
                "baseline": serial_baseline.get(node, "missing"),
                "instrumented": instrumented.get(node, "missing"),
            }
            for node in selected_nodes
            if serial_baseline.get(node, "missing")
            != instrumented.get(node, "missing")
        ]
        baseline_mismatch_node_ids = sorted(
            str(row["node_id"]) for row in baseline_state_mismatches
        )
        outcome_mismatch_node_ids = sorted(
            str(row["node_id"]) for row in mismatches
        )
        not_executed_node_ids = sorted(
            set(selected_nodes) - set(executed_node_ids)
        )
        truncated_node_ids = sorted(
            str(row.get("node_id") or "")
            for row in payload.get("nodes") or []
            if row.get("node_id") and row.get("truncated")
        )
        payload.update({
            "authoritative_to_serial_mismatch_node_ids": (
                baseline_mismatch_node_ids
            ),
            "authoritative_to_serial_mismatch_node_ids_sha256": (
                _node_ids_sha256(baseline_mismatch_node_ids)
            ),
            "serial_to_instrumented_mismatch_node_ids": (
                outcome_mismatch_node_ids
            ),
            "serial_to_instrumented_mismatch_node_ids_sha256": (
                _node_ids_sha256(outcome_mismatch_node_ids)
            ),
            "not_executed_node_ids": not_executed_node_ids,
            "not_executed_node_ids_sha256": _node_ids_sha256(
                not_executed_node_ids
            ),
            "truncated_node_ids": truncated_node_ids,
            "truncated_node_ids_sha256": _node_ids_sha256(
                truncated_node_ids
            ),
        })
        payload["baseline_comparison"] = {
            "baseline_node_count": len(selected_nodes),
            "serial_baseline_node_count": len(serial_baseline),
            "instrumented_node_count": len(instrumented),
            "baseline_rootdir_remapped_count": len(baseline_remapped),
            "instrumented_rootdir_remapped_count": len(remapped),
            "authoritative_to_serial_mismatch_count": len(
                baseline_state_mismatches
            ),
            "authoritative_to_serial_mismatch_sample": (
                baseline_state_mismatches[:50]
            ),
            "outcome_mismatch_count": len(mismatches),
            "outcome_mismatch_sample": mismatches[:50],
            "baseline_process_returncode": baseline_process.returncode,
            "process_returncode": process.returncode,
            "baseline_output_tail": (
                baseline_process.stdout + "\n" + baseline_process.stderr
            )[-4000:],
            "instrumented_output_tail": (
                process.stdout + "\n" + process.stderr
            )[-4000:],
        }
        reason_codes: list[str] = []
        if baseline_mismatch_node_ids:
            reason_codes.append("authoritative_to_serial_mismatch")
        if outcome_mismatch_node_ids:
            reason_codes.append("serial_to_instrumented_mismatch")
        if not_executed_node_ids:
            reason_codes.append("not_all_selected_nodes_executed")
        if truncated_node_ids:
            reason_codes.append("node_evidence_truncated")
        if int(payload.get("boundary_node_count") or 0) == 0:
            reason_codes.append("no_distribution_boundary_observed")

        invalid_nodes = set(baseline_mismatch_node_ids)
        invalid_nodes.update(outcome_mismatch_node_ids)
        invalid_nodes.update(not_executed_node_ids)
        invalid_nodes.update(truncated_node_ids)
        verified_node_ids = sorted(set(selected_nodes) - invalid_nodes)
        payload["nodes"] = [
            row for row in payload.get("nodes") or []
            if str(row.get("node_id") or "") in set(verified_node_ids)
        ]
        payload["verified_node_ids"] = verified_node_ids
        payload["verified_node_count"] = len(verified_node_ids)
        payload["excluded_instrumentation_mismatch_node_count"] = len(
            outcome_mismatch_node_ids
        )
        payload["reason_codes"] = reason_codes
        if not_executed_node_ids:
            payload["status"] = "incomplete_execution"
        elif baseline_mismatch_node_ids:
            payload["status"] = "baseline_state_mismatch"
        elif outcome_mismatch_node_ids:
            payload["status"] = "outcome_mismatch"
        elif truncated_node_ids:
            payload["status"] = "truncated"
        elif "no_distribution_boundary_observed" in reason_codes:
            payload["status"] = "captured_no_boundary"
        else:
            payload["status"] = "captured_complete"
        payload["verified_node_ids_sha256"] = _node_ids_sha256(
            list(payload.get("verified_node_ids") or [])
        )
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        temporary_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_output, output)
        return payload


def runtime_behavior_node_shards(
    node_ids: list[str], *, max_nodes_per_shard: int = 20,
    grouping: str = "test_file_then_node_count",
) -> list[list[str]]:
    """Create deterministic behavior shards without reducing denominator.

    ``single_suite`` is an explicit rescue mode for repositories whose tests
    pass only when pytest preserves cross-file fixture or process state. It
    must be selected by an audited repair plan; the default remains bounded,
    file-local shards so one slow module cannot monopolize an ordinary job.
    """
    if max_nodes_per_shard < 1:
        raise ValueError("max_nodes_per_shard must be positive")
    if grouping not in {"test_file_then_node_count", "single_suite"}:
        raise ValueError("unsupported runtime behavior shard grouping")
    values = sorted(set(node_ids))
    if grouping == "single_suite":
        return [values] if values else []
    by_file: dict[str, list[str]] = {}
    for node_id in values:
        by_file.setdefault(node_id.split("::", 1)[0], []).append(node_id)
    shards: list[list[str]] = []
    for path in sorted(by_file):
        values = by_file[path]
        shards.extend(
            values[offset:offset + max_nodes_per_shard]
            for offset in range(0, len(values), max_nodes_per_shard)
        )
    return shards


def _fragment_node_ids(fragment: dict[str, Any], field: str) -> list[str]:
    return sorted({str(value) for value in fragment.get(field) or [] if value})


def runtime_behavior_identity(
    *, snapshot_id: str, image_digest: str, environment_sha256: str,
    accepted_node_ids: list[str], selected_node_ids: list[str],
    collector_contract_sha256: str,
) -> dict[str, str]:
    """Bind a behavior fragment to both the stable and selected denominators.

    A targeted repair normally selects only a few broken nodes.  Its
    ``passed_ptp_sha256`` must still describe the complete outer-validation
    denominator; otherwise a subset can masquerade as a complete replacement
    and cannot be safely merged with evidence from an earlier generation.
    """
    return {
        "snapshot_id": snapshot_id,
        "image_digest": image_digest,
        "environment_sha256": environment_sha256,
        "passed_ptp_sha256": _node_ids_sha256(accepted_node_ids),
        "selected_node_ids_sha256": _node_ids_sha256(selected_node_ids),
        "collector_contract_sha256": collector_contract_sha256,
    }


def merge_runtime_behavior_fragments(
    fragments: list[dict[str, Any]], *, identity: dict[str, str],
    required_node_ids: list[str], scope: str = "targeted",
) -> dict[str, Any]:
    """Merge shard evidence by node ID and retain complete failure ledgers."""
    required = sorted(set(required_node_ids))
    selected: set[str] = set()
    executed: set[str] = set()
    verified: set[str] = set()
    ledgers = {
        "authoritative_to_serial_mismatch_node_ids": set(),
        "serial_to_instrumented_mismatch_node_ids": set(),
        "not_executed_node_ids": set(),
        "truncated_node_ids": set(),
    }
    reason_codes: set[str] = set()
    rows: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    import_roots: set[str] = set()
    for index, fragment in enumerate(fragments):
        fragment_identity = dict(fragment.get("identity") or {})
        if fragment_identity and fragment_identity != identity:
            raise ValueError("runtime behavior fragment identity mismatch")
        fragment_selected = set(_fragment_node_ids(fragment, "selected_node_ids"))
        fragment_executed = set(_fragment_node_ids(fragment, "executed_node_ids"))
        fragment_verified = set(_fragment_node_ids(fragment, "verified_node_ids"))
        selected.update(fragment_selected)
        executed.update(fragment_executed)
        verified.update(fragment_verified)
        reason_codes.update(str(value) for value in fragment.get("reason_codes") or [])
        import_roots.update(str(value) for value in fragment.get("import_roots") or [])
        for field, values in ledgers.items():
            values.update(_fragment_node_ids(fragment, field))
        fragment_sha = hashlib.sha256(json.dumps(
            fragment, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        shard = dict(fragment.get("shard") or {})
        provenance.append({
            "shard_index": int(shard.get("index", index)),
            "selected_node_count": len(fragment_selected),
            "status": str(fragment.get("status") or "unknown"),
            "sha256": fragment_sha,
        })
        for row in fragment.get("nodes") or []:
            node_id = str(row.get("node_id") or "")
            if not node_id or node_id not in fragment_verified:
                continue
            candidate = dict(row)
            candidate["provenance"] = {
                "shard_index": int(shard.get("index", index)),
                "fragment_sha256": fragment_sha,
                "generation": str(shard.get("generation") or ""),
                "collector_contract_sha256": str(
                    shard.get("collector_contract_sha256") or ""
                ),
            }
            # A verified newer fragment may replace a verified older row.  An
            # invalid fragment never deletes already verified evidence.
            rows[node_id] = candidate
    selected.update(required)
    verified.intersection_update(rows)
    missing = set(required) - verified
    ledgers["not_executed_node_ids"].update(set(required) - executed)
    if missing:
        reason_codes.add("required_nodes_not_verified")
    for field, values in ledgers.items():
        if values:
            mapping = {
                "authoritative_to_serial_mismatch_node_ids": (
                    "authoritative_to_serial_mismatch"
                ),
                "serial_to_instrumented_mismatch_node_ids": (
                    "serial_to_instrumented_mismatch"
                ),
                "not_executed_node_ids": "not_all_selected_nodes_executed",
                "truncated_node_ids": "node_evidence_truncated",
            }
            reason_codes.add(mapping[field])
    if ledgers["not_executed_node_ids"]:
        status = "incomplete_execution"
    elif ledgers["authoritative_to_serial_mismatch_node_ids"]:
        status = "baseline_state_mismatch"
    elif ledgers["serial_to_instrumented_mismatch_node_ids"]:
        status = "outcome_mismatch"
    elif ledgers["truncated_node_ids"]:
        status = "truncated"
    elif missing:
        status = "incomplete_execution"
    elif not rows:
        status = "captured_no_boundary"
        reason_codes.add("no_distribution_boundary_observed")
    else:
        status = "captured_complete"
    payload: dict[str, Any] = {
        "schema_version": 4,
        "status": status,
        "identity": identity,
        "scope": scope,
        "import_roots": sorted(import_roots),
        "selected_node_ids": sorted(selected),
        "executed_node_ids": sorted(executed),
        "verified_node_ids": sorted(verified),
        "required_node_ids": required,
        "reason_codes": sorted(reason_codes),
        "nodes": [rows[node] for node in sorted(rows)],
        "fragments": sorted(provenance, key=lambda row: row["shard_index"]),
        "selected_node_count": len(selected),
        "executed_node_count": len(executed),
        "executed_selected_node_count": len(executed & set(selected)),
        "verified_node_count": len(verified),
        "required_node_count": len(required),
        "observed_node_count": len(rows),
        "boundary_node_count": len(rows),
        "event_node_count": sum(bool(row.get("events")) for row in rows.values()),
        "truncated_node_count": len(ledgers["truncated_node_ids"]),
    }
    for field, values in ledgers.items():
        ordered = sorted(values)
        payload[field] = ordered
        payload[field + "_count"] = len(ordered)
        payload[field + "_sha256"] = _node_ids_sha256(ordered)
    for field in ("selected_node_ids", "executed_node_ids", "verified_node_ids", "required_node_ids"):
        payload[field + "_sha256"] = _node_ids_sha256(list(payload[field]))
    payload["plan_sha256"] = payload["required_node_ids_sha256"]
    return payload


def capture_runtime_behavior_evidence(
    *, output: Path, workspace: Path, env: dict[str, str],
    draft: dict[str, Any], identity: dict[str, str],
    selected_nodes: list[str], baseline_outcomes: dict[str, str],
    pytest_command: list[str], addopts: list[str], timeout: int,
    scope: str = "targeted", import_roots: list[str] | None = None,
    root_discovery: dict[str, Any] | None = None,
    max_nodes_per_shard: int = 20, shard_timeout: int | None = None,
    shard_grouping: str = "test_file_then_node_count",
    max_symbols_per_node: int = 256, max_events_per_node: int = 512,
    max_events_per_symbol: int = 2, max_event_bytes: int = 2000,
    trace_mode: str = "bounded_value",
    generation: str = "spec-runtime-v5-targeted-repair1",
    collector_contract_sha256: str = "",
) -> dict[str, Any]:
    """Capture independent pristine shards, continue failures, then merge."""
    shards = runtime_behavior_node_shards(
        selected_nodes, max_nodes_per_shard=max_nodes_per_shard,
        grouping=shard_grouping,
    )
    fragment_root = output.parent / (output.stem + ".fragments")
    fragment_root.mkdir(parents=True, exist_ok=True)
    fragments: list[dict[str, Any]] = []
    for index, shard_nodes in enumerate(shards):
        fragment_path = fragment_root / f"shard-{index:05d}.json"
        with tempfile.TemporaryDirectory(
            prefix=f"taskforge-behavior-shard-{index:05d}-", dir="/tmp",
        ) as temporary:
            shard_workspace = Path(temporary) / "workspace"
            _copy_runtime_pristine(workspace, shard_workspace)
            try:
                fragment = _capture_runtime_behavior_shard(
                    output=fragment_path, workspace=shard_workspace, env=env,
                    draft=draft, identity=identity,
                    selected_nodes=shard_nodes,
                    baseline_outcomes={
                        node: baseline_outcomes.get(node, "missing")
                        for node in shard_nodes
                    },
                    pytest_command=pytest_command, addopts=addopts,
                    timeout=max(1, shard_timeout or timeout), scope=scope,
                    import_roots=import_roots, root_discovery=root_discovery,
                    max_symbols_per_node=max_symbols_per_node,
                    max_events_per_node=max_events_per_node,
                    max_events_per_symbol=max_events_per_symbol,
                    max_event_bytes=max_event_bytes,
                    trace_mode=trace_mode,
                )
            except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                fragment = {
                    "schema_version": 4,
                    "status": "shard_failed",
                    "identity": identity,
                    "scope": scope,
                    "selected_node_ids": sorted(shard_nodes),
                    "executed_node_ids": [],
                    "verified_node_ids": [],
                    "not_executed_node_ids": sorted(shard_nodes),
                    "reason_codes": ["shard_capture_failed"],
                    "reason": str(exc)[:4000],
                    "nodes": [],
                }
            fragment["shard"] = {
                "index": index,
                "count": len(shards),
                "max_nodes_per_shard": max_nodes_per_shard,
                "timeout_seconds": max(1, shard_timeout or timeout),
                "generation": generation,
                "collector_contract_sha256": collector_contract_sha256,
            }
            fragment_path.write_text(
                json.dumps(fragment, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fragments.append(fragment)
    merged = merge_runtime_behavior_fragments(
        fragments, identity=identity, required_node_ids=selected_nodes,
        scope=scope,
    )
    merged["shard_policy"] = {
        "max_nodes_per_shard": max_nodes_per_shard,
        "shard_timeout_seconds": max(1, shard_timeout or timeout),
        "shard_count": len(shards),
        "grouping": shard_grouping,
        "generation": generation,
        "collector_contract_sha256": collector_contract_sha256,
        "max_symbols_per_node": max_symbols_per_node,
        "max_events_per_node": max_events_per_node,
        "max_events_per_symbol": max_events_per_symbol,
        "max_event_bytes": max_event_bytes,
        "trace_mode": trace_mode,
    }
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_output, output)
    return merged


def _direct_pytest_argv(*, runner: Path, arguments_file: Path) -> list[str]:
    return [sys.executable, str(runner), str(arguments_file)]


def _context_pytest_argv(
    *, runner: Path, arguments_file: Path, workspace: Path, append: bool,
) -> list[str]:
    argv = ["python", "-m", "coverage", "run"]
    if append:
        argv.append("--append")
    argv.extend([
        "--rcfile=/dev/null", "--branch", f"--source={workspace}",
        str(runner), str(arguments_file),
    ])
    return argv


def _split_frozen_shell_words(value: str) -> list[str]:
    """Split the shell-free command grammar in linear time.

    Dataset commands are produced with :func:`shlex.quote`, but ``shlex``
    builds a token through repeated string concatenation.  A parametrized
    pytest node can legitimately contain megabytes of generated data, making
    that implementation effectively quadratic.  This parser accepts POSIX
    quoting and escaping while accumulating token fragments in a list.  Its
    output is used as argv and is never evaluated by a shell.
    """
    words: list[str] = []
    token: list[str] = []
    token_started = False
    quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote == "'":
            if character == "'":
                quote = None
            else:
                token.append(character)
            token_started = True
        elif quote == '"':
            if character == '"':
                quote = None
            elif character == "\\":
                index += 1
                if index >= len(value):
                    raise ValueError("trailing escape in double-quoted word")
                escaped = value[index]
                # POSIX double quotes only consume the backslash for these
                # characters. Preserve it for all other quoted characters.
                if escaped not in {'$', '`', '"', "\\", "\n"}:
                    token.append("\\")
                token.append(escaped)
            else:
                token.append(character)
            token_started = True
        elif character.isspace():
            if token_started:
                words.append("".join(token))
                token = []
                token_started = False
        elif character in {"'", '"'}:
            quote = character
            token_started = True
        elif character == "\\":
            index += 1
            if index >= len(value):
                raise ValueError("trailing escape in word")
            token.append(value[index])
            token_started = True
        else:
            token.append(character)
            token_started = True
        index += 1
    if quote is not None:
        raise ValueError("unterminated quoted word")
    if token_started:
        words.append("".join(token))
    return words


def published_pytest_chunks(
    commands: Iterable[str], passed_ptp: Iterable[str],
) -> list[tuple[list[str], list[str]]]:
    """Parse and authenticate the exact commands shipped in the dataset.

    The public scorer executes ``test_commands`` one process at a time.  Gold
    validation must preserve those process boundaries instead of silently
    coalescing all selectors into one pytest session.  Only TaskForge's
    deliberately shell-free command grammar is accepted; the command is
    parsed into argv and never passed to a shell. Both published prefixes
    are supported: legacy commands inherit pytest's configured addopts,
    while commands with ``-o addopts=`` clear them. Preserve that choice
    per command rather than silently adding or removing the override.
    """
    expected = list(passed_ptp)
    expected_set = set(expected)
    if not expected or len(expected) != len(expected_set):
        raise ValidationError("published pytest denominator is empty or duplicated")
    raw_commands = list(commands)
    if not raw_commands or not all(
        isinstance(command, str) and command.strip() for command in raw_commands
    ):
        raise ValidationError("published test_commands must be non-empty strings")
    legacy_prefix = [
        "python", "-m", "pytest", "-q", "--no-header",
        "-p", "no:cacheprovider",
    ]
    # Match the longer form first; the shorter one is its literal prefix.
    prefixes = ([*legacy_prefix, "-o", "addopts="], legacy_prefix)
    chunks: list[tuple[list[str], list[str]]] = []
    seen: set[str] = set()
    for index, command in enumerate(raw_commands):
        try:
            argv = _split_frozen_shell_words(command)
        except ValueError as exc:
            raise ValidationError(
                f"published test command {index} cannot be parsed: {exc}"
            ) from exc
        prefix = next((value for value in prefixes if argv[:len(value)] == value), None)
        if prefix is None:
            raise ValidationError(
                f"published test command {index} does not use the frozen "
                "TaskForge pytest prefix"
            )
        nodes = argv[len(prefix):]
        if not nodes or any(node not in expected_set for node in nodes):
            raise ValidationError(
                f"published test command {index} contains an unknown selector"
            )
        duplicated = seen.intersection(nodes)
        if duplicated or len(nodes) != len(set(nodes)):
            raise ValidationError(
                f"published test command {index} repeats selectors"
            )
        seen.update(nodes)
        chunks.append((prefix[3:], nodes))
    if seen != expected_set:
        missing = sorted(expected_set - seen)
        raise ValidationError(
            "published test_commands do not cover the frozen denominator: "
            + ", ".join(missing[:20])
        )
    return chunks


def validate(
    *, prepared_dir: Path, output: Path, clean_script: Path = Path("/opt/taskforge_clean.sh"),
    submission_patch: Path | None = None,
    submission_dir: Path | None = None,
    suite_timeout_s: int = 1800, command_timeout_s: int = 900,
    min_pass_rate: float = 0.90, min_passed_tests: int = 10,
    context_data_output: Path | None = None,
    runtime_api_output: Path | None = None,
    delivery_evidence_output: Path | None = None,
    runtime_behavior_output: Path | None = None,
    runtime_behavior_node_ids_file: Path | None = None,
    runtime_api_script: Path = Path("/opt/taskforge_runtime_surface.py"),
    delivery_evidence_script: Path = Path("/opt/taskforge_delivery_evidence.py"),
    collector_manifest: Path | None = None,
    context_max_bytes: int = 512 * 1024 * 1024,
    node_ids_file: Path | None = None,
    pytest_session_policy: str = "contract",
) -> dict[str, Any]:
    if submission_patch is not None and submission_dir is not None:
        raise ValueError("provide a submission patch or directory, not both")
    if not 0.0 <= min_pass_rate <= 1.0 or min_passed_tests < 1:
        raise ValueError("invalid final ground-truth policy")
    started = time.monotonic()
    collector_identity = _load_collector_identity(collector_manifest)
    behavior_workspace_root: Path | None = None
    task_path = prepared_dir / "task.json"
    draft = json.loads((
        task_path if task_path.is_file() else prepared_dir / "draft.json"
    ).read_text(encoding="utf-8"))
    spec = json.loads((prepared_dir / "validation_spec.json").read_text(encoding="utf-8"))
    workspace = Path(str(spec["workdir"])).resolve()
    h3_install_contract = dict(spec.get("h3_install_contract") or {})
    declared_h3_contract_sha256 = str(
        spec.get("h3_install_contract_sha256") or ""
    )
    supplied_h3_contract_sha256 = hashlib.sha256(json.dumps(
        h3_install_contract, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if (
        declared_h3_contract_sha256
        and declared_h3_contract_sha256 != supplied_h3_contract_sha256
    ):
        raise ValueError("H3 install contract digest mismatch")
    install_targets = list(spec.get("install_targets") or [])
    require_editable_install = bool(spec.get("require_editable_install", False))
    parent_commit = str(spec["parent_commit"])
    passed_ptp = list(spec["passed_ptp"])
    execution_contract = str(
        spec.get("pytest_execution_contract") or "single_session_v1"
    )
    gold_injection_policy = str(
        spec.get("gold_injection_policy") or "implementation_only_v1"
    )
    if gold_injection_policy != "implementation_only_v1":
        raise ValueError(
            f"unsupported gold injection policy: {gold_injection_policy}"
        )
    if node_ids_file is not None:
        selected_ptp = json.loads(node_ids_file.read_text(encoding="utf-8"))
        if not isinstance(selected_ptp, list) or not all(
            isinstance(node, str) for node in selected_ptp
        ):
            raise ValueError("node ID override must be a JSON string list")
        if not set(selected_ptp).issubset(passed_ptp):
            raise ValueError("node ID override is not a subset of validation ground truth")
        passed_ptp = list(dict.fromkeys(selected_ptp))
    if pytest_session_policy not in {"contract", "single_session"}:
        raise ValueError("invalid pytest session policy")
    explicit_environment = {
        str(key): str(value)
        for key, value in (spec.get("environment") or {}).items()
    }
    env = os.environ.copy()
    env.update(explicit_environment)
    env.update({"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"})
    if execution_contract == "published_test_commands_v1":
        # The public scorer does not add /workspace/src as an escape hatch.
        # A src-layout project must become importable through the same editable
        # install that an agent submission receives.
        env["PYTHONPATH"] = env.get("PYTHONPATH", "")
    else:
        env["PYTHONPATH"] = _workspace_pythonpath(
            workspace, env.get("PYTHONPATH", ""),
        )

    result: dict[str, Any] = {
        "schema_version": 2,
        "snapshot_id": spec["snapshot_id"],
        "instance_id": spec["instance_id"],
        "status": "parked",
        "reason": "validation_error",
        "image": os.environ.get("TASKFORGE_IMAGE", ""),
        "image_digest": os.environ.get("TASKFORGE_IMAGE_DIGEST", ""),
        "parent_commit": parent_commit,
        "expected": len(passed_ptp),
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "observed_node_ids_sha256": "",
        "install_attempts": [],
        "install_fallback_targets": [],
        "install_contract": {
            "status": "pending", "requested_targets": [],
            "editable_installed_targets": [],
            "h3_contract": h3_install_contract,
            "supplied_h3_contract_sha256": supplied_h3_contract_sha256,
        },
        "non_passed": [],
        "accepted_passed_ptp": [],
        "pass_rate": 0.0,
        "ground_truth_adjusted": False,
        "outcome_counts": {},
        "pytest_diagnostics": [],
        "selector_recovery": [],
        "pytest_execution_contract": execution_contract,
        "pytest_session_policy": pytest_session_policy,
        "h3_runtime_contract": {
            "status": "pending", "environment": {},
            "rejected_environment": [], "pytest_args": [],
            "dropped_pytest_args": [],
        },
        "timing": {
            "schema_version": 1,
            "total_wall_seconds": None,
            "editable_install_wall_seconds": 0.0,
            "pytest_wall_seconds": None,
            "pytest_planned_command_count": 0,
            "pytest_completed_command_count": 0,
            "pytest_max_command_wall_seconds": 0.0,
            "pytest_timed_out": False,
            "pytest_commands": [],
        },
        "gold_injection": {
            "policy": gold_injection_policy,
            "status": "pending",
            "authoritative_test_source": (
                "test_patch_plus_binary_fixtures"
            ),
        },
        "environment": {},
        "environment_sha256": "",
        "runtime_api_inventory": {
            "schema_version": 1, "status": "pending", "symbols": [],
            "symbol_count": 0, "import_errors": [],
            "artifact": os.environ.get("TASKFORGE_RUNTIME_API_ARTIFACT", ""),
        },
        "runtime_api_inventory_sha256": "",
        "delivery_evidence": {
            "schema_version": 1, "status": "pending",
        },
        "delivery_evidence_sha256": "",
        "runtime_behavior_evidence": {
            "requested": runtime_behavior_output is not None,
            "status": "not_requested" if runtime_behavior_output is None else "pending",
            "selected_node_count": 0,
            "executed_selected_node_count": 0,
            "observed_node_count": 0,
            "truncated_node_count": 0,
            "sha256": "",
        },
        "coverage_context": {
            "requested": context_data_output is not None,
            "status": "not_requested" if context_data_output is None else "pending",
            "complete": False,
            "artifact": os.environ.get("TASKFORGE_CONTEXT_ARTIFACT", ""),
            "chunks_recorded": 0,
            "data_bytes": 0,
            "data_sha256": "",
        },
        "specification_collector": collector_identity,
    }
    expected_behavior_identity: dict[str, str] = {}
    pytest_suite_started: float | None = None
    pytest_suite_finished: float | None = None
    active_pytest_timing: dict[str, Any] | None = None
    # Finalization below runs for every outcome, including failures raised
    # before pytest planning begins (missing workspace, bad patch identity,
    # failed clean/install).  Keep its denominator inputs initialized outside
    # the guarded execution block so diagnostic materialization itself cannot
    # mask the original validation error.
    behavior_nodes: list[str] = []
    if runtime_behavior_output is not None:
        runtime_behavior_output.parent.mkdir(parents=True, exist_ok=True)
        runtime_behavior_output.unlink(missing_ok=True)
    try:
        h3_install_contract = _resolve_h3_install_contract(spec)
        install_targets = list(h3_install_contract["install_targets"])
        h3_buildspec = h3_install_contract.get("buildspec") or {}
        if not isinstance(h3_buildspec, dict):
            raise ValidationError("H3 runtime contract BuildSpec is not a mapping")
        h3_environment, rejected_h3_environment = _safe_h3_environment(
            h3_buildspec, workspace=workspace,
        )
        h3_pytest_args, dropped_h3_pytest_args = _h3_runtime_pytest_args(
            h3_buildspec,
        )
        # Explicit release settings take precedence over historical H3, while
        # evaluator-owned HOME/bytecode settings always remain fixed.
        env.update(h3_environment)
        env.update(explicit_environment)
        env.update({"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"})
        result["h3_runtime_contract"] = {
            "status": "restored" if h3_buildspec else "not_available",
            "environment": h3_environment,
            "rejected_environment": rejected_h3_environment,
            "pytest_args": h3_pytest_args,
            "dropped_pytest_args": dropped_h3_pytest_args,
            "buildspec_sha256": str(
                h3_install_contract.get("buildspec_sha256") or ""
            ),
        }
        if str(h3_install_contract.get("status") or "") in {
            "resolved", "resolved_from_checkpoint", "resolved_from_image",
        }:
            require_editable_install = bool(install_targets)
        result["install_contract"]["h3_contract"] = h3_install_contract
        expected_environment = prepared_dir / "environment.expected.json"
        if expected_environment.is_file():
            # write_json is stable but includes whitespace; compare the canonical
            # object digest used by preparation rather than the file byte digest.
            expected_value = json.loads(expected_environment.read_text(encoding="utf-8"))
            expected_digest = hashlib.sha256(json.dumps(
                expected_value, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            if expected_digest != str(spec.get("expected_environment_sha256") or ""):
                raise ValidationError("expected environment contract digest mismatch")
        if not workspace.is_dir() or not (workspace / ".git").is_dir():
            raise ValidationError("final image does not contain the expected Git workspace")
        _require_success(
            ["git", "checkout", "-q", "--detach", "--force", parent_commit],
            cwd=workspace, env=env, timeout=120, phase="parent checkout",
        ).stdout
        actual = _require_success(
            ["git", "rev-parse", "HEAD"], cwd=workspace, env=env, timeout=30,
            phase="commit verification",
        ).stdout.strip()
        if actual != parent_commit:
            raise ValidationError(f"HEAD {actual} does not equal parent_commit")

        patch_bytes = (prepared_dir / "test_patch.diff").read_bytes()
        patch_text = patch_bytes.decode("utf-8")
        patch_digest = hashlib.sha256(patch_bytes).hexdigest()
        if patch_digest != spec["test_patch_sha256"]:
            raise ValidationError("test patch digest mismatch")
        binary = prepared_dir / "binfix.tar.gz"
        expected_binary = draft.get("test_binary_files") or []
        if expected_binary:
            if not binary.is_file():
                raise ValidationError("draft declares binary fixtures but archive is missing")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            if digest != spec["binary_archive_sha256"]:
                raise ValidationError("binary archive digest mismatch")

        # Capture a perfect solver implementation before clean destroys the
        # parent source.  The hidden test bundle is explicitly subtracted from
        # this copy.  Only after clean do we restore the implementation and,
        # as a separate operation, materialize the authoritative text/binary
        # test payload.  This is the same ordering as real scoring and prevents
        # gold tests from hitchhiking in the solver payload.
        with tempfile.TemporaryDirectory(prefix="taskforge-gold-", dir="/tmp") as temp:
            gold = Path(temp) / "workspace"
            patch_targets = patch_new_files(patch_text)
            result["gold_injection"] = _prepare_gold_implementation(
                workspace, gold,
                declared_test_files=draft.get("test_files") or [],
                patch_targets=patch_targets,
                binary_targets=expected_binary,
            )
            cleaned_roots: list[str] = []
            cleanup_attempts: list[dict[str, Any]] = []
            for cleanup_root in _cleanup_roots(workspace, install_targets):
                cleanup_attempts.append(_run_clean_script(
                    clean_script, cleanup_root, workspace=workspace, env=env,
                ))
                cleaned_roots.append(str(cleanup_root))
            result["install_contract"]["cleaned_roots"] = cleaned_roots
            result["install_contract"]["cleanup_attempts"] = cleanup_attempts
            declared_test_files = [
                _safe_relative(str(relative)).as_posix()
                for relative in (draft.get("test_files") or [])
            ]

            def remove_declared_test_material() -> int:
                removed = 0
                for relative in declared_test_files:
                    target = workspace / _safe_relative(relative)
                    if target.exists() or target.is_symlink():
                        removed += 1
                    _remove_path(target)
                return removed

            if submission_dir is not None:
                workspace_material_removed = remove_declared_test_material()
                document = str(draft.get("document") or "")
                if document and "README.md" not in declared_test_files:
                    (workspace / "README.md").unlink(missing_ok=True)
                    (workspace / "README.md").write_text(document, encoding="utf-8")
                result["submission"] = materialize_implementation(
                    submission_dir, workspace, protected_paths=declared_test_files,
                )
                result["gold_injection"]["implementation_source"] = "agent_directory"
            elif submission_patch is not None:
                if not submission_patch.is_file():
                    raise ValidationError("submission patch is missing")
                submission_bytes = submission_patch.read_bytes()
                if submission_bytes and not submission_bytes.strip():
                    raise ValidationError("submission patch contains only whitespace")
                # OpenHands sees the Specification as README.md in its clean
                # workspace.  The cleaner intentionally removes upstream
                # Markdown, so restore the same untracked input before applying
                # the patch produced relative to that clean baseline.
                document = str(draft.get("document") or "")
                if document:
                    (workspace / "README.md").write_text(
                        document, encoding="utf-8",
                    )
                if submission_bytes:
                    _require_success(
                        ["git", "apply", "--binary", str(submission_patch)],
                        cwd=workspace, env=env, timeout=120,
                        phase="OpenHands submission patch",
                    )
                result["gold_injection"].update({
                    "implementation_source": (
                        "openhands_submission_patch" if submission_bytes
                        else "openhands_empty_submission"
                    ),
                    "submission_patch_sha256": hashlib.sha256(
                        submission_bytes
                    ).hexdigest(),
                })
                # The submitted implementation must not be able to smuggle in
                # any declared hidden-test material. Remove it before restoring
                # the authoritative frozen payload below.
                workspace_material_removed = remove_declared_test_material()
            else:
                # Reference and directory submissions intentionally share the
                # exact same materializer and protected-path policy.
                workspace_material_removed = remove_declared_test_material()
                document = str(draft.get("document") or "")
                if document and "README.md" not in declared_test_files:
                    (workspace / "README.md").unlink(missing_ok=True)
                    (workspace / "README.md").write_text(document, encoding="utf-8")
                result["submission"] = materialize_implementation(
                    gold, workspace, protected_paths=declared_test_files,
                )
                result["gold_injection"]["implementation_source"] = (
                    "fixed_commit_reference_gold"
                )
            # Restoring directories may expose lower-layer files again on
            # nested overlay storage. Regardless of implementation origin,
            # the authoritative bundle exclusively owns these paths. Also
            # handles README files used as doctest/test-payload resources.
            workspace_material_removed += remove_declared_test_material()
            unexpectedly_present = [
                relative for relative in [*patch_targets, *expected_binary]
                if (workspace / _safe_relative(relative)).exists()
                or (workspace / _safe_relative(relative)).is_symlink()
            ]
            if unexpectedly_present:
                raise ValidationError(
                    "clean/gold overlay left test material before injection: "
                    + ", ".join(unexpectedly_present[:20])
                )
            _require_success(
                ["git", "apply", "--binary", str(prepared_dir / "test_patch.diff")],
                cwd=workspace, env=env, timeout=120, phase="test patch",
            )
            if expected_binary:
                safe_extract_binary(binary, workspace, expected_binary)
            missing_after_injection = [
                relative for relative in [*patch_targets, *expected_binary]
                if not (workspace / _safe_relative(relative)).is_file()
            ]
            if missing_after_injection:
                raise ValidationError(
                    "test material missing after injection: "
                    + ", ".join(missing_after_injection[:20])
                )
            result["gold_injection"].update({
                "materialized_test_file_count": len(patch_targets) + len(expected_binary),
                "workspace_test_material_removed_before_injection": (
                    workspace_material_removed
                ),
                "status": "materialized",
            })
            test_manifest = [
                {"path": relative, "sha256": _sha256_file(workspace / relative),
                 "mode": (workspace / relative).stat().st_mode & 0o777}
                for relative in sorted(declared_test_files)
            ]
            result["frozen_test_material"] = {
                "files": test_manifest,
                "sha256": hashlib.sha256(json.dumps(
                    test_manifest, sort_keys=True, separators=(",", ":"),
                ).encode()).hexdigest(),
            }
            result["pipeline_stage"] = "implementation_and_tests_materialized"

        deferred_required_installs: list[dict[str, Any]] = []
        for target in install_targets:
            result["install_contract"]["requested_targets"].append(str(target))
            relative = Path(".") if target in {"", "."} else _safe_relative(str(target))
            install_target = (workspace / relative).resolve()
            install_env = _install_environment(workspace, env)
            install_modes = (
                (("public_scorer_no_build_isolation", [
                    "--no-deps", "--no-build-isolation",
                ]),)
                if execution_contract == "published_test_commands_v1" and (
                    h3_install_contract.get("install_mode")
                    == "no_build_isolation"
                )
                else (("public_scorer", ["--no-deps"]),)
                if execution_contract == "published_test_commands_v1"
                else (
                    ("no_build_isolation", ["--no-deps", "--no-build-isolation"]),
                    ("pep517_default", ["--no-deps"]),
                )
            )
            installed = False
            target_attempts: list[dict[str, Any]] = []
            for mode, flags in install_modes:
                install_started = time.monotonic()
                try:
                    install = _run(
                        [sys.executable, "-m", "pip", "install", "-e", str(install_target), *flags],
                        cwd=Path("/tmp"), env=install_env, timeout=600,
                    )
                except ValidationError:
                    elapsed = round(time.monotonic() - install_started, 3)
                    result["timing"]["editable_install_wall_seconds"] = round(
                        result["timing"]["editable_install_wall_seconds"] + elapsed,
                        3,
                    )
                    result["install_attempts"].append({
                        "target": str(target), "mode": mode,
                        "returncode": None, "wall_seconds": elapsed,
                        "timed_out": True,
                    })
                    raise
                elapsed = round(time.monotonic() - install_started, 3)
                result["timing"]["editable_install_wall_seconds"] = round(
                    result["timing"]["editable_install_wall_seconds"] + elapsed,
                    3,
                )
                attempt = {
                    "target": str(target), "mode": mode,
                    "returncode": install.returncode,
                    "wall_seconds": elapsed,
                    "timed_out": False,
                }
                if install.returncode:
                    attempt["output_tail"] = (install.stdout + "\n" + install.stderr)[-4000:]
                result["install_attempts"].append(attempt)
                target_attempts.append(attempt)
                if not install.returncode:
                    installed = True
                    result["install_contract"]["editable_installed_targets"].append(
                        str(target)
                    )
                    break
            if (
                not installed
                and execution_contract != "published_test_commands_v1"
                and _needs_git_version_anchor(target_attempts)
            ):
                anchor = _ensure_git_version_anchor(
                    workspace, parent_commit=parent_commit, env=install_env,
                    attempts=target_attempts,
                )
                result["install_contract"]["version_anchor"] = anchor
                install_started = time.monotonic()
                try:
                    install = _run(
                        [
                            sys.executable, "-m", "pip", "install", "-e",
                            str(install_target), "--no-deps", "--no-build-isolation",
                            "--no-cache-dir",
                        ],
                        cwd=Path("/tmp"), env=install_env, timeout=600,
                    )
                except ValidationError:
                    elapsed = round(time.monotonic() - install_started, 3)
                    result["timing"]["editable_install_wall_seconds"] = round(
                        result["timing"]["editable_install_wall_seconds"] + elapsed,
                        3,
                    )
                    result["install_attempts"].append({
                        "target": str(target),
                        "mode": "version_anchor_no_build_isolation",
                        "returncode": None,
                        "version_anchor_tag": anchor["tag"],
                        "wall_seconds": elapsed,
                        "timed_out": True,
                    })
                    raise
                elapsed = round(time.monotonic() - install_started, 3)
                result["timing"]["editable_install_wall_seconds"] = round(
                    result["timing"]["editable_install_wall_seconds"] + elapsed,
                    3,
                )
                attempt = {
                    "target": str(target),
                    "mode": "version_anchor_no_build_isolation",
                    "returncode": install.returncode,
                    "version_anchor_tag": anchor["tag"],
                    "wall_seconds": elapsed,
                    "timed_out": False,
                }
                if install.returncode:
                    attempt["output_tail"] = (
                        install.stdout + "\n" + install.stderr
                    )[-4000:]
                result["install_attempts"].append(attempt)
                if not install.returncode:
                    installed = True
                    result["install_contract"]["editable_installed_targets"].append(
                        str(target)
                    )
            if not installed:
                if require_editable_install:
                    # Do not abort a monorepo on its first failed editable.
                    # H3 installs all local projects in one pip transaction,
                    # so an earlier target can legitimately depend on a later
                    # one that was already present before clean. Finish the
                    # first pass, then retry only the failed targets.
                    deferred_required_installs.append({
                        "target": str(target),
                        "install_target": install_target,
                        "attempts": target_attempts,
                        "flags": list(install_modes[0][1]),
                    })
                    continue
                # H3 already installed the selected distribution into this
                # immutable base image. Keep testing through the explicit
                # workspace PYTHONPATH; acceptance still requires the real
                # tests to clear the unchanged 90%/10-test gate. The failure
                # remains auditable and Specification can avoid claiming that
                # a fresh editable build succeeded.
                result["install_fallback_targets"].append(str(target))

        for deferred in deferred_required_installs:
            target = str(deferred["target"])
            install_target = Path(deferred["install_target"])
            target_attempts = list(deferred["attempts"])
            flags = list(deferred["flags"])
            base_install_env = _install_environment(workspace, env)
            repair_profiles: list[tuple[str, Path, dict[str, str]]] = [
                ("deferred_dependency_order", Path("/tmp"), base_install_env),
            ]
            if _needs_source_metadata_path(target_attempts):
                source_env = base_install_env.copy()
                source_env["PYTHONPATH"] = _workspace_pythonpath(
                    workspace, source_env.get("PYTHONPATH", ""),
                )
                repair_profiles.append((
                    "metadata_source_path", install_target, source_env,
                ))
            if _needs_setuptools_scm_override(target_attempts):
                scm_env = base_install_env.copy()
                scm_env.setdefault(
                    "SETUPTOOLS_SCM_PRETEND_VERSION", "0+taskforge",
                )
                repair_profiles.append((
                    "setuptools_scm_pretend_version", install_target, scm_env,
                ))

            installed = False
            attempted_modes: set[str] = set()
            for mode, install_cwd, repair_env in repair_profiles:
                if mode in attempted_modes:
                    continue
                attempted_modes.add(mode)
                install_started = time.monotonic()
                try:
                    install = _run(
                        [
                            sys.executable, "-m", "pip", "install", "-e",
                            str(install_target), *flags,
                        ],
                        cwd=install_cwd, env=repair_env, timeout=600,
                    )
                except ValidationError:
                    elapsed = round(time.monotonic() - install_started, 3)
                    result["timing"]["editable_install_wall_seconds"] = round(
                        result["timing"]["editable_install_wall_seconds"] + elapsed,
                        3,
                    )
                    result["install_attempts"].append({
                        "target": target, "mode": mode,
                        "returncode": None, "wall_seconds": elapsed,
                        "timed_out": True,
                    })
                    raise
                elapsed = round(time.monotonic() - install_started, 3)
                result["timing"]["editable_install_wall_seconds"] = round(
                    result["timing"]["editable_install_wall_seconds"] + elapsed,
                    3,
                )
                attempt = {
                    "target": target, "mode": mode,
                    "returncode": install.returncode,
                    "wall_seconds": elapsed, "timed_out": False,
                }
                if install.returncode:
                    attempt["output_tail"] = (
                        install.stdout + "\n" + install.stderr
                    )[-4000:]
                result["install_attempts"].append(attempt)
                target_attempts.append(attempt)
                if not install.returncode:
                    installed = True
                    result["install_contract"]["editable_installed_targets"].append(
                        target
                    )
                    result["install_contract"].setdefault(
                        "repair_modes", [],
                    ).append(mode)
                    break
            if not installed:
                raise ValidationError(
                    f"required editable install failed for {target}"
                )

        if result["install_fallback_targets"]:
            result["install_contract"]["status"] = "fallback_unverified"
        elif result["install_contract"]["requested_targets"]:
            result["install_contract"]["status"] = "editable_installed"
        else:
            result["install_contract"]["status"] = "preinstalled_only"
            # Some successful H3 contracts intentionally set
            # ``install_project=false`` for source-tree applications that do
            # not publish Python package metadata.  Their authoritative test
            # command is ``python -m pytest`` from the checkout root, which
            # exposes that root on sys.path.  Our JSON argument runner lives
            # under /tmp, so reproduce that Python invocation property
            # explicitly instead of incorrectly declaring imports missing.
            env["PYTHONPATH"] = _workspace_pythonpath(
                workspace, env.get("PYTHONPATH", ""),
            )
            result["install_contract"]["source_tree_pythonpath"] = [
                str(workspace), str(workspace / "src"),
            ]

        result["environment"], result["environment_sha256"] = capture_runtime_environment(spec)

        observed: dict[str, str] = {}
        # Context instrumentation adds CPU overhead but must not consume the
        # benchmark's original validation budget. Reserve extra wall time for
        # instrumentation and, only if the wrapper itself breaks, one direct
        # fail-open execution of the affected chunk.
        deadline_multiplier = 3 if context_data_output is not None else 1
        deadline = time.monotonic() + max(1, suite_timeout_s) * deadline_multiplier
        addopts = list(spec.get("pytest_addopts") or [])
        pytest_command = list(spec.get("pytest_command") or [])
        if not pytest_command:
            expected_environment = prepared_dir / "environment.expected.json"
            if expected_environment.is_file():
                pytest_command = list(
                    json.loads(expected_environment.read_text(encoding="utf-8"))
                    .get("pytest_command") or []
                )
        if pytest_command[:3] in (["python", "-m", "pytest"], ["python3", "-m", "pytest"]):
            pytest_command = pytest_command[3:]
        elif pytest_command[:1] in (["pytest"], ["py.test"]):
            pytest_command = pytest_command[1:]
        requested_behavior_nodes: list[str] | None = None
        behavior_plan: dict[str, Any] = {}
        behavior_nodes = []
        behavior_scope = "all_stable"
        if runtime_behavior_output is not None:
            if runtime_behavior_node_ids_file is not None:
                selected = json.loads(
                    runtime_behavior_node_ids_file.read_text(encoding="utf-8")
                )
                if isinstance(selected, dict):
                    behavior_plan = dict(selected)
                    plan_snapshot = str(behavior_plan.get("snapshot_id") or "")
                    if plan_snapshot and plan_snapshot != str(spec["snapshot_id"]):
                        raise ValueError(
                            "runtime behavior plan snapshot identity mismatch"
                        )
                    selected = (
                        behavior_plan.get("node_ids")
                        or behavior_plan.get("required_node_ids")
                        or behavior_plan.get("target_node_ids")
                    )
                if not isinstance(selected, list) or not all(
                    isinstance(node, str) for node in selected
                ):
                    raise ValueError(
                        "runtime behavior node IDs must be a JSON string list "
                        "or a plan object containing node_ids"
                    )
                if not set(selected).issubset(passed_ptp):
                    raise ValueError(
                        "runtime behavior nodes are not a subset of ground truth"
                    )
                requested_behavior_nodes = list(dict.fromkeys(selected))
                behavior_scope = "targeted"
            behavior_workspace_root = Path(tempfile.mkdtemp(
                prefix="taskforge-behavior-pristine-", dir="/tmp",
            ))
            _copy_runtime_pristine(
                workspace, behavior_workspace_root / "workspace",
            )
        with tempfile.TemporaryDirectory(prefix="taskforge-reports-", dir="/tmp") as report_temp:
            report_root = Path(report_temp)
            context_active = context_data_output is not None and _context_runtime_available()
            # A normal benchmark run must preserve pytest's session/module
            # fixture lifecycle and ordering. The old file-sized chunks could
            # repeatedly initialize session fixtures or run dependent tests in
            # different processes, manufacturing hundreds of ``missing``
            # outcomes. The JSON argument runner removes the OS argv limit, so
            # only the optional coverage-context calibration remains chunked.
            if execution_contract == "published_test_commands_v1":
                authenticated_chunks = published_pytest_chunks(
                    draft.get("test_commands") or [], passed_ptp,
                )
                if pytest_session_policy == "single_session":
                    frozen_prefix = list(authenticated_chunks[0][0])
                    pytest_chunks = [(
                        [*frozen_prefix, *h3_pytest_args], passed_ptp,
                    )]
                else:
                    pytest_chunks = authenticated_chunks
            elif execution_contract == "single_session_v1":
                node_chunks = list(_chunks(passed_ptp)) if context_active else [passed_ptp]
                default_prefix = [
                    "-q", "--no-header",
                    "-o", "addopts=" + shlex.join(addopts),
                    *pytest_command, *h3_pytest_args,
                ]
                pytest_chunks = [(default_prefix, nodes) for nodes in node_chunks]
            else:
                raise ValidationError(
                    f"unsupported pytest execution contract: {execution_contract}"
                )
            result["pytest_command_count"] = len(pytest_chunks)
            result["timing"]["pytest_planned_command_count"] = len(pytest_chunks)
            pytest_suite_started = time.monotonic()
            context_append = False
            context_failure = ""
            runner = _write_context_runner(report_root) if context_active else None
            direct_runner = _write_direct_runner(report_root)
            if context_data_output is not None:
                context_data_output.parent.mkdir(parents=True, exist_ok=True)
                context_data_output.unlink(missing_ok=True)
                result["coverage_context"]["status"] = (
                    "recording" if context_active else "coverage_unavailable"
                )
            for index, (command_prefix, nodes) in enumerate(pytest_chunks):
                active_pytest_timing = {
                    "command_index": index,
                    "requested_node_count": len(nodes),
                    "wall_seconds": None,
                    "returncode": None,
                    "missing_node_count": None,
                    "selector_recovery_rounds": 0,
                    "timed_out": False,
                    "status": "running",
                    "_started": time.monotonic(),
                }
                remaining = int(deadline - time.monotonic())
                if remaining <= 0:
                    raise ValidationError("validation suite timeout")
                report = report_root / f"report-{index}.json"
                outcome_report = report_root / f"outcomes-{index}.json"
                pytest_arguments = [
                    *command_prefix,
                    "--json-report", f"--json-report-file={report}",
                    *nodes,
                ]
                arguments_file = report_root / f"arguments-{index}.json"
                arguments_file.write_text(
                    json.dumps(pytest_arguments), encoding="utf-8",
                )
                argv = _direct_pytest_argv(
                    runner=direct_runner, arguments_file=arguments_file,
                )
                # The authoritative pass is always uninstrumented. Runtime
                # evidence is collected in a second pass and compared below.
                run_env = env.copy()
                run_env["TASKFORGE_PYTEST_WORKSPACE"] = str(workspace)
                run_env["TASKFORGE_PYTEST_OUTCOME_FILE"] = str(outcome_report)
                # PYTHONSAFEPATH=1 prevents Python from adding the runner's
                # directory to sys.path.  Always expose the generated pytest
                # plugins explicitly, not only when coverage is enabled.
                run_env["PYTHONPATH"] = _published_pytest_pythonpath(
                    workspace, report_root, env.get("PYTHONPATH", ""),
                )
                if context_active:
                    run_env["COVERAGE_FILE"] = str(context_data_output)
                    argv = _context_pytest_argv(
                        runner=runner, arguments_file=arguments_file,
                        workspace=workspace, append=context_append,
                    )
                try:
                    execution_timeout = (
                        min(max(1, remaining), max(1, command_timeout_s * 2))
                        if context_active else max(1, remaining)
                    )
                    process = _run(
                        argv, cwd=workspace, env=run_env,
                        timeout=execution_timeout,
                    )
                except ValidationError as exc:
                    if not context_active:
                        raise
                    context_failure = str(exc)
                    context_active = False
                    outcome_report.unlink(missing_ok=True)
                    process = _run(
                        _direct_pytest_argv(
                            runner=direct_runner, arguments_file=arguments_file,
                        ),
                        cwd=workspace, env=run_env,
                        timeout=min(
                            max(1, int(deadline - time.monotonic())),
                            max(1, command_timeout_s),
                        ),
                    )
                # Context evidence must never turn a valid benchmark into a
                # rejected one. If instrumentation itself fails before pytest
                # can report, finish this chunk once without instrumentation
                # and retain an explicit incomplete sidecar status.
                if context_active and not report.is_file():
                    context_failure = (
                        f"coverage wrapper exit {process.returncode}: "
                        + (process.stdout + process.stderr)[-2000:]
                    )
                    context_active = False
                    outcome_report.unlink(missing_ok=True)
                    process = _run(
                        _direct_pytest_argv(
                            runner=direct_runner, arguments_file=arguments_file,
                        ),
                        cwd=workspace, env=run_env,
                        timeout=min(max(1, remaining), max(1, command_timeout_s)),
                    )
                # One stale parametrized selector makes pytest exit 4 before
                # it executes any of the other requested tests. Remove only
                # selectors named by pytest's exact ``ERROR: not found``
                # diagnostic, then rerun the survivors together so session
                # fixtures and ordering retain normal benchmark semantics.
                selector_nodes = list(nodes)
                selector_rejected: list[str] = []
                selector_attempts: list[dict[str, Any]] = []
                for recovery_round in range(1, 9):
                    if context_active or process.returncode != 4:
                        break
                    rejected = pytest_not_found_nodeids(
                        process.stdout + "\n" + process.stderr,
                        selector_nodes, workspace=workspace,
                    )
                    if not rejected or len(rejected) == len(selector_nodes):
                        break
                    selector_attempts.append({
                        "round": recovery_round,
                        "requested_count": len(selector_nodes),
                        "rejected_count": len(rejected),
                        "rejected_sample": rejected[:20],
                        "returncode": process.returncode,
                        "output_tail": (
                            process.stdout + "\n" + process.stderr
                        )[-4000:],
                    })
                    selector_rejected.extend(rejected)
                    rejected_set = set(rejected)
                    selector_nodes = [
                        node for node in selector_nodes
                        if node not in rejected_set
                    ]
                    report.unlink(missing_ok=True)
                    outcome_report.unlink(missing_ok=True)
                    pytest_arguments = [
                        *command_prefix,
                        "--json-report", f"--json-report-file={report}",
                        *selector_nodes,
                    ]
                    arguments_file.write_text(
                        json.dumps(pytest_arguments), encoding="utf-8",
                    )
                    remaining = int(deadline - time.monotonic())
                    if remaining <= 0:
                        raise ValidationError("validation suite timeout")
                    process = _run(
                        _direct_pytest_argv(
                            runner=direct_runner,
                            arguments_file=arguments_file,
                        ),
                        cwd=workspace, env=run_env,
                        timeout=max(1, remaining),
                    )
                if selector_attempts:
                    result["selector_recovery"].append({
                        "chunk_index": index,
                        "attempts": selector_attempts,
                        "rejected_count": len(set(selector_rejected)),
                        "rejected_nodeids": list(dict.fromkeys(
                            selector_rejected,
                        )),
                        "survivor_count": len(selector_nodes),
                        "final_returncode": process.returncode,
                    })
                if not report.is_file():
                    raise ValidationError(
                        f"pytest did not create structured report (exit {process.returncode}): "
                        + (process.stdout + process.stderr)[-4000:]
                    )
                payload = json.loads(report.read_text(encoding="utf-8"))
                raw_chunk_observed: dict[str, str] = {}
                for item in payload.get("tests") or []:
                    nodeid = str(item.get("nodeid") or "").strip()
                    if not nodeid:
                        continue
                    outcome = normalize_pytest_outcome(item.get("outcome"))
                    raw_chunk_observed[nodeid] = merge_pytest_outcome(
                        raw_chunk_observed.get(nodeid), outcome,
                    )
                canonical_observed: dict[str, str] = {}
                if outcome_report.is_file():
                    try:
                        outcome_payload = json.loads(
                            outcome_report.read_text(encoding="utf-8"),
                        )
                        for nodeid, value in (
                            outcome_payload.get("outcomes") or {}
                        ).items():
                            canonical_observed[str(nodeid)] = (
                                normalize_pytest_outcome(value)
                            )
                    except (OSError, json.JSONDecodeError, AttributeError):
                        canonical_observed = {}
                chunk_observed, canonical_remapped = reconcile_pytest_nodeids(
                    nodes, canonical_observed,
                )
                remaining_nodes = [
                    node for node in nodes if node not in chunk_observed
                ]
                fallback_observed, remapped = reconcile_pytest_nodeids(
                    remaining_nodes, raw_chunk_observed,
                )
                for nodeid, outcome in fallback_observed.items():
                    chunk_observed[nodeid] = merge_pytest_outcome(
                        chunk_observed.get(nodeid), outcome,
                    )
                remapped = {**canonical_remapped, **remapped}
                for nodeid, outcome in chunk_observed.items():
                    observed[nodeid] = merge_pytest_outcome(
                        observed.get(nodeid), outcome,
                    )
                missing_count = len(set(nodes) - set(chunk_observed))
                if process.returncode or missing_count:
                    result["pytest_diagnostics"].append(pytest_diagnostic(
                        payload=payload, process=process, nodes=nodes,
                        observed=chunk_observed, chunk_index=index,
                        raw_observed=raw_chunk_observed, remapped=remapped,
                    ))
                if context_active:
                    context_append = True
                    result["coverage_context"]["chunks_recorded"] += 1
                    try:
                        current_size = context_data_output.stat().st_size
                    except OSError:
                        current_size = 0
                    if (
                        current_size > max(1, context_max_bytes)
                        and index + 1 < len(pytest_chunks)
                    ):
                        context_failure = (
                            f"context data exceeded {context_max_bytes} bytes after "
                            f"chunk {index + 1}/{len(chunks)}"
                        )
                        context_active = False

                active_pytest_timing.update({
                    "wall_seconds": round(
                        time.monotonic() - active_pytest_timing["_started"], 3,
                    ),
                    "returncode": process.returncode,
                    "missing_node_count": missing_count,
                    "selector_recovery_rounds": len(selector_attempts),
                    "structured_report_available": report.is_file() or outcome_report.is_file(),
                    "status": "completed",
                })
                active_pytest_timing.pop("_started", None)
                result["timing"]["pytest_commands"].append(active_pytest_timing)
                result["timing"]["pytest_completed_command_count"] += 1
                active_pytest_timing = None

            pytest_suite_finished = time.monotonic()

            if context_data_output is not None:
                context = result["coverage_context"]
                if context_data_output.is_file():
                    context["data_bytes"] = context_data_output.stat().st_size
                    context["data_sha256"] = _sha256_file(context_data_output)
                if context_failure:
                    context["status"] = "partial"
                    context["reason"] = context_failure
                elif (
                    context["chunks_recorded"] == len(pytest_chunks)
                    and pytest_chunks
                ):
                    context["status"] = "recorded"
                    context["complete"] = True
                elif context["status"] == "recording":
                    context["status"] = "no_chunks"

        expected_set = set(passed_ptp)
        result["pipeline_stage"] = "pytest_finished"
        outcome_counts: dict[str, int] = {}
        for node in passed_ptp:
            outcome = observed.get(node, "missing")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        result["outcome_counts"] = dict(sorted(outcome_counts.items()))
        result["passed"] = sum(observed.get(node) == "passed" for node in passed_ptp)
        result["failed"] = sum(observed.get(node) == "failed" for node in passed_ptp)
        result["skipped"] = sum(observed.get(node) == "skipped" for node in passed_ptp)
        result["errors"] = len(expected_set - set(observed)) + sum(
            observed.get(node) not in {"passed", "failed", "skipped"} for node in passed_ptp
            if node in observed
        )
        result["non_passed"] = [
            {"nodeid": node, "outcome": observed.get(node, "missing")}
            for node in passed_ptp if observed.get(node) != "passed"
        ]
        result["accepted_passed_ptp"] = [
            node for node in passed_ptp if observed.get(node) == "passed"
        ]
        if runtime_behavior_output is not None:
            accepted_behavior_denominator = list(result["accepted_passed_ptp"])
            if requested_behavior_nodes is None:
                behavior_nodes = accepted_behavior_denominator
            else:
                # A targeted recollection plan is pinned to the historical
                # scored denominator.  A node that fails the new pristine run
                # must surface as a baseline mismatch; silently dropping it
                # would make a broken task appear complete by shrinking the
                # denominator.
                behavior_nodes = list(requested_behavior_nodes)
                result["runtime_behavior_evidence"][
                    "dropped_nonaccepted_node_count"
                ] = 0
            if not behavior_nodes:
                raise ValidationError(
                    "runtime behavior selection is empty after final validation"
                )
            expected_behavior_identity = runtime_behavior_identity(
                snapshot_id=str(spec.get("snapshot_id") or ""),
                image_digest=str(os.environ.get("TASKFORGE_IMAGE_DIGEST", "")),
                environment_sha256=str(result.get("environment_sha256") or ""),
                accepted_node_ids=accepted_behavior_denominator,
                selected_node_ids=behavior_nodes,
                collector_contract_sha256=str(
                    collector_identity.get("collector_contract_sha256") or ""
                ),
            )
            result["runtime_behavior_evidence"].update({
                "scope": behavior_scope,
                "selected_node_count": len(behavior_nodes),
                "accepted_denominator_count": len(accepted_behavior_denominator),
            })
        result["pass_rate"] = round(
            result["passed"] / len(passed_ptp), 6,
        ) if passed_ptp else 0.0
        result["ground_truth_adjusted"] = bool(result["non_passed"])
        observed_values = sorted(f"{node}\0{outcome}" for node, outcome in observed.items())
        result["observed_node_ids_sha256"] = hashlib.sha256(
            ("\n".join(observed_values) + "\n").encode()
        ).hexdigest()
        # DeNovo first accepts a repository environment at >=90% baseline
        # pass rate, then publishes only the pass-to-pass node IDs.  This outer
        # run is the authoritative final environment, so a small reproducible
        # difference from the H3 preflight narrows ground truth rather than
        # discarding an otherwise valid repository.  Every excluded node is
        # retained above as provenance and never silently enters scoring.
        if final_ground_truth_accepted(
            passed=result["passed"], expected=len(passed_ptp),
            min_pass_rate=min_pass_rate, min_passed_tests=min_passed_tests,
        ):
            result.update(status="validated", reason="")
            result["delivery_evidence"] = capture_delivery_evidence(
                {**draft, "passed_ptp": result["accepted_passed_ptp"]},
                workspace=workspace, env=env, timeout=command_timeout_s,
                script=delivery_evidence_script,
            )
            result["delivery_evidence"]["identity"] = {
                "snapshot_id": str(result.get("snapshot_id") or ""),
                "image_digest": str(result.get("image_digest") or ""),
                "environment_sha256": str(result.get("environment_sha256") or ""),
                "collector_contract_sha256": str(
                    collector_identity.get("collector_contract_sha256") or ""
                ),
            }
            owned_roots = list(
                result["delivery_evidence"].get("selected_package_roots") or []
            )
            owned_discovery = {
                "schema_version": 1,
                "status": str(result["delivery_evidence"].get("status") or ""),
                "source": "distribution_owned_delivery_evidence",
                "selected_import_names": owned_roots,
                "rejected_roots": list(
                    result["delivery_evidence"].get("rejected_roots") or []
                ),
            }
            if not owned_roots:
                # Some historical records put the distribution spelling in
                # ``import_names`` (``agensgraph_python``) while the actual
                # package imported by tests is ``agensgraph``. Delivery still
                # verifies every scored import, but may have no package root
                # whose installed metadata owner matches that stale spelling.
                # Reuse the runtime-surface reconciler, which combines stable
                # test imports, workspace packages, and distribution hints,
                # instead of dropping all behavior evidence.
                owned_roots, fallback_discovery = discover_runtime_behavior_roots(
                    draft, workspace=workspace, script=runtime_api_script,
                )
                owned_discovery = {
                    **fallback_discovery,
                    "source": "runtime_surface_fallback_after_empty_delivery_roots",
                }
                if owned_roots:
                    # Persist the reconciled roots in the delivery sidecar as
                    # well.  Otherwise behavior capture succeeds but the
                    # downstream delivery contract still sees an empty public
                    # package and parks a document whose scored imports were
                    # actually verified.
                    result["delivery_evidence"]["selected_package_roots"] = list(
                        owned_roots
                    )
                    result["delivery_evidence"]["root_discovery_fallback"] = dict(
                        owned_discovery
                    )
            if runtime_behavior_output is not None:
                try:
                    behavior_workspace = (
                        behavior_workspace_root / "workspace"
                        if behavior_workspace_root is not None else workspace
                    )
                    behavior_env = env.copy()
                    behavior_env["HOME"] = str(
                        behavior_workspace_root / "home"
                        if behavior_workspace_root is not None else Path("/tmp")
                    )
                    Path(behavior_env["HOME"]).mkdir(parents=True, exist_ok=True)
                    behavior_env["PYTHONPATH"] = _workspace_pythonpath(
                        behavior_workspace,
                        behavior_env.get("PYTHONPATH", ""),
                    )
                    capture_runtime_behavior_evidence(
                        output=runtime_behavior_output,
                        workspace=behavior_workspace,
                        env=behavior_env,
                        draft=draft,
                        identity=expected_behavior_identity,
                        selected_nodes=behavior_nodes,
                        baseline_outcomes={
                            node: str(
                                dict(behavior_plan.get("authoritative_outcomes") or {})
                                .get(node, observed.get(node, "missing"))
                            )
                            for node in behavior_nodes
                        },
                        pytest_command=pytest_command,
                        addopts=addopts,
                        timeout=max(command_timeout_s, suite_timeout_s),
                        scope=behavior_scope,
                        import_roots=owned_roots,
                        root_discovery=owned_discovery,
                        max_nodes_per_shard=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "max_nodes_per_shard", 20,
                            )
                        ),
                        shard_timeout=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "shard_timeout_s",
                                max(command_timeout_s, suite_timeout_s),
                            )
                        ),
                        shard_grouping=str(
                            dict(behavior_plan.get("policy") or {}).get(
                                "grouping", "test_file_then_node_count",
                            )
                        ),
                        max_symbols_per_node=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "max_symbols_per_node", 256,
                            )
                        ),
                        max_events_per_node=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "max_events_per_node", 512,
                            )
                        ),
                        max_events_per_symbol=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "max_events_per_symbol", 2,
                            )
                        ),
                        max_event_bytes=int(
                            dict(behavior_plan.get("policy") or {}).get(
                                "max_event_bytes", 2000,
                            )
                        ),
                        trace_mode=str(
                            dict(behavior_plan.get("policy") or {}).get(
                                "trace_mode", "bounded_value",
                            )
                        ),
                        generation=str(
                            behavior_plan.get("generation")
                            or "spec-runtime-v5-targeted-repair1"
                        ),
                        collector_contract_sha256=str(
                            collector_identity.get(
                                "collector_contract_sha256"
                            ) or ""
                        ),
                    )
                except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
                    result["runtime_behavior_evidence"].update({
                        "status": "capture_failed",
                        "reason": str(exc)[:4000],
                    })
            # Reuse this already-started, already-installed final task image to
            # collect H5 public-surface evidence.  Reflection is deliberately
            # fail-open: its status is recorded for Specification triage but
            # can never invalidate a repository whose tests passed.
            try:
                runtime_module_names = sorted({
                    str(row.get("module") or "")
                    for row in result["delivery_evidence"].get("scored_imports") or []
                    if row.get("status") == "resolved" and row.get("module")
                })
                runtime_api = capture_runtime_api_inventory(
                    {
                        **draft,
                        "import_names": owned_roots or draft.get("import_names"),
                        "runtime_module_names": runtime_module_names,
                    },
                    workspace=workspace, env=env, timeout=command_timeout_s,
                    script=runtime_api_script,
                )
            except Exception as exc:  # evidence capture must remain fail-open
                runtime_api = {
                    "schema_version": 1,
                    "status": "unavailable",
                    "reason": f"runtime_surface_unexpected_failure: {exc}",
                    "symbols": [],
                    "symbol_count": 0,
                    "import_errors": [],
                }
            runtime_api["artifact"] = os.environ.get(
                "TASKFORGE_RUNTIME_API_ARTIFACT", "",
            )
            result["runtime_api_inventory"] = runtime_api
            result["install_contract"]["distribution_records"] = list(
                result["delivery_evidence"].get("distribution_records") or []
            )
            result["install_contract"]["pytest_entry_points"] = [
                entry
                for distribution in result["install_contract"]["distribution_records"]
                for entry in distribution.get("entry_points") or []
                if entry.get("group") == "pytest11"
            ]
        else:
            result["reason"] = "final_pass_rate_below_floor"
    except (OSError, ValueError, ValidationError, json.JSONDecodeError) as exc:
        result["reason"] = str(exc)[:8000]
    timing = result["timing"]
    if active_pytest_timing is not None:
        reason = str(result.get("reason") or "")
        active_pytest_timing.update({
            "wall_seconds": round(
                time.monotonic() - active_pytest_timing["_started"], 3,
            ),
            "timed_out": "timeout" in reason.lower(),
            "status": "timed_out" if "timeout" in reason.lower() else "aborted",
        })
        active_pytest_timing.pop("_started", None)
        timing["pytest_commands"].append(active_pytest_timing)
    if pytest_suite_started is not None:
        pytest_end = pytest_suite_finished or time.monotonic()
        timing["pytest_wall_seconds"] = round(pytest_end - pytest_suite_started, 3)
    command_durations = [
        float(row.get("wall_seconds") or 0.0)
        for row in timing["pytest_commands"]
    ]
    timing["pytest_max_command_wall_seconds"] = round(
        max(command_durations, default=0.0), 3,
    )
    timing["pytest_timed_out"] = any(
        bool(row.get("timed_out")) for row in timing["pytest_commands"]
    )
    if not result["environment"]:
        result["environment"], result["environment_sha256"] = capture_runtime_environment(spec)
    if result["runtime_api_inventory"].get("status") == "pending":
        result["runtime_api_inventory"].update(
            status="not_requested",
            reason="outer_validation_not_validated",
        )
    if result["delivery_evidence"].get("status") == "pending":
        result["delivery_evidence"].update(
            status="not_requested", reason="outer_validation_not_validated",
        )
    # Bind the sidecar to the exact validation identity.  The digest lives in
    # result.json (outside the sidecar) so it can authenticate the complete
    # canonical sidecar payload without a self-referential hash.
    result["runtime_api_inventory"].update({
        "snapshot_id": str(result.get("snapshot_id") or ""),
        "image_digest": str(result.get("image_digest") or ""),
        "environment_sha256": str(result.get("environment_sha256") or ""),
        "collector_contract_sha256": str(
            collector_identity.get("collector_contract_sha256") or ""
        ),
    })
    runtime_payload = json.dumps(
        result["runtime_api_inventory"], sort_keys=True, separators=(",", ":"),
    ).encode()
    result["runtime_api_inventory_sha256"] = hashlib.sha256(runtime_payload).hexdigest()
    delivery_payload = json.dumps(
        result["delivery_evidence"], sort_keys=True, separators=(",", ":"),
    ).encode()
    result["delivery_evidence_sha256"] = hashlib.sha256(
        delivery_payload
    ).hexdigest()
    if runtime_behavior_output is not None:
        behavior_state = result["runtime_behavior_evidence"]
        if runtime_behavior_output.is_file():
            try:
                behavior_payload = json.loads(
                    runtime_behavior_output.read_text(encoding="utf-8")
                )
                actual_identity = dict(behavior_payload.get("identity") or {})
                if actual_identity != expected_behavior_identity:
                    raise ValueError(
                        "runtime behavior identity does not match validation run"
                    )
                behavior_state.update({
                    "status": str(behavior_payload.get("status") or "captured"),
                    "executed_selected_node_count": int(
                        behavior_payload.get("executed_selected_node_count") or 0
                    ),
                    "observed_node_count": int(
                        behavior_payload.get("observed_node_count") or 0
                    ),
                    "boundary_node_count": int(
                        behavior_payload.get("boundary_node_count") or 0
                    ),
                    "event_node_count": int(
                        behavior_payload.get("event_node_count") or 0
                    ),
                    "verified_node_count": int(
                        behavior_payload.get("verified_node_count") or 0
                    ),
                    "truncated_node_count": int(
                        behavior_payload.get("truncated_node_count") or 0
                    ),
                    "sha256": _sha256_file(runtime_behavior_output),
                })
                if (
                    behavior_state["executed_selected_node_count"]
                    != behavior_state["selected_node_count"]
                ):
                    behavior_payload["status"] = "incomplete_execution"
                    temporary_behavior = runtime_behavior_output.with_suffix(
                        runtime_behavior_output.suffix + ".tmp"
                    )
                    temporary_behavior.write_text(
                        json.dumps(
                            behavior_payload, indent=2, sort_keys=True,
                        ) + "\n",
                        encoding="utf-8",
                    )
                    os.replace(temporary_behavior, runtime_behavior_output)
                    behavior_state.update(
                        status="incomplete_execution",
                        reason=(
                            "not every selected runtime behavior node executed"
                        ),
                        sha256=_sha256_file(runtime_behavior_output),
                    )
            except (OSError, ValueError) as exc:
                behavior_state.update(
                    status="invalid", reason=str(exc)[:1000],
                )
        else:
            # Preserve the actionable exception recorded by the capture pass.
            # Replacing it with a generic "did not write sidecar" made a
            # deterministic plugin/import failure indistinguishable from an
            # absent artifact and prevented targeted repair.
            if behavior_state.get("status") not in {"capture_failed", "invalid"}:
                behavior_state.update(
                    status="missing", reason="pytest did not write sidecar",
                )
    behavior_covers_complete_denominator = runtime_behavior_covers_denominator(
        behavior_nodes,
        list(result.get("accepted_passed_ptp") or []),
        requested=runtime_behavior_output is not None,
    )
    result["runtime_behavior_evidence"][
        "covers_complete_accepted_denominator"
    ] = behavior_covers_complete_denominator
    result["specification_evidence_ready"] = bool(
        result.get("status") == "validated"
        and result.get("install_contract", {}).get("status")
        in {"editable_installed", "preinstalled_only"}
        and result.get("runtime_api_inventory", {}).get("status") == "captured"
        and result.get("delivery_evidence", {}).get("status") == "captured"
        and result.get("delivery_evidence", {}).get(
            "scored_imports_runtime_verified"
        ) is True
        and not int(
            result.get("delivery_evidence", {}).get(
                "required_unresolved_scored_import_count"
            ) or 0
        )
        and (
            not result.get("runtime_behavior_evidence", {}).get("requested")
            or (
                result.get("runtime_behavior_evidence", {}).get("status")
                == "captured_complete"
                and behavior_covers_complete_denominator
            )
        )
    )
    result["duration_seconds"] = round(time.monotonic() - started, 3)
    result["timing"]["total_wall_seconds"] = result["duration_seconds"]
    if behavior_workspace_root is not None:
        shutil.rmtree(behavior_workspace_root, ignore_errors=True)
    if runtime_api_output is not None:
        runtime_api_output.parent.mkdir(parents=True, exist_ok=True)
        runtime_temporary = runtime_api_output.with_suffix(runtime_api_output.suffix + ".tmp")
        runtime_temporary.write_text(
            json.dumps(result["runtime_api_inventory"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(runtime_temporary, runtime_api_output)
    if delivery_evidence_output is not None:
        delivery_evidence_output.parent.mkdir(parents=True, exist_ok=True)
        delivery_temporary = delivery_evidence_output.with_suffix(
            delivery_evidence_output.suffix + ".tmp"
        )
        delivery_temporary.write_text(
            json.dumps(result["delivery_evidence"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        os.replace(delivery_temporary, delivery_evidence_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clean-script", type=Path, default=Path("/opt/taskforge_clean.sh"))
    parser.add_argument("--submission-patch", type=Path)
    parser.add_argument("--submission-dir", type=Path)
    parser.add_argument("--suite-timeout", type=int, default=1800)
    parser.add_argument("--command-timeout", type=int, default=900)
    parser.add_argument("--min-pass-rate", type=float, default=0.90)
    parser.add_argument("--min-passed-tests", type=int, default=10)
    parser.add_argument("--context-data-output", type=Path)
    parser.add_argument("--runtime-api-output", type=Path)
    parser.add_argument("--delivery-evidence-output", type=Path)
    parser.add_argument("--runtime-behavior-output", type=Path)
    parser.add_argument("--runtime-behavior-node-ids-file", type=Path)
    parser.add_argument("--runtime-api-script", type=Path, default=Path("/opt/taskforge_runtime_surface.py"))
    parser.add_argument("--delivery-evidence-script", type=Path, default=Path("/opt/taskforge_delivery_evidence.py"))
    parser.add_argument("--collector-manifest", type=Path)
    parser.add_argument("--context-max-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--node-ids-file", type=Path)
    parser.add_argument(
        "--pytest-session-policy",
        choices=("contract", "single_session"), default="contract",
    )
    args = parser.parse_args(argv)
    result = validate(
        prepared_dir=args.prepared_dir, output=args.output, clean_script=args.clean_script,
        submission_patch=args.submission_patch,
        submission_dir=args.submission_dir,
        suite_timeout_s=args.suite_timeout, command_timeout_s=args.command_timeout,
        min_pass_rate=args.min_pass_rate, min_passed_tests=args.min_passed_tests,
        context_data_output=args.context_data_output,
        runtime_api_output=args.runtime_api_output,
        delivery_evidence_output=args.delivery_evidence_output,
        runtime_behavior_output=args.runtime_behavior_output,
        runtime_behavior_node_ids_file=args.runtime_behavior_node_ids_file,
        runtime_api_script=args.runtime_api_script,
        delivery_evidence_script=args.delivery_evidence_script,
        collector_manifest=args.collector_manifest,
        context_max_bytes=args.context_max_bytes,
        node_ids_file=args.node_ids_file,
        pytest_session_policy=args.pytest_session_policy,
    )
    runtime_api = result.get("runtime_api_inventory") or {}
    print("TASKFORGE_VALIDATION " + json.dumps({
        "duration_seconds": result.get("duration_seconds"),
        "pytest_wall_seconds": (result.get("timing") or {}).get(
            "pytest_wall_seconds"
        ),
        "pytest_timed_out": (result.get("timing") or {}).get(
            "pytest_timed_out"
        ),
        "failed": result.get("failed"),
        "errors": result.get("errors"),
        "image_digest": result.get("image_digest"),
        "pass_rate": result.get("pass_rate"),
        "passed": result.get("passed"),
        "reason": result.get("reason"),
        "non_passed_preview": list(result.get("non_passed") or [])[:10],
        "runtime_api_status": runtime_api.get("status"),
        "runtime_api_symbols": runtime_api.get("symbol_count"),
        "runtime_api_truncated": runtime_api.get("truncated"),
        "skipped": result.get("skipped"),
        "snapshot_id": result.get("snapshot_id"),
        "status": result.get("status"),
    }, sort_keys=True))
    # A deterministic validation failure is a successfully completed audit,
    # not an infrastructure failure.  Returning zero lets the controller retry
    # only jobs that died before they could write a result (pull/OOM/PAI).
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
