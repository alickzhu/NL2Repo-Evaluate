"""Durable attempt ownership and append-only run records."""

from __future__ import annotations

import fcntl
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import DatasetTask, GradeResult, json_value


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must match {_IDENTIFIER.pattern}")
    return value


def atomic_json(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    """Publish one JSON document with fsync and directory durability."""
    data = json.dumps(json_value(payload), ensure_ascii=False, indent=2) + "\n"
    _atomic_lines(path, [data], mode)


def atomic_jsonl(
    path: Path, payloads: Iterable[dict[str, Any]], *, mode: int = 0o600
) -> None:
    """Replace a derived JSONL export atomically without duplicating targets."""
    lines = (
        json.dumps(json_value(payload), ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for payload in payloads
    )
    _atomic_lines(path, lines, mode)


def _atomic_lines(path: Path, lines: Iterable[str], mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.writelines(lines)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any], *, mode: int = 0o600) -> None:
    """Append one complete JSON line while holding an advisory process lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(json_value(payload), ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, mode)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if os.write(descriptor, encoded) != len(encoded):
            raise OSError("short append to JSONL ledger")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class AttemptPaths:
    run_root: Path
    run_id: str
    task_key: str
    task_id: str
    sample_index: int
    attempt_index: int
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "state.json"

    @property
    def lifecycle(self) -> Path:
        return self.root / "lifecycle.jsonl"

    @property
    def events(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def completions(self) -> Path:
        return self.root / "completions"

    @property
    def controller_workspace(self) -> Path:
        return self.root / "controller-workspace"

    @property
    def conversation(self) -> Path:
        return self.root / "conversation"

    @property
    def private(self) -> Path:
        return self.root / "private"


class RunLedger:
    """Run-owned storage with atomic attempt reservation and result commits."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_id = _identifier(run_id, "run_id")
        self.root = Path(output_root).resolve() / self.run_id
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def publish_config(self, payload: dict[str, Any]) -> None:
        path = self.root / "run.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != json_value(payload):
                raise ValueError("run configuration differs from the existing run")
            return
        atomic_json(path, payload)

    def reserve(self, task: DatasetTask, attempt_index: int) -> AttemptPaths:
        if type(attempt_index) is not int or attempt_index < 0:
            raise ValueError("attempt_index must be a nonnegative integer")
        task_root = self.root / "tasks" / task.stable_key
        root = task_root / f"attempt-{attempt_index:04d}"
        root.mkdir(parents=True, exist_ok=False, mode=0o700)
        paths = AttemptPaths(
            run_root=self.root,
            run_id=self.run_id,
            task_key=task.stable_key,
            task_id=task.public.task_id,
            sample_index=task.public.sample_index,
            attempt_index=attempt_index,
            root=root,
        )
        for directory in (
            paths.completions,
            paths.controller_workspace,
            paths.conversation,
            paths.private,
        ):
            directory.mkdir(mode=0o700)
        atomic_json(paths.root / "task.json", task.public_dict())
        self.transition(paths, "reserved")
        return paths

    def transition(
        self, paths: AttemptPaths, status: str, **details: Any
    ) -> dict[str, Any]:
        status = _identifier(status, "status")
        event = {
            "utc": utc_now(),
            "status": status,
            "run_id": paths.run_id,
            "task_key": paths.task_key,
            "task_id": paths.task_id,
            "sample_index": paths.sample_index,
            "attempt_index": paths.attempt_index,
            "details": json_value(details),
        }
        append_jsonl(paths.lifecycle, event)
        atomic_json(paths.state, event)
        return event

    def commit_result(
        self,
        paths: AttemptPaths,
        *,
        outcome: str,
        grade: GradeResult,
        public_details: dict[str, Any],
    ) -> dict[str, Any]:
        if outcome not in {"completed", "failed", "timeout"}:
            raise ValueError("unsupported attempt outcome")
        result = {
            "schema_version": 1,
            "utc": utc_now(),
            "run_id": paths.run_id,
            "task_key": paths.task_key,
            "task_id": paths.task_id,
            "sample_index": paths.sample_index,
            "attempt_index": paths.attempt_index,
            "outcome": outcome,
            "grade": {
                name: value
                for name, value in asdict(grade).items()
                if name != "details"
            },
            "details": {
                **json_value(public_details, public=True),
                "all_tests_passed": grade.all_tests_passed,
                "score_capped": grade.status == "ok" and grade.passed > grade.total,
            },
        }
        atomic_json(paths.root / "result.json", result)
        append_jsonl(self.root / "results.jsonl", result)
        self.transition(paths, outcome, grade_status=grade.status)
        return result
