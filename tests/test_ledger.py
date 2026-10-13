from __future__ import annotations

import json

import pytest
from openhands_research.contracts import (
    DatasetTask,
    GradeResult,
    PrivateEvalSpec,
    PublicTaskSpec,
)
from openhands_research.ledger import RunLedger, append_jsonl, atomic_jsonl


@pytest.fixture
def task():
    return DatasetTask(
        PublicTaskSpec("fixture", "v1", "task", "Implement it.", "fixture:1"),
        PrivateEvalSpec("command", 1, ("true",), ()),
    )


def test_reservation_is_atomic_and_public(task, tmp_path):
    ledger = RunLedger(tmp_path, "run-1")
    paths = ledger.reserve(task, 0)
    assert paths.root.is_dir()
    public = json.loads((paths.root / "task.json").read_text())
    assert public == task.public_dict()
    assert "private" not in public
    assert json.loads(paths.state.read_text())["status"] == "reserved"
    with pytest.raises(FileExistsError):
        ledger.reserve(task, 0)


def test_run_config_is_immutable(tmp_path):
    ledger = RunLedger(tmp_path, "run")
    ledger.publish_config({"schema_version": 1, "profile": "smoke"})
    ledger.publish_config({"schema_version": 1, "profile": "smoke"})
    with pytest.raises(ValueError, match="differs"):
        ledger.publish_config({"schema_version": 1, "profile": "other"})


def test_result_separates_private_details(task, tmp_path):
    ledger = RunLedger(tmp_path, "run")
    paths = ledger.reserve(task, 0)
    result = ledger.commit_result(
        paths,
        outcome="completed",
        grade=GradeResult(
            "ok", reward=0.5, passed=1, total=2, details={"private": "SECRET"}
        ),
        public_details={"events": 7, "private": "SECRET"},
    )
    encoded = json.dumps(result)
    assert "SECRET" not in encoded
    assert result["grade"]["reward"] == 0.5
    lines = (ledger.root / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0]) == result
    assert json.loads(paths.state.read_text())["status"] == "completed"


def test_append_jsonl_writes_complete_lines(tmp_path):
    path = tmp_path / "events.jsonl"
    append_jsonl(path, {"index": 1, "text": "ü"})
    append_jsonl(path, {"index": 2})
    data = path.read_bytes()
    assert data.endswith(b"\n")
    assert [json.loads(line)["index"] for line in data.splitlines()] == [1, 2]


@pytest.mark.parametrize(
    "passed,total,failed,exit_code,clean",
    [
        (102, 101, 43, 1, False),
        (2, 2, 0, 0, True),
        (2, 2, 0, 1, False),
        (3, 2, 0, 0, False),
    ],
)
def test_clean_pass_does_not_follow_capped_reward(
    task, tmp_path, passed, total, failed, exit_code, clean
):
    ledger = RunLedger(tmp_path, "score")
    paths = ledger.reserve(task, 0)
    grade = GradeResult(
        "ok",
        reward=1.0,
        resolved=True,
        passed=passed,
        total=total,
        failed=failed,
        details={"commands": [{"exit_code": exit_code, "timed_out": False}]},
    )
    result = ledger.commit_result(
        paths, outcome="completed", grade=grade, public_details={}
    )
    assert result["grade"]["reward"] == 1.0
    assert result["grade"]["resolved"] is True
    assert result["details"]["all_tests_passed"] is clean
    assert result["details"]["score_capped"] is (passed > total)


def test_atomic_jsonl_reexport_replaces_without_partial_or_duplicate_rows(tmp_path):
    path = tmp_path / "sft.jsonl"
    rows = [{"response_id": "one"}, {"response_id": "two"}]
    atomic_jsonl(path, rows)
    atomic_jsonl(path, iter(rows))
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows
    with pytest.raises(ValueError):
        atomic_jsonl(path, [{"response_id": "new"}, {"bad": float("nan")}])
    assert [json.loads(line) for line in path.read_text().splitlines()] == rows


@pytest.mark.parametrize("commands", [None, "missing", {}, [None]])
def test_missing_command_evidence_is_not_a_clean_pass(commands):
    grade = GradeResult(
        "ok",
        reward=1.0,
        resolved=True,
        passed=1,
        total=1,
        details={"commands": commands},
    )
    assert not grade.all_tests_passed


@pytest.mark.parametrize("value", ["", "a/b", "../x", " space", "x" * 129])
def test_run_identifier_is_restricted(tmp_path, value):
    with pytest.raises(ValueError):
        RunLedger(tmp_path, value)
