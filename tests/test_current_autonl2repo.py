import copy
import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from openhands_research.benchmarks import current_autonl2repo as current


@pytest.fixture
def row():
    return {
        "instance_id": "owner__example--t2",
        "repo": "example",
        "pypi_name": "example-lib",
        "import_names": ["example"],
        "document": "Build Example exactly as specified.\r\n",
        "image": "docker.io/owner/example:task",
        "workdir": "/project/nested",
        "parent_commit": "a" * 40,
        "passed_ptp": ["tests/test_one.py::test_a", "tests/test_one.py::test_b"],
        "test_case_count": 2,
        "test_files": ["tests/test_one.py"],
        "test_patch": "PRIVATE\r\n",
        "test_commands": ["pytest tests/test_one.py"],
        "test_binary_archive_b64": "AP8NCg==",
        "test_binary_files": ["tests/data.bin"],
        "quality": {"new_field": "preserved"},
    }


def task(row):
    return current.task_from_row(row, current.evaluator_identity())


def result(task, passed=2):
    identity = task.private.payload["evaluator_identity"]
    return {
        "instance_id": task.public.task_id,
        "image_digest": "sha256:" + "f" * 64,
        "expected": 2,
        "passed": passed,
        "failed": 2 - passed,
        "skipped": 0,
        "score": passed / 2,
        "status": "scored",
        "scoring_valid": True,
        "evaluator_version": identity["version"],
        "canonical_runner_sha256": identity["files"]["_canonical_runner.py"],
        "timing": {
            "pytest_commands": [
                {"returncode": 0 if passed == 2 else 1, "timed_out": False}
            ]
        },
    }


def bundle():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name, content in {
            "workspace/impl.py": b"answer=42\r\n",
            "workspace/data.bin": b"\x00\xff",
            "workspace/.git/config": b"SECRET",
            "workspace/.venv/lib.py": b"CACHE",
        }.items():
            info = tarfile.TarInfo(name)
            info.mode, info.size = 0o644, len(content)
            tar.addfile(info, io.BytesIO(content))
    return stream.getvalue()


def test_exact_current_row_and_generic_isolation(row):
    before = copy.deepcopy(row)
    item = task(row)
    assert item.private.payload["task"] == before == row
    assert item.public.prompt == row["document"]
    assert item.public.workdir == "/workspace"
    assert item.private.payload["task"]["workdir"] == "/project/nested"
    assert item.environment.solver_image == "generic"
    assert not item.environment.grader_dependencies
    public = json.dumps(item.public_dict())
    assert (
        "PRIVATE" not in public
        and "test_one.py" not in public
        and "new_field" not in public
    )
    assert item.public.metadata["blocked_packages"] == ["example", "example-lib"]
    assert item.private.payload["evaluator_identity"]["version"] == "0.3.2"
    assert "current-" in item.public.dataset_revision


def test_task_identity_tracks_changes_even_when_id_unchanged(row):
    first = task(row)
    for key, value in (
        ("document", "Updated specification"),
        ("test_patch", "NEW\n"),
        ("quality", {"status": "updated"}),
    ):
        assert task({**row, key: value}).stable_key != first.stable_key


def test_load_current_flat_cohort_and_jsonl(row, tmp_path):
    batch = tmp_path / "batch_001"
    batch.mkdir()
    (batch / (row["instance_id"] + ".json")).write_text(json.dumps(row))
    (tmp_path / "manifest.json").write_text('{"count":1}')
    assert len(current.load_current(tmp_path)) == 1
    assert current.load_current(tmp_path)[0].private.payload["task"] == row
    source = tmp_path / "cohort.jsonl"
    source.write_text(json.dumps(row) + "\n")
    assert len(current.load_current(source)) == 1
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="duplicate task"):
        current.load_current(source)
    with pytest.raises(ValueError, match="missing"):
        current.load_current(tmp_path, ["not-present"])


def test_current_public_evaluate_is_only_scoring_path(row, tmp_path, monkeypatch):
    item = task(row)
    calls = []

    def evaluate(effective, **kwargs):
        calls.append(effective)
        assert effective["image"] == "docker.io/owner/example@sha256:" + "f" * 64
        assert effective["workdir"] == row["workdir"]
        assert effective["test_patch"] == "PRIVATE\r\n"
        directory = Path(kwargs["submission_dir"])
        assert (directory / "impl.py").read_bytes() == b"answer=42\r\n"
        assert (directory / "data.bin").read_bytes() == b"\x00\xff"
        assert not (directory / ".git").exists()
        assert not (directory / ".venv").exists()
        assert kwargs["pull"] is False
        assert kwargs["timeout_seconds"] == 600
        assert set(kwargs) == {"submission_dir", "pull", "timeout_seconds", "docker"}
        assert os.environ["REPOGEN_SCORE_MEMORY"] == "12g"
        return result(item)

    monkeypatch.setattr(current.evaluator, "evaluate", evaluate)
    grade = current.grade_current(
        item,
        bundle(),
        timeout=600,
        artifact_dir=tmp_path,
        image_identity={
            "immutable_image": "docker.io/owner/example@sha256:" + "f" * 64,
            "image_digest": "sha256:" + "f" * 64,
        },
        memory="12g",
    )
    assert grade.reward == 1 and grade.all_tests_passed
    assert len(calls) == 1 and row["image"].endswith(":task")
    assert "REPOGEN_SCORE_MEMORY" not in os.environ
    assert json.loads((tmp_path / "canonical_result.json").read_text())["scoring_valid"]


@pytest.mark.parametrize("passed", [0, 1, 2])
def test_zero_score_is_not_harness_failure(row, passed):
    item = task(row)
    raw = result(item, passed)
    grade = current.current_grade(
        raw, item, raw["image_digest"], item.private.payload["evaluator_identity"]
    )
    assert grade.status == "ok" and grade.reward == passed / 2
    raw.update(status="evaluator_failed", scoring_valid=False, score=None)
    assert (
        current.current_grade(
            raw, item, raw["image_digest"], item.private.payload["evaluator_identity"]
        ).reward
        is None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_digest", "wrong"),
        ("expected", 1),
        ("instance_id", "another"),
        ("score", 0.5),
        ("canonical_runner_sha256", "wrong"),
        ("evaluator_version", "0.3.1"),
    ],
)
def test_result_identity_fail_closed(row, field, value):
    item = task(row)
    raw = result(item)
    raw[field] = value
    with pytest.raises(ValueError):
        current.current_grade(
            raw, item, "sha256:" + "f" * 64, item.private.payload["evaluator_identity"]
        )

