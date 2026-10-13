from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from autonl2repo_eval import (
    CANONICAL_RUNNER_SHA256,
    TASKFORGE_SOURCE_COMMIT,
    __version__,
)
import autonl2repo_eval
from autonl2repo_eval._version import SUBMISSION_MATERIALIZER_SHA256
from autonl2repo_eval.evaluator import (
    EvaluationError,
    _install_targets,
    _materialize_task,
    _validate_task,
)


def test_published_runner_is_content_bound():
    runner = Path(autonl2repo_eval.__file__).with_name("_canonical_runner.py")
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == CANONICAL_RUNNER_SHA256
    materializer = Path(autonl2repo_eval.__file__).with_name("_submission.py")
    assert hashlib.sha256(materializer.read_bytes()).hexdigest() == SUBMISSION_MATERIALIZER_SHA256
    assert __version__ == "0.3.2"
    assert TASKFORGE_SOURCE_COMMIT == "fc10f55567f86c6d5ad549e5815fa0e1dc6c936f"


def task(**updates):
    value = {
        "instance_id": "owner__repo--t2--l10--01234567",
        "image": "docker.io/example/tasks:one",
        "workdir": "/workspace",
        "parent_commit": "0" * 40,
        "passed_ptp": ["tests/test_x.py::test_a", "tests/test_x.py::test_b"],
        "test_case_count": 2,
        "test_files": ["tests/test_x.py", "tests/data.bin"],
        "test_patch": (
            "diff --git a/tests/test_x.py b/tests/test_x.py\n"
            "new file mode 100644\n--- /dev/null\n+++ b/tests/test_x.py\n"
            "@@ -0,0 +1 @@\n+def test_a(): pass\n"
        ),
        "test_binary_archive_b64": base64.b64encode(b"archive").decode(),
        "test_binary_files": ["tests/data.bin"],
        "package_setup_files": ["pyproject.toml", "packages/a/setup.py"],
    }
    value.update(updates)
    return value


def test_task_denominator_is_frozen():
    with pytest.raises(EvaluationError, match="denominator"):
        _validate_task(task(test_case_count=1))


def test_task_paths_cannot_escape_workspace():
    with pytest.raises(EvaluationError, match="unsafe"):
        _validate_task(task(test_files=["../secret"]))


def test_install_targets_follow_declared_package_files():
    assert _install_targets(task()) == [".", "packages/a"]


def test_materialization_preserves_patch_bytes_and_freezes_contract(tmp_path: Path):
    value = task()
    prepared, snapshot_id = _materialize_task(value, tmp_path, b"binary patch\n")
    assert snapshot_id == hashlib.sha256(value["instance_id"].encode()).hexdigest()[:32]
    assert (prepared / "test_patch.diff").read_bytes() == value["test_patch"].encode()
    assert (prepared / "binfix.tar.gz").read_bytes() == b"archive"
    assert (tmp_path / "submission.patch").read_bytes() == b"binary patch\n"
    spec = json.loads((prepared / "validation_spec.json").read_text())
    assert spec["passed_ptp"] == value["passed_ptp"]
    assert spec["pytest_execution_contract"] == "published_test_commands_v1"
    assert spec["install_targets"] == [".", "packages/a"]


def test_empty_submission_is_rejected(tmp_path: Path):
    with pytest.raises(EvaluationError, match="empty"):
        _materialize_task(task(), tmp_path, b"\n")
