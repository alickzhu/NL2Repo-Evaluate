"""Current task rows and the installed public evaluator, without a second scorer."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from autonl2repo_eval import evaluator
from autonl2repo_eval._version import __version__ as evaluator_version
from packaging.utils import InvalidName, canonicalize_name

from ..contracts import DatasetTask, GradeResult, PrivateEvalSpec, PublicTaskSpec
from ..ledger import atomic_json
from .autonl2repo_directory import validate_directory_submission

CURRENT_CONTRACT = "autonl2repo-public-evaluate-directory-v1"


def evaluator_identity() -> dict[str, Any]:
    root = Path(evaluator.__file__).parent
    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".json"}
        and "__pycache__" not in path.parts
    }
    return {
        "version": evaluator_version,
        "files": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate task JSON key: {key}")
        result[key] = value
    return result


def _blocked_names(row):
    names = [row.get("repo"), row.get("pypi_name"), *(row.get("import_names") or [])]
    names.extend(
        item.get("name") if isinstance(item, dict) else item
        for item in row.get("pypi_name_candidates") or []
    )
    blocked = set()
    for name in names:
        if not name:
            continue
        try:
            blocked.add(str(canonicalize_name(name, validate=True)))
        except InvalidName:
            continue
    if not blocked:
        raise ValueError("task has no usable target-package identity")
    return sorted(blocked)


def task_from_row(row, identity):
    evaluator._validate_task(row)
    if not isinstance(row.get("document"), str) or not row["document"].strip():
        raise ValueError("task document is required")
    binary = row.get("test_binary_archive_b64") or ""
    if binary:
        base64.b64decode(binary, validate=True)
    if not row.get("test_patch") and not binary:
        raise ValueError("task has no frozen test payload")
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return DatasetTask(
        PublicTaskSpec(
            dataset_id="autonl2repo",
            dataset_revision=f"current-{digest}-{identity['sha256'][:16]}",
            task_id=row["instance_id"],
            prompt=row["document"],
            image=row["image"],
            workdir="/workspace",
            metadata={
                "source_sha256": digest,
                "split": "test",
                "blocked_packages": _blocked_names(row),
            },
        ),
        PrivateEvalSpec(
            kind="autonl2repo",
            test_case_count=len(row["passed_ptp"]),
            test_commands=tuple(row["test_commands"]),
            test_files=tuple(row.get("test_files") or []),
            payload={
                "task": row,
                "evaluation_contract": CURRENT_CONTRACT,
                "evaluator_identity": identity,
                "source_sha256": digest,
            },
        ),
    )


def load_current(source: Path, select: list[str] | None = None) -> list[DatasetTask]:
    source = source.resolve(strict=True)
    if select is not None and (not select or len(set(select)) != len(select)):
        raise ValueError("task selection must be nonempty and unique")
    if source.is_dir():
        directory = source / "tasks" if (source / "tasks").is_dir() else source
        files = sorted(directory.glob("*.json"))
        files += sorted(directory.glob("batch_*/*.json"))
        if select is not None:
            files = [path for path in files if path.stem in set(select)]
    else:
        files = [source]
    identity = evaluator_identity()
    tasks = {}
    for path in files:
        if path.is_symlink():
            raise ValueError("task JSON must not be a symlink")
        raw = path.read_text(encoding="utf-8")
        records = raw.splitlines() if path.suffix == ".jsonl" else [raw]
        for record in records:
            if not record.strip():
                continue
            row = json.loads(record, object_pairs_hook=_object)
            if not isinstance(row, dict) or "instance_id" not in row:
                if source.is_dir():
                    continue  # Sampling manifests are not task rows.
                raise ValueError("expected one complete dataset row per record")
            task_id = row["instance_id"]
            if select is not None and task_id not in select:
                continue
            if task_id in tasks:
                raise ValueError(f"duplicate task: {task_id}")
            if path.suffix == ".json" and source.is_dir() and path.stem != task_id:
                raise ValueError("task filename and instance_id disagree")
            tasks[task_id] = task_from_row(row, identity)
    if select is not None and set(select) != set(tasks):
        raise ValueError(f"selected tasks missing: {sorted(set(select) - set(tasks))}")
    if not tasks:
        raise ValueError("no tasks selected")
    return [tasks[key] for key in (select if select is not None else sorted(tasks))]


def current_grade(result, task, digest, identity):
    if result.get("instance_id") != task.public.task_id:
        raise ValueError("public evaluator returned another task's result")
    details = {
        "evaluation_contract": CURRENT_CONTRACT,
        "evaluator_version": identity["version"],
        "evaluator_sha256": identity["sha256"],
        "image_digest": digest,
        "scoring_valid": result.get("scoring_valid") is True,
        "reason": str(result.get("reason") or "")[:2000],
    }
    if result.get("scoring_valid") is not True or result.get("status") != "scored":
        return GradeResult("error", total=task.private.test_case_count, details=details)
    total = task.private.test_case_count
    passed = result.get("passed")
    score = result.get("score")
    if (
        result.get("image_digest") != digest
        or result.get("expected") != total
        or result.get("evaluator_version") != identity["version"]
        or result.get("canonical_runner_sha256")
        != identity["files"]["_canonical_runner.py"]
        or type(passed) is not int
        or not 0 <= passed <= total
        or type(score) not in (int, float)
        or not math.isfinite(score)
        or score != passed / total
    ):
        raise ValueError("public evaluator result identity or score disagrees")
    commands = [
        {
            "phase": "test",
            "exit_code": row.get("returncode"),
            "timed_out": bool(row.get("timed_out")),
        }
        for row in (result.get("timing") or {}).get("pytest_commands") or []
    ]
    details.update(
        commands=commands[:100],
        command_count=len(commands),
        commands_truncated=len(commands) > 100,
    )
    failed = int(result.get("failed") or 0)
    skipped = int(result.get("skipped") or 0)
    return GradeResult(
        "ok",
        reward=score,
        resolved=passed == total,
        total=total,
        passed=passed,
        failed=failed,
        skipped=skipped,
        errors=total - passed - failed - skipped,
        details=details,
    )


def grade_current(
    task,
    archive,
    *,
    timeout,
    artifact_dir,
    image_identity,
    memory="8g",
    nano_cpus=4_000_000_000,
):
    identity = evaluator_identity()
    if identity != task.private.payload["evaluator_identity"]:
        raise ValueError("public evaluator changed since queue preparation")
    original = task.private.payload["task"]
    digest = hashlib.sha256(
        json.dumps(
            original, ensure_ascii=False, sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()
    if (
        digest != task.private.payload["source_sha256"]
        or original["document"] != task.public.prompt
    ):
        raise ValueError("task row changed after preparation")
    row = dict(original)
    row["image"] = image_identity["immutable_image"]
    wrapper = Path(sys.executable).with_name("autonl2repo-docker")
    settings = {
        "REPOGEN_SCORE_MEMORY": memory,
        "REPOGEN_SCORE_CPUS": str(nano_cpus / 1_000_000_000),
    }
    previous = {key: os.environ.get(key) for key in settings}
    try:
        os.environ.update(settings)
        with tempfile.TemporaryDirectory(prefix="repogen-submission-") as directory:
            with tarfile.open(
                fileobj=io.BytesIO(validate_directory_submission(archive))
            ) as bundle:
                bundle.extractall(directory, filter="data")
            result = evaluator.evaluate(
                row,
                submission_dir=directory,
                pull=False,
                timeout_seconds=int(timeout),
                docker=str(wrapper),
            )
        atomic_json(artifact_dir / "canonical_result.json", result)
        atomic_json(artifact_dir / "evaluator_identity.json", identity)
        return current_grade(result, task, image_identity["image_digest"], identity)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
