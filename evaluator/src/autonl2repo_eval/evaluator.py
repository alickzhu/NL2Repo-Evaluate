"""Run the canonical AutoNL2Repo scorer for one OpenHands submission."""

from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

from ._submission import materialize_implementation
from ._test_support import prepare_test_support
from ._version import __version__


class EvaluationError(RuntimeError):
    """The evaluation infrastructure or task contract is invalid."""


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path == PurePosixPath(".") or path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise EvaluationError(f"unsafe task path: {value!r}")
    return path.as_posix()


def _install_targets(task: Mapping[str, Any]) -> list[str]:
    targets: list[str] = []
    for raw in task.get("package_setup_files") or []:
        path = PurePosixPath(_safe_relative(str(raw)))
        if path.name not in {"pyproject.toml", "setup.py", "setup.cfg"}:
            continue
        parent = path.parent.as_posix()
        value = "." if parent in {"", "."} else parent
        if value not in targets:
            targets.append(value)
    return targets


def _validate_task(task: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    # Repository-wide withdrawal: upstream E2E tests contact flagged external sites.
    identity = str(task.get("instance_id", "")).lower()
    repository = str(task.get("github_url", "")).lower().rstrip("/").removesuffix(".git")
    owner_repo = f"{task.get('user', '')}/{task.get('repo', '')}".lower()
    if (identity.startswith("soxoj__socid-extractor--")
            or repository.endswith("/soxoj/socid-extractor")
            or owner_repo == "soxoj/socid-extractor"):
        raise EvaluationError("task disabled: soxoj/socid-extractor (external-network E2E security withdrawal)")
    required = ("instance_id", "image", "workdir", "parent_commit", "passed_ptp")
    missing = [key for key in required if not task.get(key)]
    if missing:
        raise EvaluationError("missing task fields: " + ", ".join(missing))
    commit = str(task["parent_commit"])
    if len(commit) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in commit):
        raise EvaluationError("parent_commit must be a 40-character hexadecimal commit")
    workdir = str(task["workdir"])
    if not workdir.startswith("/") or "\x00" in workdir:
        raise EvaluationError("workdir must be an absolute container path")
    nodes = [str(node) for node in task["passed_ptp"]]
    if not nodes or len(nodes) != len(set(nodes)):
        raise EvaluationError("passed_ptp must be a non-empty unique node list")
    declared = int(task["test_case_count"]) if task.get("test_case_count") is not None else len(nodes)
    if declared != len(nodes):
        raise EvaluationError("test_case_count differs from the frozen denominator")
    test_files = [_safe_relative(str(value)) for value in task.get("test_files") or []]
    if len(test_files) != len(set(test_files)):
        raise EvaluationError("test_files contains duplicate paths")
    return workdir, commit, nodes


def _run(
    argv: Sequence[str], *, timeout: float, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(
        list(argv), text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=max(1.0, timeout), check=False,
    )
    if check and process.returncode:
        tail = (process.stdout + "\n" + process.stderr)[-8000:]
        raise EvaluationError(
            f"command failed ({process.returncode}): {' '.join(argv[:3])}\n{tail}"
        )
    return process


def _remaining(deadline: float, cap: float | None = None) -> float:
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("evaluation timeout")
    return min(seconds, cap) if cap is not None else seconds


def _docker(
    executable: str, arguments: Sequence[str], *, deadline: float,
    cap: float | None = None, check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [executable, *arguments], timeout=_remaining(deadline, cap), check=check,
    )


def _resolved_image(
    executable: str, image: str, *, deadline: float,
) -> tuple[str, str]:
    inspected = _docker(
        executable,
        ["image", "inspect", image, "--format", "{{json .RepoDigests}}"],
        deadline=deadline, cap=60,
    )
    try:
        repo_digests = json.loads(inspected.stdout.strip())
    except json.JSONDecodeError as exc:
        raise EvaluationError("container runtime returned invalid RepoDigests") from exc
    # One local image can carry several registry manifests (e.g. an OCI index
    # and its platform manifest). List order is not the requested identity.
    if "@sha256:" in image:
        expected = image.rsplit("@", 1)[1]
        if not any(str(value).endswith("@" + expected) for value in repo_digests):
            raise EvaluationError("container runtime is missing the requested immutable image digest")
        return image, expected
    immutable = next(
        (str(value) for value in repo_digests if "@sha256:" in str(value)), "",
    )
    if not immutable:
        raise EvaluationError("task image has no immutable registry digest")
    return immutable, immutable.split("@", 1)[1]


def _materialize_task(
    task: Mapping[str, Any], root: Path, agent_patch: bytes | None,
) -> tuple[Path, str]:
    workdir, commit, nodes = _validate_task(task)
    if agent_patch is not None and not agent_patch.strip():
        raise EvaluationError("agent_patch is empty")
    test_patch = str(task.get("test_patch") or "").encode("utf-8")
    binary_text = str(task.get("test_binary_archive_b64") or "")
    try:
        binary = base64.b64decode(binary_text, validate=True) if binary_text else b""
    except ValueError as exc:
        raise EvaluationError("test_binary_archive_b64 is invalid") from exc
    if not test_patch and not binary:
        raise EvaluationError("task has no frozen test payload")

    instance_id = str(task["instance_id"])
    snapshot_id = str(task.get("snapshot_id") or hashlib.sha256(
        instance_id.encode("utf-8"),
    ).hexdigest()[:32])
    targets = _install_targets(task)
    task_copy = dict(task)
    task_copy["snapshot_id"] = snapshot_id
    specification = {
        "schema_version": 1,
        "instance_id": instance_id,
        "snapshot_id": snapshot_id,
        "workdir": workdir,
        "parent_commit": commit,
        "passed_ptp": nodes,
        "environment": {},
        "pytest_addopts": [],
        "pytest_command": [],
        "pytest_execution_contract": "published_test_commands_v1",
        "gold_injection_policy": "implementation_only_v1",
        "install_targets": targets,
        "require_editable_install": bool(targets),
        "test_patch_sha256": hashlib.sha256(test_patch).hexdigest(),
        "binary_archive_sha256": hashlib.sha256(binary).hexdigest() if binary else "",
    }
    prepared = root / "prepared"
    prepared.mkdir()
    (prepared / "task.json").write_text(
        json.dumps(task_copy, sort_keys=True), encoding="utf-8",
    )
    (prepared / "validation_spec.json").write_text(
        json.dumps(specification, sort_keys=True), encoding="utf-8",
    )
    (prepared / "test_patch.diff").write_bytes(test_patch)
    if binary:
        (prepared / "binfix.tar.gz").write_bytes(binary)
    if agent_patch is not None:
        (root / "submission.patch").write_bytes(agent_patch)
    return prepared, snapshot_id


def evaluate(
    task: Mapping[str, Any],
    agent_patch: str | bytes | None = None,
    *,
    submission_dir: str | Path | None = None,
    docker: str = "docker",
    timeout_seconds: int = 3600,
    pull: bool = True,
) -> dict[str, Any]:
    """Score an implementation directory (recommended) or a legacy Git patch.

    Directory submissions do not assume the development and scorer Git
    baselines match. Legacy patches must match the actual image clean baseline.
    Missing submissions never select reference Gold implicitly.
    """
    if (agent_patch is None) == (submission_dir is None):
        raise EvaluationError("provide exactly one of agent_patch or submission_dir")
    return _evaluate(task, agent_patch, submission_dir=submission_dir,
                     docker=docker, timeout_seconds=timeout_seconds, pull=pull)


def audit_reference(task: Mapping[str, Any], *, docker: str = "docker",
                    timeout_seconds: int = 3600, pull: bool = True) -> dict[str, Any]:
    """Audit the reference through the same directory delivery path as agents.

    This is a task-health check, never a model score. The result explicitly
    records reference origin; evaluate() cannot fall back to this operation.
    """
    return _evaluate(task, None, submission_dir=None, reference=True,
                     docker=docker, timeout_seconds=timeout_seconds, pull=pull)


def _score_result(result: dict[str, Any], task: Mapping[str, Any], returncode: int) -> None:
    """A known denominator or successful shell exit is not a valid score."""
    nodes = set(task["passed_ptp"])
    timing = result.get("timing") or {}
    commands = timing.get("pytest_commands") or []
    accepted = result.get("accepted_passed_ptp") or []
    non_passed = result.get("non_passed") or []
    ledger = [*accepted, *(row.get("nodeid") for row in non_passed)]
    observed_count = sum(int(value) for key, value in (result.get("outcome_counts") or {}).items()
                         if key != "missing")
    ledger_counts = Counter({"passed": len(accepted)})
    for row in non_passed:
        ledger_counts[str(row.get("outcome") or "missing")] += 1
    # Matching node IDs alone is insufficient if the outcome counters disagree
    # with the per-node ledger. Fail closed on corrupted/misassembled reports.
    counts_match = (
        {key: value for key, value in ledger_counts.items() if value}
        == {key: int(value) for key, value in (result.get("outcome_counts") or {}).items() if value}
        and all(row.get("outcome") != "passed" for row in non_passed)
    )
    valid = bool(
        returncode == 0 and result.get("pipeline_stage") == "pytest_finished"
        and result.get("expected") == len(nodes)
        and len(ledger) == len(nodes) and set(ledger) == nodes
        and int(result.get("passed") or 0) == len(accepted)
        and counts_match and observed_count > 0 and commands
        and timing.get("pytest_planned_command_count") == len(commands)
        and timing.get("pytest_completed_command_count") == len(commands)
        and all(row.get("status") == "completed" and row.get("structured_report_available")
                and not row.get("timed_out") for row in commands)
    )
    result["canonical_status"] = result.get("status")
    result["scoring_valid"] = valid
    result["expected"] = len(nodes)
    result["score"] = int(result.get("passed") or 0) / len(nodes) if valid else None
    result["accepted"] = valid and len(accepted) == len(nodes)
    result["status"] = "scored" if valid else "evaluator_failed"
    if not valid:
        result["reason"] = result.get("reason") or "incomplete_or_invalid_pytest_report"


def _evaluate(task, agent_patch, *, submission_dir, docker, timeout_seconds, pull,
              reference=False):
    started = time.monotonic()
    deadline = started + max(1, int(timeout_seconds))
    patch = (agent_patch.encode("utf-8") if isinstance(agent_patch, str)
             else bytes(agent_patch) if agent_patch is not None else None)
    container = "autonl2repo-eval-" + uuid.uuid4().hex[:16]
    created = False
    result: dict[str, Any] = {
        "schema_version": 1,
        "instance_id": str(task.get("instance_id") or ""),
        "status": "evaluator_failed",
        "accepted": False,
        "score": None,
        "scoring_valid": False,
        "evaluation_kind": "reference_audit" if reference else "agent_submission",
        "evaluator_version": __version__,
        "reason": "evaluation_not_started",
    }
    try:
        task, support_repair = prepare_test_support(task)
        if support_repair is not None:
            result["test_support_repair"] = support_repair
        with tempfile.TemporaryDirectory(prefix="autonl2repo-eval-") as temporary:
            root = Path(temporary)
            prepared, snapshot_id = _materialize_task(task, root, patch)
            staged = root / "implementation"
            if submission_dir is not None:
                materialize_implementation(Path(submission_dir), staged,
                                           protected_paths=task.get("test_files") or [])
            image = str(task["image"])
            if pull:
                _docker(docker, ["pull", image], deadline=deadline, cap=900)
            immutable_image, image_digest = _resolved_image(
                docker, image, deadline=deadline,
            )
            declared_digest = str(task.get("image_digest") or "")
            if declared_digest and declared_digest != image_digest:
                raise EvaluationError(
                    f"image digest mismatch: expected {declared_digest}, got {image_digest}"
                )
            if support_repair is not None and support_repair["required_image_digest"] != image_digest:
                raise EvaluationError("frozen test support repair image digest mismatch")

            _docker(
                docker,
                [
                    "create", "--name", container, "--network", "none",
                    "--env", f"TASKFORGE_IMAGE_DIGEST={image_digest}",
                    "--entrypoint", "tail", immutable_image, "-f", "/dev/null",
                ],
                deadline=deadline, cap=120,
            )
            created = True
            _docker(docker, ["start", container], deadline=deadline, cap=60)
            _docker(
                docker, ["exec", container, "mkdir", "-p", "/tmp/taskforge-eval"],
                deadline=deadline, cap=30,
            )
            runner = Path(__file__).with_name("_canonical_runner.py")
            runner_digest = hashlib.sha256(runner.read_bytes()).hexdigest()
            transfers = [
                (prepared, "/tmp/taskforge-eval/prepared"),
                (runner, "/tmp/taskforge-eval/runner.py"),
                (Path(__file__).with_name("_submission.py"), "/tmp/taskforge-eval/_submission.py"),
            ]
            if submission_dir is not None:
                transfers.append((staged, "/tmp/taskforge-eval/implementation"))
            elif patch is not None:
                transfers.append((root / "submission.patch", "/tmp/taskforge-eval/submission.patch"))
            for source, destination in transfers:
                _docker(
                    docker, ["cp", str(source), f"{container}:{destination}"],
                    deadline=deadline, cap=180,
                )
            command = [
                "exec", container, "python3", "/tmp/taskforge-eval/runner.py",
                "--prepared-dir", "/tmp/taskforge-eval/prepared",
                "--output", "/tmp/taskforge-eval/result.json",
                "--suite-timeout", str(max(1, int(timeout_seconds))),
                "--command-timeout", str(min(max(1, int(timeout_seconds)), 1800)),
                "--min-pass-rate", "1.0", "--min-passed-tests", "1",
            ]
            if submission_dir is not None:
                command.extend(["--submission-dir", "/tmp/taskforge-eval/implementation"])
            elif patch is not None:
                command.extend(["--submission-patch", "/tmp/taskforge-eval/submission.patch"])
            process = _docker(
                docker, command, deadline=deadline, check=False,
            )
            local_result = root / "result.json"
            copied = _docker(
                docker,
                ["cp", f"{container}:/tmp/taskforge-eval/result.json", str(local_result)],
                deadline=deadline, cap=60, check=False,
            )
            if copied.returncode or not local_result.is_file():
                tail = (process.stdout + "\n" + process.stderr)[-8000:]
                raise EvaluationError(f"canonical scorer produced no result.json\n{tail}")
            result = json.loads(local_result.read_text(encoding="utf-8"))
            if support_repair is not None:
                result["test_support_repair"] = support_repair
            _score_result(result, task, process.returncode)
            timing = dict(result.get("timing") or {})
            result.update({
                "instance_id": str(task["instance_id"]),
                "evaluator_version": __version__,
                "snapshot_id": snapshot_id,
                "image": image,
                "immutable_image": immutable_image,
                "image_digest": image_digest,
                "evaluation_kind": "reference_audit" if reference else "agent_submission",
                "pytest_seconds": timing.get("pytest_wall_seconds"),
                "install_seconds": timing.get("editable_install_wall_seconds"),
                "canonical_runner_sha256": runner_digest,
                "submission_materializer_sha256": hashlib.sha256(
                    Path(__file__).with_name("_submission.py").read_bytes()).hexdigest(),
                "test_support_loader_sha256": hashlib.sha256(
                    Path(__file__).with_name("_test_support.py").read_bytes()).hexdigest(),
                "scorer_returncode": process.returncode,
            })
            return result
    except (subprocess.TimeoutExpired, TimeoutError):
        result.update(status="evaluator_failed", scoring_valid=False, score=None, accepted=False)
        result["reason"] = "evaluation_timeout"
        return result
    except (EvaluationError, OSError, ValueError, json.JSONDecodeError) as exc:
        result.update(status="evaluator_failed", scoring_valid=False, score=None, accepted=False)
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    finally:
        if created:
            try:
                _run([docker, "rm", "-f", container], timeout=60, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
        result["total_wall_seconds"] = round(time.monotonic() - started, 3)
