"""Trusted preparation, bounded submission transport, and private grading.

No untrusted archive is ever extracted on the controller. ``validate_submission``
accepts Docker's uncompressed ``get_archive('/workspace')`` format and returns a
root-STRIPPED tar for ``target.restore(tar, path='/workspace')``. Only ordinary
files/directories are supported; all symlinks, hardlinks and sparse files are
rejected. This deliberately conservative first protocol does not encode deletes.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import math
import secrets
import tarfile
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol

from .contracts import SCHEMA_VERSION, DatasetTask, GradeResult, relative_path


MAX_SUBMISSION_FILES = 50_000
MAX_SUBMISSION_BYTES = 128 * 1024 * 1024
MAX_REPORT_BYTES = 64 * 1024
_CONTROL_DIR = "/tmp/openhands-research-grading"
_REPORTER = "openhands-research-grade-v1"


class TaskTarget(Protocol):
    """Structural view of the owned runtime; no Docker/SDK runtime import needed.

    The current runtime records role/task in immutable ownership labels, checked
    against Docker by every transport operation. The caller must allocate a NEW
    role='grading' container from the task image, never retarget a work container.
    """

    _image: str

    @property
    def _labels(self) -> Mapping[str, str]: ...

    @property
    def workdir(self) -> str: ...

    def exec(
        self,
        command: str | list[str],
        timeout: float = 300,
        workdir: str | None = None,
        user: str | None = None,
        privileged: bool = False,
    ) -> tuple[int, str, str]: ...

    def write_file(self, path: str, content: str | bytes) -> None: ...

    def read_file(self, path: str) -> bytes: ...

    def restore(self, archive: bytes, path: str = "/") -> None: ...


def validate_submission(archive: bytes, *, root_name: str = "workspace") -> bytes:
    """Validate and normalize a Docker workspace tar without host extraction.

    Enforce 50,000 members and 128 MiB of declared regular-file content, including
    dropped Git content. Reject ambiguous path/type collisions before filtering.
    Ownership/timestamps/permissions are normalized, with executable bits retained.
    """
    if not isinstance(archive, bytes):
        raise ValueError("submission must be tar bytes")
    if (
        relative_path(root_name) != root_name
        or len(PurePosixPath(root_name).parts) != 1
    ):
        raise ValueError("submission root must be one trusted directory name")
    wire_limit = MAX_SUBMISSION_BYTES + MAX_SUBMISSION_FILES * 1536 + 10240
    if len(archive) > wire_limit:
        raise ValueError("submission tar exceeds its wire-size limit")
    output = io.BytesIO()
    seen: dict[str, str] = {}
    parents: set[str] = set()
    size = 0
    try:
        with (
            tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source,
            tarfile.open(
                fileobj=output,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as destination,
        ):
            for index, member in enumerate(source, 1):
                if index > MAX_SUBMISSION_FILES:
                    raise ValueError("submission exceeds the member-count limit")
                name = relative_path(member.name)
                parts = PurePosixPath(name).parts
                if parts[0] != root_name:
                    raise ValueError("submission must have exactly the workspace root")
                if name in seen:
                    raise ValueError(f"duplicate normalized archive path: {name!r}")
                if member.type not in (
                    tarfile.REGTYPE,
                    tarfile.AREGTYPE,
                    tarfile.DIRTYPE,
                ):
                    raise ValueError(
                        "submission links and special file types are not supported"
                    )
                if member.sparse is not None or any(
                    key.startswith("GNU.sparse") for key in member.pax_headers
                ):
                    raise ValueError("sparse submission files are not supported")
                if member.size < 0 or (member.isdir() and member.size != 0):
                    raise ValueError("invalid tar member size")
                if member.isfile() and name in parents:
                    raise ValueError(
                        "archive file collides with an existing parent directory"
                    )
                for depth in range(1, len(parts)):
                    parent = "/".join(parts[:depth])
                    if seen.get(parent) == "file":
                        raise ValueError("archive path descends through a regular file")
                    parents.add(parent)
                seen[name] = "directory" if member.isdir() else "file"
                if len(parts) == 1:
                    if not member.isdir():
                        raise ValueError("workspace root must be a directory")
                    continue
                if member.isfile():
                    size += member.size
                    if size > MAX_SUBMISSION_BYTES:
                        raise ValueError("submission exceeds 128 MiB of file content")
                if ".git" in parts[1:]:
                    continue
                normalized = tarfile.TarInfo("/".join(parts[1:]))
                normalized.uid = normalized.gid = 0
                normalized.mtime = 0
                executable = member.isdir() or member.mode & 0o111
                normalized.mode = 0o755 if executable else 0o644
                normalized.type = tarfile.DIRTYPE if member.isdir() else tarfile.REGTYPE
                if member.isdir():
                    destination.addfile(normalized)
                else:
                    stream = source.extractfile(member)
                    if stream is None:
                        raise ValueError("submission file contents are missing")
                    with stream:
                        content = stream.read(member.size + 1)
                    if len(content) != member.size:
                        raise ValueError("submission file contents are truncated")
                    normalized.size = len(content)
                    destination.addfile(normalized, io.BytesIO(content))
    except (tarfile.TarError, OSError, EOFError, UnicodeError) as exc:
        raise ValueError("invalid uncompressed workspace tar") from exc
    if not seen:
        raise ValueError("submission is missing the workspace root")
    return output.getvalue()


def make_submission(
    files: dict[str, bytes | str], *, root_name: str = "workspace"
) -> bytes:
    """Create a Docker-rooted tar for CPU fixtures and small explicit submissions."""
    if not isinstance(files, dict) or len(files) + 1 > MAX_SUBMISSION_FILES:
        raise ValueError("files must be a bounded path-to-content dictionary")
    buffer = io.BytesIO()
    seen: set[str] = set()
    total = 0
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        root = tarfile.TarInfo(root_name)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)
        for path, content in files.items():
            name = relative_path(path)
            if name in seen:
                raise ValueError("duplicate normalized fixture path")
            seen.add(name)
            if isinstance(content, str):
                content = content.encode("utf-8")
            if not isinstance(content, bytes):
                raise ValueError("submission file contents must be bytes or text")
            total += len(content)
            if total > MAX_SUBMISSION_BYTES:
                raise ValueError("submission exceeds 128 MiB")
            member = tarfile.TarInfo(root_name + "/" + name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))
    raw = buffer.getvalue()
    validate_submission(raw, root_name=root_name)
    return raw


def _bootstrap(spec, root="/", site_paths=None):
    """Executed as trusted task root before tools; also runnable on CPU fixtures."""
    import ast
    import configparser
    import importlib.metadata
    import json
    import os
    import re
    import shutil
    import sysconfig
    from pathlib import Path
    from urllib.parse import unquote, urlparse

    base = Path(root)
    workspace = base / "workspace"
    original = base / "workspace_orig"
    evaluation = base / "workspace_eval"
    agent_copy = base / "agent_copy"
    control = base / "tmp/openhands-research-grading"

    def remove(path):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    def inside(path, parent):
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def canonical_name(value):
        return re.sub(r"[-_.]+", "-", value).lower()

    def remove_installed_target():
        names = {canonical_name(spec["task_id"])}
        names.update(canonical_name(name) for name in spec.get("blocked_packages", []))
        setup = workspace / "setup.py"
        if setup.is_file() and not setup.is_symlink():
            try:
                tree = ast.parse(setup.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and (
                        isinstance(node.func, ast.Name)
                        and node.func.id == "setup"
                        or (
                            isinstance(node.func, ast.Attribute)
                            and node.func.attr == "setup"
                        )
                    ):
                        for keyword in node.keywords:
                            if keyword.arg == "name" and isinstance(
                                keyword.value, ast.Constant
                            ):
                                value = keyword.value.value
                                if isinstance(value, str):
                                    names.add(canonical_name(value))
            except (SyntaxError, UnicodeError):
                pass
        config = workspace / "setup.cfg"
        if config.is_file() and not config.is_symlink():
            parser = configparser.ConfigParser(interpolation=None)
            parser.read(config, encoding="utf-8")
            if parser.has_option("metadata", "name"):
                names.add(canonical_name(parser.get("metadata", "name")))
        project = workspace / "pyproject.toml"
        if project.is_file() and not project.is_symlink():
            section = ""
            for line in project.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("["):
                    section = line.strip()
                match = re.match(r"\s*name\s*=\s*['\"]([^'\"]+)['\"]", line)
                if match and section in ("[project]", "[tool.poetry]"):
                    names.add(canonical_name(match.group(1)))
        roots = [
            Path(path).resolve()
            for path in (
                site_paths
                if site_paths is not None
                else {
                    sysconfig.get_path("purelib"),
                    sysconfig.get_path("platlib"),
                }
            )
            if path
        ]
        distributions = importlib.metadata.distributions(
            path=[str(path) for path in roots]
        )
        for distribution in distributions:
            name = distribution.metadata.get("Name", "")
            direct = distribution.read_text("direct_url.json")
            from_workspace = False
            if direct:
                url = urlparse(json.loads(direct).get("url", ""))
                if url.scheme == "file":
                    from_workspace = inside(
                        Path(unquote(url.path)).resolve(), workspace.resolve()
                    )
            if canonical_name(name) not in names and not from_workspace:
                continue
            recorded = distribution.files
            if not recorded:
                raise RuntimeError(
                    "installed target lacks a file manifest; "
                    "image-specific cleanup required"
                )
            # Remove only this distribution's recorded files, not whole shared
            # site-packages or namespace directories and not dependency packages.
            for entry in recorded:
                path = Path(str(distribution.locate_file(entry)))
                path = path.parent.resolve() / path.name
                if not any(inside(path, directory) for directory in roots):
                    continue
                if path.is_file() or path.is_symlink():
                    path.unlink()
                if path.suffix == ".py":
                    cache_dir = path.parent / "__pycache__"
                    for cache in cache_dir.glob(path.stem + ".*.pyc"):
                        if cache.is_file() or cache.is_symlink():
                            cache.unlink()
        # Legacy editable links expose the original source through site paths.
        for directory in roots:
            for link in directory.glob("*.egg-link"):
                lines = link.read_text(encoding="utf-8").splitlines()
                if lines and inside(Path(lines[0]).resolve(), workspace.resolve()):
                    link.unlink()

    if os.geteuid() != 0:
        raise PermissionError(
            "trusted workspace bootstrap requires task-container root"
        )
    if workspace.is_symlink():
        raise ValueError("workspace root must not be a symlink")
    workspace.mkdir(exist_ok=True)
    if spec["phase"] == "grading":
        control.parent.mkdir(parents=True, exist_ok=True)
        control.mkdir(mode=0o700)
        for path in (original, evaluation, agent_copy):
            remove(path)
        shutil.copytree(workspace, original, symlinks=True)
        for path in workspace.iterdir():
            remove(path)
    elif spec["phase"] == "work":
        for path in (original, evaluation, agent_copy, control):
            remove(path)
        if spec["kind"] == "nl2repo" or spec.get("empty_workspace", False):
            remove_installed_target()
            for path in workspace.iterdir():
                remove(path)
        else:
            # Custom images must contain public starter code, not hidden answers.
            # Only explicitly declared private tests and Git history are removed.
            for relative in spec["test_files"]:
                path = workspace / relative
                if not inside(path.parent.resolve(), workspace.resolve()):
                    raise ValueError(
                        "custom protected path escapes through an image symlink"
                    )
                remove(path)
            for directory, subdirs, files in os.walk(workspace, followlinks=False):
                if ".git" in subdirs or ".git" in files:
                    remove(Path(directory) / ".git")
                    if ".git" in subdirs:
                        subdirs.remove(".git")
    else:
        raise ValueError("unsupported preparation phase")
    os.chown(workspace, 0, 0)
    workspace.chmod(0o755)


def _require_target(
    target: TaskTarget,
    task: DatasetTask,
    phase: str,
    work_image: str | None = None,
    *,
    allow_nested: bool = False,
) -> None:
    if phase not in {"work", "grading"}:
        raise ValueError("phase must be 'work' or 'grading'")
    if target.workdir != task.public.workdir or (
        not allow_nested and task.public.workdir != "/workspace"
    ):
        raise ValueError("the initial submission protocol requires workdir=/workspace")
    if target._labels.get("org.repogeneration.role") != phase:
        raise ValueError(f"preparation requires a new role={phase!r} target")
    if target._labels.get("org.repogeneration.task") not in {
        task.public.task_id,
        task.stable_key,
    }:
        raise ValueError("target ownership belongs to a different task")
    expected_image = work_image if phase == "work" and work_image else task.public.image
    if target._image != expected_image:
        raise ValueError("target image does not match the public task image")


def prepare_task(
    target: TaskTarget,
    task: DatasetTask,
    phase: Literal["work", "grading"] = "work",
    *,
    blocked_packages: tuple[str, ...] = (),
    work_image: str | None = None,
    empty_workspace: bool = False,
) -> None:
    """Bootstrap one owned target as task root, before any solver/tool discovery.

    NL2Repo work is empty (no README/Git/reference backup). Custom work retains
    public starter code from its trusted image, minus declared private test paths.
    Grading preserves the fresh canonical image tree before clearing /workspace;
    an exclusive grading directory makes accidental container reuse fail closed.
    """
    _require_target(target, task, phase, work_image)
    spec = {
        "phase": phase,
        "kind": task.private.kind,
        "task_id": task.public.task_id,
        "blocked_packages": list(blocked_packages),
        "empty_workspace": empty_workspace,
        "test_files": (
            list(task.private.test_files) if task.private.kind == "command" else []
        ),
    }
    source = (
        "from __future__ import annotations\n"
        + inspect.getsource(_bootstrap)
        + "\nimport json, sys\n_bootstrap(json.loads(sys.argv[1]))\n"
    )
    code, _, stderr = target.exec(
        ["python3", "-I", "-S", "-c", source, json.dumps(spec)],
        timeout=300,
        workdir="/",
        user="root",
        privileged=True,
    )
    if code != 0:
        raise RuntimeError(f"trusted {phase} bootstrap failed: {stderr[-1500:]}")


def _error(
    task: DatasetTask,
    reason: str,
    *,
    status: Literal["error", "timeout"] = "error",
    **details: Any,
) -> GradeResult:
    return GradeResult(
        status=status,
        total=task.private.test_case_count,
        details={"reason": reason, **details},
    )


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate result field")
        result[key] = value
    return result


def _read_result(raw: bytes | str, task: DatasetTask, token: str) -> GradeResult:
    if isinstance(raw, str):
        raw = raw.encode("utf-8")
    if not isinstance(raw, bytes) or len(raw) > MAX_REPORT_BYTES:
        raise ValueError("private report exceeds its size limit")
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_object)
    labels = {"reporter", "schema_version", "task_key", "report_token"}
    grade_fields = {item.name for item in fields(GradeResult)}
    if not isinstance(payload, dict) or set(payload) != labels | grade_fields:
        raise ValueError("private report has an invalid schema")
    if (
        payload["reporter"] != _REPORTER
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != SCHEMA_VERSION
        or payload["task_key"] != task.stable_key
        or payload["report_token"] != token
    ):
        raise ValueError("private report identity does not match the trusted evaluator")
    result = GradeResult(**{name: payload[name] for name in grade_fields})
    if result.total != task.private.test_case_count:
        raise ValueError("private report changed the declared denominator")
    if result.status == "ok":
        expected = min(result.passed / result.total, 1.0)
        if result.reward != expected or result.resolved != (expected >= 1.0):
            raise ValueError("private report violates the declared scoring rule")
        if task.private.kind == "command" and (
            result.passed + result.failed != result.total
            or result.errors
            or result.skipped
        ):
            raise ValueError("private report violates one-count-per-command scoring")
        if task.private.kind == "denovoswe" and (
            result.passed + result.failed + result.errors + result.skipped
            != result.total
        ):
            raise ValueError("selected pytest report must account for every scored ID")
    return result


def grade_submission(
    target: TaskTarget,
    task: DatasetTask,
    archive: bytes,
    timeout: float = 1200,
) -> GradeResult:
    """Prepare a fresh private target, restore safely, run and read a trusted report.

    The caller owns container allocation and teardown. A transport deadline can
    leave code running: this function never retries it or reuses the target.
    Result tokens prevent stale/misrouted reports and stdout spoofing, not attacks
    by code that has root access inside its own private grading container.
    """
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("grading timeout must be positive and finite")
    stage = "target_validation"
    try:
        _require_target(target, task, "grading")
        stage = "submission_validation"
        normalized = validate_submission(archive)
        stage = "preparation"
        prepare_task(target, task, phase="grading")
        stage = "submission_restore"
        target.restore(normalized, path="/workspace")
        token = secrets.token_hex(24)
        script = Path(__file__).with_name("grade_task.py").read_bytes()
        spec = {
            **asdict(task.private),
            "schema_version": SCHEMA_VERSION,
            "task_key": task.stable_key,
            "report_token": token,
            "timeout": timeout,
            "log_path": _CONTROL_DIR + "/commands.log",
        }
        stage = "evaluator_bootstrap"
        target.write_file(
            _CONTROL_DIR + "/spec.json", json.dumps(spec, allow_nan=False)
        )
        target.write_file(_CONTROL_DIR + "/grade_task.py", script)
        stage = "evaluator_execution"
        code, stdout, stderr = target.exec(
            [
                "python3",
                "-I",
                "-S",
                _CONTROL_DIR + "/grade_task.py",
                "--spec",
                _CONTROL_DIR + "/spec.json",
                "--result",
                _CONTROL_DIR + "/result.json",
            ],
            timeout=timeout + 10,
            workdir="/",
            user="root",
        )
        if code != 0:
            return _error(
                task,
                "evaluator_exit",
                status="timeout" if code == 124 else "error",
                evaluator_exit_code=code,
                output_tail=(stdout + "\n" + stderr)[-2000:],
            )
        stage = "private_report_read"
        raw = target.read_file(_CONTROL_DIR + "/result.json")
        stage = "private_report_validation"
        result = _read_result(raw, task, token)
        return GradeResult(
            **{
                **asdict(result),
                "details": {
                    **result.details,
                    "evaluator_sha256": hashlib.sha256(script).hexdigest(),
                    "submission_sha256": hashlib.sha256(normalized).hexdigest(),
                },
            }
        )
    except TimeoutError:
        return _error(task, stage + "_timeout", status="timeout")
    except Exception as exc:
        return _error(
            task,
            stage + "_failed",
            exception_type=type(exc).__name__,
            message=str(exc)[-1500:],
        )
