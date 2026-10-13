"""Small, SDK-independent dataset and private grading contracts.

Only ``DatasetTask.public_dict()`` (or ``asdict(task.public)``) is an agent
input. Serializing the entire DatasetTask would disclose the private evaluator.
Rewards belong to completed evaluations, never to public task specifications.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal


SCHEMA_VERSION = 1
MAX_DETAILS_BYTES = 32 * 1024
PRIVATE_METADATA_KEYS = frozenset(
    {
        "private",
        "private_eval",
        "private_metadata",
        "evaluation",
        "grader",
        "test_case_count",
        "test_commands",
        "test_files",
        "setup_commands",
        "expected_exit_codes",
        "score_rule",
        "reward",
        "resolved",
        "label",
        "solution",
        "reference",
        "reference_code",
        "hidden_tests",
        "nl2repo",
        "denovoswe",
        "autonl2repo",
        "validation",
        "quality",
        "validation_spec",
        "h3_build_spec",
        "payload",
        "test_patch",
        "test_binary_archive_b64",
        "test_binary_files",
        "passed_ptp",
        "failed_ptp",
        "unit_test",
        "parent_commit",
    }
)


def relative_path(value: str) -> str:
    """Validate a portable, nonempty relative path; normalize redundant dots/slashes."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("paths must be nonempty relative POSIX strings")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("control characters are not allowed in paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or re.match(r"^[A-Za-z]:", value)
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path.as_posix()


def _identifier(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or len(value) > 512
        or relative_path(value) != value
    ):
        raise ValueError(f"{name} must be a nonempty, normalized identifier")


def json_value(value: Any, *, public: bool = False) -> Any:
    """Copy JSON values, rejecting non-finite numbers and non-JSON objects.

    Public metadata is recursively allowlisted by exclusion of reserved private
    keys. Dataset authors are still responsible for the contents of public text.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return [json_value(item, public=public) for item in value]
    if isinstance(value, dict):
        copied = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("metadata keys must be strings")
            if public and (
                key.casefold() in PRIVATE_METADATA_KEYS or key.startswith("_")
            ):
                continue
            copied[key] = json_value(item, public=public)
        return copied
    raise ValueError(f"not a JSON value: {type(value).__name__}")


def _commands(value: tuple[str, ...], name: str, *, required: bool) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of shell commands")
    if required and not value:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(command, str) or not command.strip() or "\0" in command
        for command in value
    ):
        raise ValueError(f"{name} contains an invalid shell command")
    return tuple(value)


@dataclass(frozen=True)
class PublicTaskSpec:
    dataset_id: str
    dataset_revision: str
    task_id: str
    prompt: str
    image: str
    workdir: str = "/workspace"
    sample_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("dataset_id", "dataset_revision", "task_id"):
            _identifier(getattr(self, name), name)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a nonempty string")
        if (
            not isinstance(self.image, str)
            or not self.image
            or any(char.isspace() or ord(char) < 32 for char in self.image)
        ):
            raise ValueError("image must be a nonempty container image reference")
        if (
            not isinstance(self.workdir, str)
            or not self.workdir.startswith("/")
            or self.workdir.startswith("//")
            or "/" + relative_path(self.workdir[1:]) != self.workdir
        ):
            raise ValueError("workdir must be a normalized absolute non-root path")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        # Validate even discarded values so NaN cannot silently enter a manifest.
        json_value(metadata)
        object.__setattr__(self, "metadata", json_value(metadata, public=True))

    @property
    def stable_key(self) -> str:
        """Collision-resistant identity; images may be shared by unrelated tasks."""
        identity = [
            self.dataset_id,
            self.dataset_revision,
            self.task_id,
            self.sample_index,
        ]
        encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def key(self) -> str:
        return self.stable_key

    def public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = json_value(self.metadata, public=True)
        return payload


@dataclass(frozen=True)
class PrivateEvalSpec:
    """Trusted, controller-only evaluator configuration.

    ``command_fraction`` awards one test per command whose exit code equals its
    declared expected code. Its denominator MUST equal the command count.
    ``pytest_declared`` retains NL2Repo's declared denominator and count parser.
    Setup commands run only in the private grading environment, not the solver.
    """

    kind: Literal["nl2repo", "command", "denovoswe", "autonl2repo"]
    test_case_count: int
    test_commands: tuple[str, ...]
    test_files: tuple[str, ...]
    setup_commands: tuple[str, ...] = ()
    expected_exit_codes: tuple[int, ...] = ()
    score_rule: (
        Literal["pytest_declared", "command_fraction", "pytest_selected"] | None
    ) = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"nl2repo", "command", "denovoswe", "autonl2repo"}:
            raise ValueError(f"unsupported evaluator kind: {self.kind!r}")
        if type(self.test_case_count) is not int or self.test_case_count <= 0:
            raise ValueError("test_case_count must be a positive integer")
        object.__setattr__(
            self,
            "test_commands",
            _commands(
                self.test_commands, "test_commands", required=self.kind != "denovoswe"
            ),
        )
        object.__setattr__(
            self,
            "setup_commands",
            _commands(self.setup_commands, "setup_commands", required=False),
        )
        if not isinstance(self.test_files, (list, tuple)):
            raise ValueError("test_files must be a sequence of relative paths")
        paths = tuple(relative_path(path) for path in self.test_files)
        if len(set(paths)) != len(paths):
            raise ValueError("test_files contains duplicate normalized paths")
        object.__setattr__(self, "test_files", paths)
        rule = {
            "nl2repo": "pytest_declared",
            "command": "command_fraction",
            "denovoswe": "pytest_selected",
            "autonl2repo": "pytest_selected",
        }[self.kind]
        if self.score_rule not in {None, rule}:
            raise ValueError(f"{self.kind} requires score_rule={rule!r}")
        object.__setattr__(self, "score_rule", rule)
        if not isinstance(self.expected_exit_codes, (tuple, list)):
            raise ValueError("expected_exit_codes must be a sequence")
        codes = tuple(self.expected_exit_codes)
        if self.kind != "command":
            if codes:
                raise ValueError(
                    "pytest evaluators do not use expected command exit codes"
                )
        else:
            if self.test_case_count != len(self.test_commands):
                raise ValueError(
                    "command test_case_count must equal the number of commands"
                )
            if not codes:
                codes = (0,) * len(self.test_commands)
            if len(codes) != len(self.test_commands) or any(
                type(code) is not int or not 0 <= code <= 255 for code in codes
            ):
                raise ValueError(
                    "expected_exit_codes must contain one integer in 0..255 per command"
                )
        object.__setattr__(self, "expected_exit_codes", codes)
        if not isinstance(self.payload, dict):
            raise ValueError("private evaluator payload must be a JSON object")
        object.__setattr__(self, "payload", json_value(self.payload))


@dataclass(frozen=True)
class EnvironmentSpec:
    """Dataset-owned environment policy, independent of its grading rule."""

    solver_image: Literal["generic", "task"] = "task"
    grader_dependencies: bool = True
    reuse_solver_wheels: bool = False

    def __post_init__(self) -> None:
        if self.solver_image not in {"generic", "task"}:
            raise ValueError("solver_image policy must be 'generic' or 'task'")
        if type(self.grader_dependencies) is not bool:
            raise ValueError("grader_dependencies must be a boolean")
        if type(self.reuse_solver_wheels) is not bool:
            raise ValueError("reuse_solver_wheels must be a boolean")
        if self.grader_dependencies and self.reuse_solver_wheels:
            raise ValueError(
                "choose reference-derived or solver-requested grader wheels"
            )


@dataclass(frozen=True)
class DatasetTask:
    public: PublicTaskSpec
    private: PrivateEvalSpec
    environment: EnvironmentSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.public, PublicTaskSpec) or not isinstance(
            self.private, PrivateEvalSpec
        ):
            raise ValueError(
                "DatasetTask requires separate public and private specifications"
            )
        if self.private.kind == "nl2repo" and self.public.workdir != "/workspace":
            raise ValueError("NL2Repo requires workdir=/workspace")
        if self.environment is None:
            object.__setattr__(
                self,
                "environment",
                EnvironmentSpec(
                    solver_image="generic"
                    if self.private.kind in {"nl2repo", "autonl2repo"}
                    else "task",
                    grader_dependencies=self.private.kind
                    not in {"denovoswe", "autonl2repo"},
                    reuse_solver_wheels=self.private.kind == "denovoswe",
                ),
            )
        elif not isinstance(self.environment, EnvironmentSpec):
            raise ValueError("environment must be an EnvironmentSpec")

    @property
    def stable_key(self) -> str:
        return self.public.stable_key

    @property
    def key(self) -> str:
        return self.stable_key

    def public_dict(self) -> dict[str, Any]:
        return self.public.public_dict()


@dataclass(frozen=True)
class GradeResult:
    """Controller-private result. An infrastructure failure has no reward.

    ``ok`` includes genuine zero scores. Rewards are required to be finite, not
    assumed binary or globally constrained to any dataset's scoring range.
    """

    status: Literal["ok", "error", "timeout"]
    reward: float | None = None
    resolved: bool = False
    passed: int = 0
    total: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"ok", "error", "timeout"}:
            raise ValueError(f"unsupported grade status: {self.status!r}")
        if type(self.resolved) is not bool:
            raise ValueError("resolved must be a boolean")
        if self.reward is not None:
            if type(self.reward) not in {int, float} or not math.isfinite(self.reward):
                raise ValueError("reward must be a finite number or None")
            object.__setattr__(self, "reward", float(self.reward))
        if self.status == "ok" and self.reward is None:
            raise ValueError("a completed grade requires an explicit reward")
        if self.status != "ok" and (self.reward is not None or self.resolved):
            raise ValueError(
                "failed or timed-out grading cannot have a reward or solve"
            )
        for name in ("passed", "total", "failed", "errors", "skipped"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.details, dict):
            raise ValueError("details must be a JSON object")
        details = json_value(self.details)
        encoded = json.dumps(details, ensure_ascii=False).encode("utf-8")
        if len(encoded) > MAX_DETAILS_BYTES:
            raise ValueError("grade details exceed the controller result size limit")
        object.__setattr__(self, "details", details)

    @property
    def all_tests_passed(self) -> bool:
        """Conservative clean-pass signal, separate from the official capped reward."""
        commands = self.details.get("commands", [])
        return (
            self.status == "ok"
            and self.resolved
            and self.total > 0
            and self.passed == self.total
            and self.failed == self.errors == self.skipped == 0
            and isinstance(commands, list)
            and bool(commands)
            and not self.details.get("commands_truncated", False)
            and self.details.get("command_count", len(commands)) == len(commands)
            and all(
                isinstance(command, dict)
                and command.get("exit_code") == 0
                and not command.get("timed_out", False)
                for command in commands
            )
        )
