"""One-attempt OpenHands runner with isolated task execution and private grading."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import traceback
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from openhands.sdk import Agent, AgentContext
from openhands.sdk.conversation.impl.local_conversation import LocalConversation

from . import __version__
from .benchmarks import get_benchmark
from .benchmarks.current_autonl2repo import CURRENT_CONTRACT, grade_current
from .contracts import DatasetTask, GradeResult
from .dependencies import (
    WHEELHOUSE_PATH,
    DependencyWheelhouse,
    PackageBroker,
    checked_requirement,
)
from .grading import validate_submission
from .ledger import AttemptPaths, RunLedger, append_jsonl, atomic_json
from .legacy_security import (
    POLICY,
    LegacyGuard,
    LegacyTaskContainer,
    blocklist_identity,
    enable_online_dependencies,
    legacy_prompt,
)
from .length_retry import OutputLengthLimitError
from .model_profile import ModelProfile
from .runtime import TaskCleanupError, TaskContainer
from .tools import build_task_tools
from .trajectories import TrajectoryRecorder, export_trajectory
from .task_prompt import agent_environment_instruction

NL2REPO_SOLVER_IMAGE = (
    "nikolaik/python-nodejs:python3.12-nodejs22@"
    "sha256:2bc76406cf6a7ac351c0f96de3bc37fe1c4950d283b29cc302cfabf845416488"
)
AGENT_ENVIRONMENT_PROTOCOL = "agent-environment-directory-v1"


@dataclass(frozen=True)
class RunnerConfig:
    run_id: str
    output_root: Path
    max_iterations: int = 500
    trajectory_timeout: float = 7_200
    grading_timeout: float = 1_200
    task_memory: str = "8g"
    task_nano_cpus: int = 4_000_000_000
    dependency_downloads: bool = True
    dependency_requirements: tuple[str, ...] = ()
    blocked_packages: tuple[str, ...] = ()
    solver_image: str = NL2REPO_SOLVER_IMAGE
    collection_protocol: str = "legacy"
    environment_root: Path | None = None
    dependency_cache: Path | None = None
    python_fallback: str | None = None
    # Historical upstream protocols retain their offline behavior. Current
    # run-config explicitly requires our legacy policy instead.
    network_policy: str = "offline_broker"
    blocklist_extra_seeds: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.collection_protocol != AGENT_ENVIRONMENT_PROTOCOL
                or self.network_policy != POLICY or self.dependency_downloads):
            raise ValueError("Local runs require the current AutoNL2Repo protocol and mandatory blocklist")
        if self.network_policy not in {"offline_broker", POLICY}:
            raise ValueError("unknown network policy")
        if self.network_policy == POLICY and (
            self.collection_protocol != AGENT_ENVIRONMENT_PROTOCOL
            or self.dependency_downloads
            or os.environ.get("NL2REPO_BLOCKLIST", "1") != "1"
        ):
            raise ValueError(
                "legacy_blocklist requires agent-owned setup, no package broker, "
                "and NL2REPO_BLOCKLIST enabled"
            )
        if not isinstance(self.blocklist_extra_seeds, dict) or any(
            not isinstance(key, str)
            or not isinstance(values, list)
            or not all(isinstance(value, str) and value.strip() for value in values)
            for key, values in self.blocklist_extra_seeds.items()
        ):
            raise ValueError("blocklist_extra_seeds must map task IDs to alias lists")
        if (
            self.python_fallback is not None
            and re.fullmatch(r"3\.(?:9|1[0-4])\.\d+", self.python_fallback) is None
        ):
            raise ValueError("Python fallback must be an exact supported version")
        if self.collection_protocol not in {
            "legacy",
            "prepared-directory-v1",
            AGENT_ENVIRONMENT_PROTOCOL,
        }:
            raise ValueError("unknown collection protocol")
        if self.collection_protocol == "prepared-directory-v1" and (
            self.environment_root is None or self.dependency_cache is None
        ):
            raise ValueError("prepared collection requires environment and cache paths")
        if self.collection_protocol == AGENT_ENVIRONMENT_PROTOCOL and (
            self.environment_root is not None
            or self.python_fallback is not None
            or self.dependency_cache is None
        ):
            raise ValueError(
                "agent-owned setup requires a dependency cache but no prepared "
                "environment or Python fallback"
            )
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        for name in ("trajectory_timeout", "grading_timeout"):
            value = getattr(self, name)
            if type(value) not in {int, float} or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.task_memory, str) or not self.task_memory:
            raise ValueError("task_memory must be a nonempty Docker limit")
        if type(self.task_nano_cpus) is not int or self.task_nano_cpus <= 0:
            raise ValueError("task_nano_cpus must be positive")
        if type(self.dependency_downloads) is not bool:
            raise ValueError("dependency_downloads must be a boolean")
        if (
            not isinstance(self.solver_image, str)
            or not self.solver_image
            or any(
                character.isspace() or ord(character) < 32
                for character in self.solver_image
            )
        ):
            raise ValueError("solver_image must be a container image reference")
        for name in ("dependency_requirements", "blocked_packages"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)) or len(values) > 64:
                raise ValueError(f"{name} must be a bounded requirement list")
            for value in values:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must contain strings")
                requirement = checked_requirement(value, set())
                if name == "blocked_packages" and str(requirement) != requirement.name:
                    raise ValueError("blocked_packages accepts package names only")
            object.__setattr__(self, name, tuple(values))


class EventRecorder:
    """Persist redacted Pydantic event payloads after SDK state persistence."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self._lock = threading.Lock()

    def __call__(self, event: Any) -> None:
        payload = {
            "event_type": type(event).__name__,
            "event": event.model_dump(mode="json"),
        }
        with self._lock:
            append_jsonl(self.path, payload)
            self.count += 1


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _has_durable_capture(paths: AttemptPaths) -> bool:
    """Check both archive bytes against the durable workspace-captured event."""
    try:
        workspace_hash = _file_sha256(paths.root / "workspace.tar")
        submission_hash = _file_sha256(paths.root / "submission.tar")
        for line in paths.lifecycle.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event.get("status") != "workspace_captured":
                continue
            details = event.get("details") or {}
            if (
                details.get("workspace_sha256") == workspace_hash
                and details.get("submission_sha256") == submission_hash
            ):
                return True
    except (OSError, ValueError, TypeError):
        return False
    return False


class _RunnableConversation(Protocol):
    async def arun(self) -> None: ...

    def interrupt(self) -> None: ...


async def _run_with_deadline(
    conversation: _RunnableConversation, timeout: float
) -> bool:
    """Run asynchronously; return True only when the trajectory deadline fires."""
    task = asyncio.create_task(conversation.arun())
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        return False
    except TimeoutError:
        conversation.interrupt()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AttemptRunner:
    def __init__(
        self,
        config: RunnerConfig,
        profile: ModelProfile,
        *,
        api_key: str = "not-needed",
    ) -> None:
        self.config = config
        self.profile = profile
        self.api_key = api_key
        self.blocklist_identity = (
            blocklist_identity() if config.network_policy == POLICY else None
        )
        self.ledger = RunLedger(config.output_root, config.run_id)
        self.ledger.publish_config(
            {
                "schema_version": 1,
                "run_id": config.run_id,
                "runner": {
                    "max_iterations": config.max_iterations,
                    "trajectory_timeout": config.trajectory_timeout,
                    "grading_timeout": config.grading_timeout,
                    "task_memory": config.task_memory,
                    "task_nano_cpus": config.task_nano_cpus,
                    "dependency_downloads": config.dependency_downloads,
                    "submission_build_dependencies": config.dependency_downloads,
                    "dependency_requirements": list(config.dependency_requirements),
                    "blocked_packages": list(config.blocked_packages),
                    "solver_image": config.solver_image,
                    "network_policy": config.network_policy,
                    "blocklist_extra_seeds": config.blocklist_extra_seeds,
                },
                "model_profile": profile.public_dict(),
                "sdk_version": version("openhands-sdk"),
                "harness_version": __version__,
                "execution": "controller-local-conversation/task-docker-tools-v1",
                "environment_protocol": "dataset-owned-environments-v4",
                **(
                    {
                        "collection_protocol": config.collection_protocol,
                        "prepared_environment_root": str(config.environment_root),
                        "dependency_cache": str(config.dependency_cache),
                        "network_policy": "offline-cached-dependencies-v1",
                        "python_fallback": config.python_fallback,
                    }
                    if config.collection_protocol == "prepared-directory-v1"
                    else {}
                ),
                **(
                    {
                        "collection_protocol": AGENT_ENVIRONMENT_PROTOCOL,
                        "dependency_cache": str(config.dependency_cache),
                        "dependency_preparation": "agent",
                        "network_policy": (
                            POLICY
                            if config.network_policy == POLICY
                            else "offline-model-requested-runtime-plus-build-system-v2"
                        ),
                    }
                    if config.collection_protocol == AGENT_ENVIRONMENT_PROTOCOL
                    else {}
                ),
                **(
                    {"blocklist": self.blocklist_identity}
                    if config.network_policy == POLICY
                    else {}
                ),
            }
        )

    def _work_image(self, task: DatasetTask) -> str:
        return self.config.solver_image

    def _target(self, task: DatasetTask, role: str) -> TaskContainer:
        online = role == "work" and self.config.network_policy == POLICY
        container_type = LegacyTaskContainer if online else TaskContainer
        return container_type(
            self._work_image(task) if role == "work" else task.public.image,
            self.config.run_id,
            task.stable_key,
            role=role,
            workdir=task.public.workdir,
            network_disabled=not online,
            mem_limit=self.config.task_memory,
            nano_cpus=self.config.task_nano_cpus,
        )

    @contextmanager
    def _capture_preserving_target(
        self,
        task: DatasetTask,
        role: str,
        paths: AttemptPaths,
        cleanup_diagnostics: list[dict[str, Any]],
    ):
        """Keep a durable submission handoff despite a later owned-container cleanup error."""
        target = self._target(task, role)

        def record_diagnostics() -> None:
            for diagnostic in target.cleanup_diagnostics:
                if diagnostic not in cleanup_diagnostics:
                    cleanup_diagnostics.append(diagnostic)

        try:
            opened = target.__enter__()
        except BaseException:
            record_diagnostics()
            raise

        try:
            yield opened
        except BaseException as body_error:
            try:
                suppressed = target.__exit__(
                    type(body_error), body_error, body_error.__traceback__
                )
            except BaseException as cleanup_error:
                record_diagnostics()
                try:
                    body_error.cleanup_diagnostics = target.cleanup_diagnostics
                except (AttributeError, TypeError):
                    pass
                raise body_error from cleanup_error
            finally:
                record_diagnostics()
            if not suppressed:
                raise
        else:
            try:
                target.__exit__(None, None, None)
            except TaskCleanupError:
                if not _has_durable_capture(paths):
                    raise
            finally:
                record_diagnostics()

    def _agent(self, task: DatasetTask, paths: AttemptPaths) -> Agent:
        llm = self.profile.build_agent_llm(paths.completions, self.api_key)
        condenser = self.profile.build_condenser(paths.completions, self.api_key)
        recorder = TrajectoryRecorder(paths)
        llm.telemetry.set_log_completions_callback(partial(recorder.record, "solver"))
        if condenser is not None:
            condenser.llm.telemetry.set_log_completions_callback(
                partial(recorder.record, "condenser")
            )
        context = AgentContext(
            system_message_suffix=(
                "This benchmark uses an isolated task filesystem. All terminal and "
                f"file-editor paths refer to the task environment. Its working "
                f"directory is {task.public.workdir}. The controller filesystem is "
                "not a task resource and is intentionally unavailable."
                + (
                    " The public specification's Python version and declared "
                    "third-party "
                    "dependencies are prepared. The target project is not installed. "
                    "Use ordinary pip install for cached dependencies "
                    "or your own local "
                    "project; pip uses a verified offline wheelhouse. "
                    if self.config.collection_protocol == "prepared-directory-v1"
                    else " You are responsible for project-specific "
                    "environment setup and "
                    "dependency selection. No project dependencies are prefetched. "
                )
                + (
                    "Use ordinary pip install to obtain third-party dependencies "
                    "and pip install -e . for your own project. "
                    if self.config.network_policy == POLICY
                    else "Use request_packages to obtain the dependency wheels you need, "
                    f"then install them from {WHEELHOUSE_PATH} with pip in your "
                    "chosen environment. "
                    if self.config.dependency_downloads
                    else "Package downloads are disabled for this run. "
                )
                + "Install your generated local project as needed. "
                + (
                    "Internet and localhost access are available. The mandatory "
                    "legacy command blocklist forbids retrieving the target "
                    "package/source. Virtual environments and caches are excluded "
                    "from submission. Only relative symlinks within the "
                    "implementation are supported.\n" + legacy_prompt()
                    if self.config.network_policy == POLICY
                    else "Internet access is disabled; localhost is available. "
                    "Do not retrieve the target package or reference code. "
                    "Additional named dependencies use request_packages. "
                    "Virtual environments and caches are excluded from submission. "
                    "Only relative symlinks within the implementation are supported."
                    if self.config.collection_protocol != "legacy"
                    else "Internet access is disabled. Do not download the original "
                    "target package, search for its repository, or recover reference "
                    "code from Git history or caches. "
                    "Create virtual environments under "
                    "/tmp, outside the submitted repository; submission archives "
                    "do not support symlinks."
                )
            ),
            load_user_skills=False,
            load_public_skills=False,
            load_project_skills=False,
            load_memory=False,
        )
        return Agent(
            llm=llm,
            tools=[],
            condenser=condenser,
            agent_context=context,
            system_prompt_kwargs={"cli_mode": True},
            tool_concurrency_limit=1,
        )

    def run(
        self, task: DatasetTask, attempt_index: int = 0, *, defer_grading: bool = False
    ) -> dict[str, Any]:
        current = task.private.payload.get("evaluation_contract") == CURRENT_CONTRACT
        if not current:
            raise ValueError("Only the current AutoNL2Repo evaluation contract is supported")
        if current != (self.config.network_policy == POLICY):
            raise ValueError(
                "current AutoNL2Repo generation requires legacy_blocklist; "
                "historical upstream adapters retain offline_broker"
            )
        if (
            task.private.payload.get("evaluation_contract") == CURRENT_CONTRACT
            and not defer_grading
        ):
            raise ValueError("Use the local sequential rollout entry point")
        if defer_grading and (
            self.config.collection_protocol == "legacy"
            or task.private.payload.get("evaluation_contract")
            != CURRENT_CONTRACT
        ):
            raise ValueError(
                "independent grading requires a directory collection protocol"
            )
        paths = self.ledger.reserve(task, attempt_index)
        atomic_json(
            paths.private / "evaluation.json",
            {"schema_version": 1, "spec": asdict(task.private)},
        )
        recorder = EventRecorder(paths.events)
        conversation: LocalConversation | None = None
        timed_out = False
        execution_status = "not_started"
        generation_error = None
        completion_count = 0
        trajectory_audit: dict[str, Any] | None = None
        raw_archive_path = paths.root / "workspace.tar"
        submission_path = paths.root / "submission.tar"
        broker: PackageBroker | None = None
        requested_wheels = DependencyWheelhouse({}, {"schema_version": 1, "wheels": []})
        cleanup_diagnostics: list[dict[str, Any]] = []
        try:
            guard = (
                LegacyGuard(
                    paths,
                    task.private.payload["task"],
                    [
                        *self.config.blocked_packages,
                        *self.config.blocklist_extra_seeds.get(task.public.task_id, []),
                    ],
                    grading_network="none",
                )
                if current
                else None
            )
            if guard is not None and guard.identity != self.blocklist_identity:
                raise ValueError("shared blocklist changed since run preparation")
            assert task.environment is not None
            adapter = get_benchmark(task.private.kind)
            atomic_json(
                paths.root / "environment.json",
                {
                    **asdict(task.environment),
                    "adapter": task.private.kind,
                    "work_image": self._work_image(task),
                    "grading_image": task.public.image,
                },
            )
            extra = task.public.metadata.get("python_dependencies", [])
            blocked = task.public.metadata.get("blocked_packages", [])
            for values in (extra, blocked):
                if (
                    not isinstance(values, list)
                    or len(values) > 64
                    or not all(isinstance(value, str) for value in values)
                ):
                    raise ValueError(
                        "dataset package policies must be bounded string lists"
                    )
            if not task.environment.grader_dependencies and (
                self.config.dependency_requirements or extra
            ):
                raise ValueError(
                    "this dataset uses provided/model-requested dependencies; "
                    "grader dependency overrides are not supported"
                )
            blocked_names = {
                task.public.task_id.rsplit("/", 1)[-1],
                *self.config.blocked_packages,
                *blocked,
            }
            self.ledger.transition(paths, "starting_work_container")
            with self._capture_preserving_target(
                task, "work", paths, cleanup_diagnostics
            ) as target:
                if guard is not None:
                    if not isinstance(target, LegacyTaskContainer):
                        raise ValueError("legacy guarded task container required")
                    target.bind_guard(guard)
                assert target.container is not None
                target.container.reload()
                self.ledger.transition(
                    paths,
                    "work_container_started",
                    container_id=target.container_id,
                    owner_id=target.owner_id,
                    image=self._work_image(task),
                    image_id=target.container.attrs["Image"],
                )
                instruction = adapter.prepare_work(
                    target,
                    task,
                    work_image=self._work_image(task),
                    blocked_packages=tuple(sorted(blocked_names)),
                )
                if guard is not None:
                    enable_online_dependencies(target)
                instruction = agent_environment_instruction(task, instruction)
                atomic_json(
                    paths.root / "dependencies.json",
                    {
                        "schema_version": 2,
                        "policy": (
                            "online-legacy-blocklist"
                            if guard is not None
                            else "public-prepared-plus-model-requested"
                            if self.config.collection_protocol
                            == "prepared-directory-v1"
                            else (
                                "model-requested-runtime-plus-submission-build-system"
                                if task.private.kind == "denovoswe"
                                and self.config.dependency_downloads
                                else "model-requested"
                            )
                        ),
                        "blocked_packages": sorted(blocked_names),
                        "wheels": [],
                        "requests_enabled": self.config.dependency_downloads,
                        "journal": (
                            "blocklist-audit.jsonl"
                            if guard is not None
                            else "package_requests.jsonl"
                        ),
                    },
                )
                self.ledger.transition(paths, "work_environment_ready")
                atomic_json(paths.root / "agent_input.json", {
                    "instruction": instruction,
                    "specification_path": task.public.workdir + "/README.md",
                    "specification_sha256": hashlib.sha256(task.public.prompt.encode()).hexdigest(),
                })
                agent = self._agent(task, paths).model_copy(
                    update={"tools": build_task_tools(target, broker)}
                )
                (paths.run_root / "profiles").mkdir(mode=0o700, exist_ok=True)
                conversation = LocalConversation(
                    agent=agent,
                    workspace=paths.controller_workspace,
                    persistence_dir=paths.conversation,
                    callbacks=[recorder],
                    max_iteration_per_run=self.config.max_iterations,
                    stuck_detection=True,
                    visualizer=None,
                    profile_store_dir=paths.run_root / "profiles",
                    tags={
                        "runid": paths.run_id,
                        "taskkey": paths.task_key,
                        "attempt": str(paths.attempt_index),
                    },
                )
                conversation.send_message(instruction, sender="benchmark")
                self.ledger.transition(paths, "agent_running")
                try:
                    timed_out = asyncio.run(
                        _run_with_deadline(conversation, self.config.trajectory_timeout)
                    )
                except Exception as exc:
                    if not defer_grading:
                        raise
                    generation_error = type(exc).__name__
                    atomic_json(
                        paths.private / "generation_failure.json",
                        {
                            "exception_type": generation_error,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                if broker is not None:
                    broker.interrupt()
                    if task.environment.reuse_solver_wheels:
                        requested_wheels = broker.snapshot()
                execution_status = conversation.state.execution_status.value
                self.ledger.transition(
                    paths,
                    "agent_stopped",
                    execution_status=execution_status,
                    trajectory_timeout=timed_out,
                    events=recorder.count,
                )
                directory = task.private.payload.get("evaluation_contract") in {
                    CURRENT_CONTRACT,
                }
                raw_archive = (
                    target.capture(task.public.workdir, directory=True)
                    if directory
                    else target.capture(task.public.workdir)
                )
                _write_bytes(raw_archive_path, raw_archive)
                if directory:
                    from .benchmarks.autonl2repo_directory import (
                        validate_directory_submission,
                    )

                    submission = validate_directory_submission(
                        raw_archive, root_name=Path(task.public.workdir).name
                    )
                else:
                    submission = validate_submission(
                        raw_archive, root_name=Path(task.public.workdir).name
                    )
                _write_bytes(submission_path, submission)
                self.ledger.transition(
                    paths,
                    "workspace_captured",
                    workspace_sha256=_file_sha256(raw_archive_path),
                    submission_sha256=_file_sha256(submission_path),
                )
            if conversation is not None:
                conversation.close()
                conversation = None
            completion_count = len(list(paths.completions.glob("*.json")))
            trajectory_audit = export_trajectory(paths, task)
            if defer_grading:
                generation = {
                    "schema_version": 1,
                    "run_id": paths.run_id,
                    "task_key": task.stable_key,
                    "attempt_index": attempt_index,
                    "execution_status": execution_status,
                    "trajectory_timeout": timed_out,
                    "generation_error": generation_error,
                    "event_count": recorder.count,
                    "completion_log_count": completion_count,
                    "trajectory_audit_ok": trajectory_audit["ok"],
                    "sft_rows": trajectory_audit["sft_rows"],
                    "history_reasoning_checked": trajectory_audit[
                        "history_reasoning_checked"
                    ],
                    "workspace_sha256": _file_sha256(raw_archive_path),
                    "submission_sha256": _file_sha256(submission_path),
                    "cleanup_diagnostics": cleanup_diagnostics,
                }
                if cleanup_diagnostics:
                    atomic_json(
                        paths.private / "cleanup.json",
                        {"schema_version": 1, "diagnostics": cleanup_diagnostics},
                    )
                atomic_json(paths.root / "generation.json", generation)
                self.ledger.transition(paths, "awaiting_scoring")
                return {
                    "generation_ready": True,
                    "task_key": task.stable_key,
                    "attempt_index": attempt_index,
                }

        except Exception as exc:
            if broker is not None:
                broker.interrupt()
            if conversation is not None:
                execution_status = conversation.state.execution_status.value
                conversation.close()
            completion_count = len(list(paths.completions.glob("*.json")))
            if trajectory_audit is None:
                trajectory_audit = export_trajectory(paths, task)
            failure_type = "runner_failure"
            cause: BaseException | None = exc
            while cause is not None:
                if isinstance(cause, OutputLengthLimitError):
                    failure_type = "output_length_exhausted"
                    break
                cause = cause.__cause__
            atomic_json(
                paths.private / "failure.json",
                {
                    "failure_type": failure_type,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "cleanup_diagnostics": cleanup_diagnostics,
                },
            )
            if cleanup_diagnostics:
                atomic_json(
                    paths.private / "cleanup.json",
                    {"schema_version": 1, "diagnostics": cleanup_diagnostics},
                )
            grade = GradeResult(
                "error",
                total=task.private.test_case_count,
                details={"reason": failure_type},
            )
            return self.ledger.commit_result(
                paths,
                outcome="failed",
                grade=grade,
                public_details={
                    "execution_status": execution_status,
                    "trajectory_timeout": timed_out,
                    "event_count": recorder.count,
                    "completion_log_count": completion_count,
                    "trajectory_audit_ok": trajectory_audit["ok"],
                    "sft_rows": trajectory_audit["sft_rows"],
                    "failure_type": failure_type,
                    "exception_type": type(exc).__name__,
                    "cleanup_diagnostics": cleanup_diagnostics,
                },
            )

    def grade_generated(
        self, task: DatasetTask, attempt_index: int, image_identity: dict[str, str]
    ) -> dict[str, Any]:
        if self.config.collection_protocol == "legacy":
            raise ValueError("independent grading requires a directory protocol")
        paths = AttemptPaths(
            self.ledger.root,
            self.config.run_id,
            task.stable_key,
            task.public.task_id,
            task.public.sample_index,
            attempt_index,
            self.ledger.root
            / "tasks"
            / task.stable_key
            / f"attempt-{attempt_index:04d}",
        )
        generation = json.loads((paths.root / "generation.json").read_text())
        if (
            generation.get("run_id") != self.config.run_id
            or generation.get("task_key") != task.stable_key
            or generation.get("attempt_index") != attempt_index
            or json.loads((paths.root / "task.json").read_text()) != task.public_dict()
            or _file_sha256(paths.root / "workspace.tar")
            != generation.get("workspace_sha256")
            or _file_sha256(paths.root / "submission.tar")
            != generation.get("submission_sha256")
        ):
            raise ValueError("generated artifact identity changed before grading")
        with (paths.private / "grading.claim").open("x") as claim:
            claim.write(self.config.collection_protocol)
        self.ledger.transition(paths, "grading_started")
        try:
            grade = self._grade_prepared(task, paths, image_identity)
        except Exception as exc:
            atomic_json(
                paths.private / "grading_failure.json",
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            grade = GradeResult(
                "error",
                total=task.private.test_case_count,
                details={"reason": "grading_failure"},
            )
        atomic_json(paths.private / "grade.json", asdict(grade))
        outcome = (
            "timeout"
            if generation["trajectory_timeout"]
            else "completed"
            if generation["execution_status"] == "finished"
            and not generation["generation_error"]
            and grade.status == "ok"
            and generation["trajectory_audit_ok"]
            else "failed"
        )
        return self.ledger.commit_result(
            paths, outcome=outcome, grade=grade, public_details=generation
        )

    def _grade_prepared(self, task, paths, image_identity):
        if task.private.payload.get("evaluation_contract") == CURRENT_CONTRACT:
            self.ledger.transition(
                paths,
                "public_evaluator_started",
                image_digest=image_identity["image_digest"],
            )
            return grade_current(
                task,
                (paths.root / "workspace.tar").read_bytes(),
                timeout=self.config.grading_timeout,
                artifact_dir=paths.private,
                image_identity=image_identity,
                memory=self.config.task_memory,
                nano_cpus=self.config.task_nano_cpus,
            )
        raise ValueError("Only the current AutoNL2Repo scorer is supported")

    def preparation_failed(
        self, task: DatasetTask, attempt_index: int, error: Exception
    ) -> dict[str, Any]:
        root = (
            self.ledger.root
            / "tasks"
            / task.stable_key
            / f"attempt-{attempt_index:04d}"
        )
        if (root / "result.json").exists():
            return json.loads((root / "result.json").read_text())
        paths = (
            AttemptPaths(
                self.ledger.root,
                self.config.run_id,
                task.stable_key,
                task.public.task_id,
                task.public.sample_index,
                attempt_index,
                root,
            )
            if root.exists()
            else self.ledger.reserve(task, attempt_index)
        )
        details = (
            json.loads((root / "generation.json").read_text())
            if (root / "generation.json").exists()
            else {}
        )
        agent_owned = self.config.collection_protocol == AGENT_ENVIRONMENT_PROTOCOL
        atomic_json(
            paths.private
            / (
                "collection_failure.json" if agent_owned else "preparation_failure.json"
            ),
            {"exception_type": type(error).__name__, "message": str(error)},
        )
        return self.ledger.commit_result(
            paths,
            outcome="failed",
            grade=GradeResult(
                "error",
                total=task.private.test_case_count,
                details={
                    "reason": "collection_stage_failed"
                    if agent_owned
                    else "environment_preparation_failed"
                },
            ),
            public_details={
                "trajectory_audit_ok": False,
                "completion_log_count": 0,
                **details,
                "failure_type": "collection_stage_failed",
            },
        )
