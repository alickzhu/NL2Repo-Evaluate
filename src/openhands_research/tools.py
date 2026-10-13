"""SDK tool factories bound to immutable, controller-owned task generations."""

from __future__ import annotations

import threading
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Self

from openhands.sdk.tool import (
    Action,
    DeclaredResources,
    Observation,
    Tool,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)
from openhands.tools.file_editor.definition import (
    TOOL_DESCRIPTION as EDITOR_DESCRIPTION,
)
from openhands.tools.file_editor.definition import (
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation
from openhands.tools.terminal.descriptions import UNIX_TOOL_DESCRIPTION
from openhands.tools.terminal.metadata import CmdOutputMetadata
from pydantic import Field

from openhands_research.runtime import (
    TaskContainer,
    TaskRuntimeError,
    normalize_task_path,
)

from .dependencies import WHEELHOUSE_PATH, PackageBroker
from .legacy_security import BlocklistDenied

if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.conversation.state import ConversationState


_TERMINAL_KEY = "repogeneration_task_terminal"
_EDITOR_KEY = "repogeneration_task_file_editor"
_PACKAGES_KEY = "repogeneration_task_request_packages"
_TARGET_LOCK = threading.Lock()
_TARGETS: weakref.WeakValueDictionary[str, TaskContainer] = (
    weakref.WeakValueDictionary()
)
_BROKERS: weakref.WeakValueDictionary[str, PackageBroker] = (
    weakref.WeakValueDictionary()
)

_TERMINAL_DESCRIPTION = UNIX_TOOL_DESCRIPTION.replace(
    "* Soft timeout: Commands pause for confirmation after 10 seconds without\n"
    "  new output unless you provide a longer `timeout`.",
    "* Soft timeout: Each call waits at most 10 seconds by default (wall clock),\n"
    "  or `timeout` seconds in [0, 300]. The command continues after a timeout;\n"
    "  it is NOT paused, rerun, or automatically restarted.",
).replace(
    "  changes, and running processes.",
    "  changes. Foreground/session jobs are killed; detached jobs may survive.",
) + (
    "\n\nTask-container bridge:\n"
    "* All commands run in one PTY-backed Bash INSIDE the task container, not\n"
    "  on the controller. stdout/stderr are combined. No host filesystem, Docker\n"
    "  socket, or controller/model credentials are mounted or forwarded.\n"
    "* After exit code -1, send command='' (with either is_input value) to\n"
    "  collect output. A new nonempty command is refused until that completes.\n"
    "* Text input gets a final newline; named control/navigation keys are raw\n"
    "  key sequences. Input requires a pending command; text lines may contain\n"
    "  at most 4095 UTF-8 bytes (the PTY canonical-line limit).\n"
    "* exit/exec or changing terminal prompts/redirections can close or break\n"
    "  the shell protocol. Recovery requires explicit reset=true.\n"
    "* JSON arguments are limited to 64 KiB; output retains a 64 KiB tail per\n"
    "  call, with clipping indicated. No full-output file is written on the host.\n"
    "* Transport failures have unknown outcomes. Never assume a failed response\n"
    "  means the command was not executed; use continuation/inspection, not retry."
)

_EDITOR_LIMITS = (
    "\n\nTask-container bridge limits (override more general guidance above):\n"
    "* Absolute paths refer ONLY to the task filesystem, not the controller.\n"
    "* UTF-8 text without NUL only, at most 1 MiB per file. Images, symlink\n"
    "  targets and special files are not supported. Parent directories must\n"
    "  already exist for create. No encoding detection or external formatter runs.\n"
    "* Exact, unique replacement only: no whitespace normalization or fallback.\n"
    "  Omitted new_str deletes old_str; identical replacements are rejected.\n"
    "* Writes use atomic replacement and preserve regular-file permission bits.\n"
    "* Undo retains up to 32 edits / 8 MiB globally for this task service. It\n"
    "  refuses to overwrite changes made outside this editor. Undo of create\n"
    "  removes the created file. History is lost if the service/container exits.\n"
    "* Directory views omit dotfiles, never follow directory symlinks, and show\n"
    "  at most 1000 entries over two levels. Text views are clipped at 64 KiB\n"
    "  characters. RPC JSON arguments are limited to 64 KiB UTF-8 bytes.\n"
    "* A failed transport is never automatically retried; inspect the file\n"
    "  before deciding whether an edit needs to be issued again."
)


def _target(target_id: str) -> TaskContainer:
    with _TARGET_LOCK:
        target = _TARGETS.get(target_id)
    if target is None or target.owner_id != target_id:
        raise TaskRuntimeError("Unknown or expired task target; retargeting is refused")
    return target


class TaskTerminalExecutor(ToolExecutor[TerminalAction, TerminalObservation]):
    def __init__(self, target: TaskContainer) -> None:
        self._target = target

    def __call__(
        self,
        action: TerminalAction,
        conversation: LocalConversation | None = None,
    ) -> TerminalObservation:
        del conversation
        try:
            result = dict(
                self._target.call_tool(
                    "terminal", action.model_dump(mode="json", exclude={"kind"})
                )
            )
            text = result.pop("text")
            if not isinstance(text, str):
                raise ValueError("Invalid terminal text response")
            # Task data must never select a host output persistence directory.
            result["full_output_save_dir"] = None
            return TerminalObservation.from_text(text, **result)
        except BlocklistDenied as exc:
            return TerminalObservation.from_text(
                str(exc),
                is_error=True,
                command=action.command,
                exit_code=2,
                metadata=CmdOutputMetadata(working_dir=self._target.workdir),
                full_output_save_dir=None,
            )
        except (TaskRuntimeError, ValueError, KeyError, TypeError) as exc:
            return TerminalObservation.from_text(
                f"Task terminal failed ({type(exc).__name__}). "
                "Outcome may be unknown; "
                "the action was not retried. "
                "Use continuation/inspection or explicit reset.",
                is_error=True,
                command=action.command,
                exit_code=None,
                metadata=CmdOutputMetadata(working_dir=self._target.workdir),
                full_output_save_dir=None,
            )

    def interrupt(self) -> None:
        self._target.interrupt()


class TaskFileEditorExecutor(ToolExecutor[FileEditorAction, FileEditorObservation]):
    def __init__(self, target: TaskContainer) -> None:
        self._target = target

    def __call__(
        self,
        action: FileEditorAction,
        conversation: LocalConversation | None = None,
    ) -> FileEditorObservation:
        del conversation
        try:
            result = dict(
                self._target.call_tool(
                    "file_editor", action.model_dump(mode="json", exclude={"kind"})
                )
            )
            text = result.pop("text")
            if not isinstance(text, str):
                raise ValueError("Invalid editor text response")
            return FileEditorObservation.from_text(text, **result)
        except BlocklistDenied as exc:
            return FileEditorObservation.from_text(
                str(exc),
                is_error=True,
                command=action.command,
                path=action.path,
            )
        except (TaskRuntimeError, ValueError, KeyError, TypeError) as exc:
            return FileEditorObservation.from_text(
                f"Task editor failed ({type(exc).__name__}). Outcome may be unknown; "
                "the edit was not retried. Inspect the task file before editing again.",
                is_error=True,
                command=action.command,
                path=action.path,
            )


class PackageRequestAction(Action):
    requirements: list[str] = Field(
        min_length=1,
        max_length=32,
        description="Named PyPI requirements, e.g. ['requests>=2', 'setuptools>=77']",
    )


class PackageRequestObservation(Observation):
    packages: list[str] = Field(default_factory=list)


class PackageRequestExecutor(
    ToolExecutor[PackageRequestAction, PackageRequestObservation]
):
    def __init__(self, broker: PackageBroker) -> None:
        self._broker = broker

    def __call__(
        self,
        action: PackageRequestAction,
        conversation: LocalConversation | None = None,
    ) -> PackageRequestObservation:
        del conversation
        try:
            manifest = self._broker.request(action.requirements)
            packages = [
                f"{row['name']}=={row['version']}" for row in manifest["wheels"]
            ]
            return PackageRequestObservation.from_text(
                f"Dependency wheels are available at {WHEELHOUSE_PATH}. "
                "They are not installed. Run pip install in your chosen Python "
                "environment; use the local wheelhouse for build dependencies too. "
                "Available: " + ", ".join(packages),
                packages=packages,
            )
        except ValueError as exc:
            return PackageRequestObservation.from_text(str(exc)[:1000], is_error=True)
        except Exception as exc:
            return PackageRequestObservation.from_text(
                f"Package request failed ({type(exc).__name__}). "
                "The request was not retried; no packages were installed.",
                is_error=True,
            )

    def interrupt(self) -> None:
        self._broker.interrupt()


class RepoGenerationTerminalTool(ToolDefinition[TerminalAction, TerminalObservation]):
    name: ClassVar[str] = "terminal"
    target_id: str

    def declared_resources(self, action: Action) -> DeclaredResources:
        assert isinstance(action, TerminalAction)
        return DeclaredResources(
            keys=(f"rgoh:{self.target_id}", f"terminal:rgoh:{self.target_id}"),
            declared=True,
        )

    @classmethod
    def create(
        cls, conv_state: ConversationState | None, target_id: str
    ) -> Sequence[Self]:
        del conv_state
        target = _target(target_id)
        return [
            cls(
                target_id=target_id,
                description=_TERMINAL_DESCRIPTION
                + f"\nInitial task cwd: {target.workdir}",
                action_type=TerminalAction,
                observation_type=TerminalObservation,
                executor=TaskTerminalExecutor(target),
                annotations=ToolAnnotations(
                    title="terminal",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


class RepoGenerationFileEditorTool(
    ToolDefinition[FileEditorAction, FileEditorObservation]
):
    name: ClassVar[str] = "file_editor"
    target_id: str
    task_workdir: str

    def declared_resources(self, action: Action) -> DeclaredResources:
        assert isinstance(action, FileEditorAction)
        path = normalize_task_path(action.path, self.task_workdir)
        return DeclaredResources(
            # The shared task key also excludes concurrent foreground terminal
            # writes; a lexical file key alone cannot account for task symlinks.
            keys=(f"rgoh:{self.target_id}", f"file:rgoh:{self.target_id}:{path}"),
            declared=True,
        )

    @classmethod
    def create(
        cls, conv_state: ConversationState | None, target_id: str
    ) -> Sequence[Self]:
        del conv_state
        target = _target(target_id)
        return [
            cls(
                target_id=target_id,
                task_workdir=target.workdir,
                description=(
                    EDITOR_DESCRIPTION
                    + _EDITOR_LIMITS
                    + f"\nYour task working directory is: {target.workdir}"
                ),
                action_type=FileEditorAction,
                observation_type=FileEditorObservation,
                executor=TaskFileEditorExecutor(target),
                annotations=ToolAnnotations(
                    title="file_editor",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


class RepoGenerationPackageTool(
    ToolDefinition[PackageRequestAction, PackageRequestObservation]
):
    name: ClassVar[str] = "request_packages"
    target_id: str

    def declared_resources(self, action: Action) -> DeclaredResources:
        assert isinstance(action, PackageRequestAction)
        return DeclaredResources(keys=(f"rgoh:{self.target_id}",), declared=True)

    @classmethod
    def create(
        cls, conv_state: ConversationState | None, target_id: str
    ) -> Sequence[Self]:
        del conv_state
        target = _target(target_id)
        with _TARGET_LOCK:
            broker = _BROKERS.get(target_id)
        if broker is None or broker.target is not target:
            raise TaskRuntimeError(
                "Unknown or expired package broker; retargeting is refused"
            )
        return [
            cls(
                target_id=target_id,
                description=(
                    "Request dependency wheels from PyPI for this Python runtime. "
                    "Choose packages and versions yourself. This fetches the named "
                    "requirements and their dependencies; it installs nothing. "
                    "Use the terminal to install them in your chosen environment "
                    "and to install your generated local project. Task internet "
                    "access stays disabled. Target packages and aliases, URLs, "
                    "VCS, alternate indexes, local paths, and source distributions "
                    "are forbidden. Limits: 32 requirements per call, 64 requests "
                    "and 128 MiB downloaded per task. Request time counts against "
                    "the trajectory deadline. Failures are not automatically retried."
                ),
                action_type=PackageRequestAction,
                observation_type=PackageRequestObservation,
                executor=PackageRequestExecutor(broker),
                annotations=ToolAnnotations(
                    title="request_packages",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=True,
                ),
            )
        ]


register_tool(_TERMINAL_KEY, RepoGenerationTerminalTool)
register_tool(_EDITOR_KEY, RepoGenerationFileEditorTool)
register_tool(_PACKAGES_KEY, RepoGenerationPackageTool)


def build_task_tools(
    target: TaskContainer, package_broker: PackageBroker | None = None
) -> list[Tool]:
    """Return specs, not shared instances; each resolution gets fresh executors.

    The controller must retain and close its TaskContainer after artifact capture.
    Closing a conversation/executor does not destroy the controller's container.
    Only the non-secret immutable target ID is serialized into the tool specs.
    """
    with _TARGET_LOCK:
        existing = _TARGETS.get(target.owner_id)
        if existing is not None and existing is not target:
            raise TaskRuntimeError("Task target identity collision")
        _TARGETS[target.owner_id] = target
        if package_broker is not None:
            if package_broker.target is not target:
                raise TaskRuntimeError("Package broker belongs to a different task")
            _BROKERS[target.owner_id] = package_broker
    specs = [
        Tool(name=_TERMINAL_KEY, params={"target_id": target.owner_id}),
        Tool(name=_EDITOR_KEY, params={"target_id": target.owner_id}),
    ]
    if package_broker is not None:
        specs.append(Tool(name=_PACKAGES_KEY, params={"target_id": target.owner_id}))
    return specs
