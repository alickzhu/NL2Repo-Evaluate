"""Unit contracts complement the real LocalConversation/Docker smoke tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import docker
import pytest
from openhands_research.runtime import TaskContainer, TaskRuntimeError
from openhands_research.tools import (
    PackageRequestAction,
    PackageRequestObservation,
    RepoGenerationFileEditorTool,
    RepoGenerationTerminalTool,
    TaskFileEditorExecutor,
    TaskTerminalExecutor,
    build_task_tools,
)

from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.tool import Tool, resolve_tool
from openhands.sdk.tool.registry import get_tool_module_qualnames
from openhands.tools.file_editor.definition import (
    FileEditorAction,
    FileEditorObservation,
)
from openhands.tools.terminal.definition import TerminalAction, TerminalObservation


_STATE = MagicMock(spec=ConversationState)


@pytest.fixture
def targets():
    client = MagicMock(spec=docker.DockerClient)
    client.api = MagicMock(spec=docker.APIClient)
    client.api.base_url = "http+docker://localhost"
    return (
        TaskContainer("cached", "run", "one", workdir="/srv/task", client=client),
        TaskContainer("cached", "run", "two", workdir="/srv/task", client=client),
    )


def test_stable_factories_use_standard_names_and_do_not_shadow_builtins(targets):
    target, _ = targets
    specs = build_task_tools(target)
    resolved = [resolve_tool(spec, _STATE)[0] for spec in specs]
    assert [tool.name for tool in resolved] == ["terminal", "file_editor"]
    assert isinstance(resolved[0], RepoGenerationTerminalTool)
    assert isinstance(resolved[1], RepoGenerationFileEditorTool)
    assert resolved[0].action_type is TerminalAction
    assert resolved[0].observation_type is TerminalObservation
    assert resolved[1].action_type is FileEditorAction
    assert resolved[1].observation_type is FileEditorObservation
    registered = get_tool_module_qualnames()
    assert registered["terminal"] == "openhands.tools.terminal.definition"
    assert registered["file_editor"] == "openhands.tools.file_editor.definition"
    for spec in specs:
        assert spec.name.startswith("repogeneration_task_")
        assert spec.params == {"target_id": target.owner_id}
        restored = Tool.model_validate(spec.model_dump(mode="json"))
        assert resolve_tool(restored, _STATE)[0].name in ("terminal", "file_editor")
    target.client.containers.create.assert_not_called()


def test_resolving_creates_fresh_executors_without_retargeting(targets, monkeypatch):
    first, second = targets
    first_call = MagicMock(
        return_value={"text": "ONE", "command": "probe", "exit_code": 0}
    )
    second_call = MagicMock(
        return_value={"text": "TWO", "command": "probe", "exit_code": 0}
    )
    monkeypatch.setattr(first, "call_tool", first_call)
    monkeypatch.setattr(second, "call_tool", second_call)
    first_spec = build_task_tools(first)[0]
    before = resolve_tool(first_spec, _STATE)[0]
    second_tool = resolve_tool(build_task_tools(second)[0], _STATE)[0]
    after = resolve_tool(first_spec, _STATE)[0]
    assert before.executor is not after.executor
    action = TerminalAction(command="probe", timeout=1)
    assert before(action).text == "ONE"
    assert second_tool(action).text == "TWO"
    assert after(action).text == "ONE"
    assert first_call.call_count == 2 and second_call.call_count == 1
    sent_tool, sent_action = first_call.call_args.args
    assert sent_tool == "terminal"
    assert sent_action == {
        "command": "probe",
        "is_input": False,
        "timeout": 1.0,
        "reset": False,
    }


def test_resource_keys_are_task_scoped_lexical_and_include_terminal_conflicts(
    targets, monkeypatch
):
    first, second = targets

    def forbidden(*_args, **_kwargs):
        raise AssertionError("No host path resolution may occur")

    monkeypatch.setattr(Path, "resolve", forbidden)
    terminal, editor = [
        resolve_tool(spec, _STATE)[0] for spec in build_task_tools(first)
    ]
    other_editor = resolve_tool(build_task_tools(second)[1], _STATE)[0]
    first_resources = editor.declared_resources(
        FileEditorAction(command="view", path="folder/../note")
    )
    normalized = editor.declared_resources(
        FileEditorAction(command="view", path="/srv/task/note")
    )
    other_resources = other_editor.declared_resources(
        FileEditorAction(command="view", path="/srv/task/note")
    )
    assert first_resources == normalized and first_resources.declared
    assert any(key.endswith(":/srv/task/note") for key in first_resources.keys)
    assert set(first_resources.keys).isdisjoint(other_resources.keys)
    assert set(first_resources.keys) & set(
        terminal.declared_resources(TerminalAction(command=":")).keys
    )


def test_terminal_observation_never_accepts_task_selected_host_save_dir(
    targets, monkeypatch, tmp_path
):
    target, _ = targets
    forbidden_output = tmp_path / "should-not-exist"
    monkeypatch.setattr(
        target,
        "call_tool",
        MagicMock(
            return_value={
                "text": "z" * 100000,
                "command": "probe",
                "exit_code": 0,
                "metadata": {"exit_code": 0, "working_dir": "/srv/task"},
                "full_output_save_dir": str(forbidden_output),
            }
        ),
    )
    observation = TaskTerminalExecutor(target)(TerminalAction(command="probe"))
    assert not observation.is_error
    assert observation.full_output_save_dir is None
    assert observation.to_llm_content
    assert not forbidden_output.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("tool", ["terminal", "file_editor"])
def test_transport_failures_are_typed_observations_and_never_retried(
    targets, monkeypatch, tool
):
    target, _ = targets
    invocation = MagicMock(side_effect=TaskRuntimeError("PRIVATE-DAEMON-DETAIL"))
    monkeypatch.setattr(target, "call_tool", invocation)
    if tool == "terminal":
        result = TaskTerminalExecutor(target)(TerminalAction(command="printf once"))
        assert isinstance(result, TerminalObservation)
        assert result.full_output_save_dir is None
        assert result.exit_code is None
    else:
        result = TaskFileEditorExecutor(target)(
            FileEditorAction(command="create", path="/srv/task/file", file_text="once")
        )
        assert isinstance(result, FileEditorObservation)
    assert result.is_error and "not retried" in result.text
    assert "PRIVATE-DAEMON-DETAIL" not in result.text
    invocation.assert_called_once()


def test_editor_uses_only_task_rpc_with_exact_unicode_arguments(targets, monkeypatch):
    target, _ = targets
    invocation = MagicMock(
        return_value={
            "text": "File created successfully.",
            "command": "create",
            "path": "/srv/task/ü.txt",
            "prev_exist": False,
            "new_content": "ü\n'\" $HOME $(not-a-command)",
        }
    )
    monkeypatch.setattr(target, "call_tool", invocation)
    action = FileEditorAction(
        command="create",
        path="/srv/task/ü.txt",
        file_text="ü\n'\" $HOME $(not-a-command)",
    )
    result = TaskFileEditorExecutor(target)(action)
    assert not result.is_error and result.new_content == action.file_text
    name, data = invocation.call_args.args
    assert name == "file_editor"
    assert data["file_text"] == action.file_text
    assert "kind" not in data and "target_id" not in data


def test_interrupt_is_forwarded_but_executor_close_keeps_artifacts(
    targets, monkeypatch
):
    target, _ = targets
    interrupt, close = MagicMock(), MagicMock()
    monkeypatch.setattr(target, "interrupt", interrupt)
    monkeypatch.setattr(target, "close", close)
    executor = TaskTerminalExecutor(target)
    executor.interrupt()
    interrupt.assert_called_once()
    executor.close()
    TaskFileEditorExecutor(target).close()
    close.assert_not_called()


def test_unknown_target_never_falls_back_to_a_local_executor(targets):
    target, _ = targets
    spec = build_task_tools(target)[0]
    unknown = Tool(name=spec.name, params={"target_id": "unknown-generation"})
    with pytest.raises(TaskRuntimeError, match="retargeting is refused"):
        resolve_tool(unknown, _STATE)


def test_package_tool_is_task_scoped_and_does_not_install(targets):
    from openhands_research.dependencies import PackageBroker

    target, other = targets
    broker = MagicMock(spec=PackageBroker)
    broker.target = target
    broker.request.return_value = {
        "wheels": [{"name": "requests", "version": "2.32.5"}]
    }
    specs = build_task_tools(target, broker)
    tool = resolve_tool(Tool.model_validate(specs[-1].model_dump(mode="json")), _STATE)[
        0
    ]
    assert tool.name == "request_packages"
    assert specs[-1].params == {"target_id": target.owner_id}
    result = tool(PackageRequestAction(requirements=["requests>=2"]))
    assert isinstance(result, PackageRequestObservation)
    assert not result.is_error and result.packages == ["requests==2.32.5"]
    assert "not installed" in result.text
    broker.request.assert_called_once_with(["requests>=2"])
    target.client.api.exec_create.assert_not_called()
    assert tool.declared_resources(
        PackageRequestAction(requirements=["wheel"])
    ).keys == (f"rgoh:{target.owner_id}",)
    with pytest.raises(TaskRuntimeError, match="different task"):
        build_task_tools(other, broker)
    assert tool.executor is not None
    tool.executor.interrupt()
    broker.interrupt.assert_called_once()


def test_package_tool_reports_denial_without_retry(targets):
    from openhands_research.dependencies import PackageBroker

    target, _ = targets
    broker = MagicMock(spec=PackageBroker)
    broker.target = target
    broker.request.side_effect = ValueError("target package is forbidden")
    tool = resolve_tool(build_task_tools(target, broker)[-1], _STATE)[0]
    result = tool(PackageRequestAction(requirements=["target"]))
    assert result.is_error and "forbidden" in result.text
    broker.request.assert_called_once()
