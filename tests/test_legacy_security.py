import json
import os
import subprocess
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import docker
import pytest
from openhands.sdk.conversation.state import ConversationState
from openhands.sdk.tool import resolve_tool
from openhands.tools.file_editor.definition import FileEditorAction
from openhands.tools.terminal.definition import TerminalAction

from openhands_research import legacy_security as security
from openhands_research.model_profile import ModelProfile
from openhands_research.runner import (
    AGENT_ENVIRONMENT_PROTOCOL,
    AttemptRunner,
    RunnerConfig,
)
from openhands_research.tools import build_task_tools


@pytest.fixture
def guard(tmp_path):
    return security.LegacyGuard(
        SimpleNamespace(root=tmp_path, private=tmp_path / "private"),
        {
            "instance_id": "owner__target-repo",
            "repo": "target-repo",
            "pypi_name": "target-distribution",
            "import_names": ["target_import"],
            "pypi_name_candidates": [{"name": "other-alias"}],
        },
        ["extra-alias"],
    )


@pytest.mark.parametrize(
    "command",
    [
        "pip install target-repo",
        "uv pip install target-distribution==1.0",
        "python -m pip download target_import",
        "pip install other-alias",
        "pip install extra-alias",
        "git clone https://github.com/o/target-repo",
        "curl https://pypi.org/pypi/target-repo/json",
        'python -c "import target_import, inspect; print(inspect.getsource(target_import))"',
        "unzip target_repo-1.0.whl",
    ],
)
def test_same_legacy_denials_and_audit(guard, command):
    with pytest.raises(security.BlocklistDenied, match="Command not executed"):
        guard.check("terminal", {"command": command})
    audit = json.loads(guard.audit_path.read_text().splitlines()[-1])
    assert audit["decision"] == "deny" and audit["tool_input"]["command"] == command


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests pytest",
        "python -m pip install -e .",
        "pip install ./target-repo",
        "pip uninstall -y target-repo",
        "python -m target_import",
        "curl http://localhost:8080/target-repo",
        'python -c "import target_import; print(target_import.__file__)"',
        "",  # PTY continuation is still checked but is not blocked.
    ],
)
def test_same_legacy_allowances(guard, command):
    guard.check("terminal", {"command": command})
    assert (
        json.loads(guard.audit_path.read_text().splitlines()[-1])["decision"] == "allow"
    )


def test_rules_are_exactly_shared_and_never_scan_file_bodies(guard):
    rules = json.loads(guard.rules_path.read_text())
    variants, patterns, unsplit = security._legacy().build_patterns(rules["seeds"])
    assert (rules["variants"], rules["patterns"], rules["patterns_unsplit"]) == (
        variants,
        patterns,
        unsplit,
    )
    guard.check(
        "file_editor", {"command": "create", "file_text": "pip install target-repo"}
    )
    assert "README.md" in security.legacy_prompt()
    assert "start.md" not in security.legacy_prompt()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_rules",
        "changed_rules",
        "missing_script",
        "changed_script",
        "unwritable_audit",
        "timeout",
        "bad_json",
        "nonzero_allow",
        "empty_allow",
    ],
)
def test_guard_unavailable_never_allows_execution(
    guard, monkeypatch, failure, tmp_path
):
    if failure == "missing_rules":
        guard.rules_path.unlink()
    elif failure == "changed_rules":
        guard.rules_path.write_text("{}")
    elif failure in {"missing_script", "changed_script"}:
        script = tmp_path / "untrusted.py"
        if failure == "changed_script":
            script.write_text("print('changed')")
        monkeypatch.setattr(security, "BLOCKLIST_PATH", script)
    elif failure == "unwritable_audit":
        guard.audit_path.unlink()
        guard.audit_path.mkdir()
    elif failure == "timeout":

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], 10)

        monkeypatch.setattr(security.subprocess, "run", timeout)
    else:
        result = {
            "bad_json": (0, "not json"),
            "nonzero_allow": (1, '{"decision":"allow"}'),
            "empty_allow": (0, "{}"),
        }[failure]
        monkeypatch.setattr(
            security.subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=result[0], stdout=result[1]),
        )
    with pytest.raises(security.BlocklistDenied, match="not executed"):
        guard.check("terminal", {"command": "echo should-not-run"})


def test_unsupported_tool_and_schema_fail_closed(guard):
    for name, action in [("request_packages", {}), ("browser", {}), ("terminal", {})]:
        with pytest.raises(security.BlocklistDenied):
            guard.check(name, action)


def make_target():
    client = MagicMock(spec=docker.DockerClient)
    client.api = MagicMock(spec=docker.APIClient)
    client.api.base_url = "http+docker://localhost"
    return security.LegacyTaskContainer(
        "generic",
        "run",
        "task",
        role="work",
        network_disabled=False,
        client=client,
    )


def test_real_sdk_tools_block_before_rpc_and_retain_normal_calls(guard, monkeypatch):
    target = make_target()
    target.bind_guard(guard)
    rpc = MagicMock(return_value={"text": "OK", "command": "echo ok", "exit_code": 0})
    monkeypatch.setattr(security.TaskContainer, "call_tool", rpc)
    tools = [
        resolve_tool(spec, MagicMock(spec=ConversationState))[0]
        for spec in build_task_tools(target)
    ]
    assert [tool.name for tool in tools] == ["terminal", "file_editor"]
    denied = tools[0](TerminalAction(command="pip install target-repo"))
    assert denied.is_error and denied.exit_code == 2 and "not executed" in denied.text
    rpc.assert_not_called()
    allowed = tools[0](TerminalAction(command="pip install -e ."))
    assert allowed.text == "OK" and not allowed.is_error
    rpc.assert_called_once()
    rpc.reset_mock()
    guard.rules_path.unlink()
    denied = tools[1](FileEditorAction(command="view", path="/workspace/README.md"))
    assert denied.is_error and "not executed" in denied.text
    rpc.assert_not_called()


def test_guard_required_and_not_replaceable(guard):
    target = make_target()
    with pytest.raises(security.BlocklistDenied, match="not installed"):
        target.call_tool("terminal", {"command": "echo ok"})
    target.bind_guard(guard)
    with pytest.raises(ValueError, match="fresh"):
        target.bind_guard(guard)


def test_online_pip_after_bootstrap_requires_guard(guard, monkeypatch):
    target = make_target()
    write = MagicMock()
    monkeypatch.setattr(target, "write_file", write)
    with pytest.raises(ValueError):
        security.enable_online_dependencies(target)
    write.assert_not_called()
    target.bind_guard(guard)
    security.enable_online_dependencies(target)
    path, content = write.call_args.args
    assert (
        path == "/etc/pip.conf"
        and "no-index" not in content
        and "find-links" not in content
    )


