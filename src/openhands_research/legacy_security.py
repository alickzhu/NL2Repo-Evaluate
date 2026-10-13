"""Reuse the repository's legacy blocklist at the trusted tool/RPC boundary.

No new matching rules live here. Development has ordinary network access;
this is a command blocklist, not a network sandbox or proof against evasion.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from .ledger import append_jsonl, atomic_json
from .runtime import TaskContainer, TaskRuntimeError

POLICY = "legacy_blocklist"
BLOCKLIST_PATH = Path(__file__).with_name("blocklist.py")


def blocklist_identity():
    return {
        "source": "openhands_research/blocklist.py",
        "sha256": hashlib.sha256(BLOCKLIST_PATH.read_bytes()).hexdigest(),
    }


def _legacy():
    spec = importlib.util.spec_from_file_location("_repogen_blocklist", BLOCKLIST_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("bundled blocklist.py is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_prompt():
    # Only the specification filename differs from the old server deployment.
    return _legacy().legacy_prompt().replace("start.md", "README.md")


class BlocklistDenied(TaskRuntimeError):
    """The command was blocked BEFORE sending it into the task container."""


class LegacyGuard:
    def __init__(self, paths, row, extra_seeds=(), *, grading_network="none"):
        if os.environ.get("NL2REPO_BLOCKLIST", "1") != "1":
            raise ValueError("NL2REPO_BLOCKLIST must remain enabled")
        self.task_id = row["instance_id"]
        self.session_id = paths.root.name
        self.identity = blocklist_identity()
        # Normalize optional HF nulls, without inventing any extra deny rules.
        row = {
            **row,
            "import_names": row.get("import_names") or [],
            "pypi_name_candidates": [
                item if isinstance(item, dict) else {"name": item}
                for item in row.get("pypi_name_candidates") or []
            ],
        }
        self.rules_path, self.rules_sha256 = _legacy().prepare(
            paths.private / "blocklist", row, extra_seeds
        )
        self.audit_path = paths.root / "blocklist-audit.jsonl"
        self.check("terminal", {"command": ":"})  # Includes audit write preflight.
        atomic_json(
            paths.root / "security.json",
            {
                "policy": POLICY,
                "enforcement": "trusted-controller-before-task-rpc",
                "source": self.identity,
                "rules_sha256": self.rules_sha256,
                "development_network": "bridge",
                "grading_network": grading_network,
                "audit": self.audit_path.name,
            },
        )

    def check(self, tool_name, action):
        event = {
            "event_type": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": action,
            "session_id": self.session_id,
        }
        try:
            if (
                blocklist_identity() != self.identity
                or hashlib.sha256(self.rules_path.read_bytes()).hexdigest()
                != self.rules_sha256
            ):
                raise ValueError("blocklist changed during the run")
            payload = json.dumps(event)
            if len(payload.encode()) > 2 * 1024 * 1024:
                raise ValueError("blocklist input exceeds the legacy hook limit")
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(BLOCKLIST_PATH),
                    str(self.rules_path),
                    str(self.audit_path),
                ],
                input=payload,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
                env={"PATH": os.defpath},
            )
            decision = json.loads(result.stdout)
            if result.returncode == 0 and decision.get("decision") == "allow":
                return
            if result.returncode == 2 and decision.get("decision") == "deny":
                raise BlocklistDenied(
                    "Command not executed: "
                    + str(decision.get("reason", "blocked by security policy")).replace(
                        "start.md", "README.md"
                    )
                )
            raise ValueError("invalid blocklist response")
        except BlocklistDenied:
            raise
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            subprocess.SubprocessError,
        ) as exc:
            # Fail closed even when the old hook cannot start, times out, loses
            # its rules, or cannot write its audit. Never fall back to execution.
            try:
                append_jsonl(
                    self.audit_path,
                    {
                        "task": self.task_id,
                        "decision": "deny",
                        "tool": tool_name,
                        "error": type(exc).__name__,
                        "session_id": self.session_id,
                    },
                )
            except OSError:
                pass
            raise BlocklistDenied(
                "Command not executed: blocklist unavailable; ask the operator "
                "to inspect the security audit."
            ) from exc


class LegacyTaskContainer(TaskContainer):
    """Every agent tool RPC is guarded; trusted setup/capture use separate APIs."""

    _guard: LegacyGuard | None = None

    def bind_guard(self, guard):
        if self._guard is not None or self._labels["org.repogeneration.role"] != "work":
            raise ValueError("blocklist binding requires a fresh work target")
        self._guard = guard

    def call_tool(self, tool_name, action):
        if self._guard is None:
            raise BlocklistDenied("Command not executed: blocklist is not installed")
        self._guard.check(tool_name, action)
        return super().call_tool(tool_name, action)


def enable_online_dependencies(target):
    """Undo only the offline pip policy installed by trusted prepare_work."""
    if not isinstance(target, LegacyTaskContainer) or target._guard is None:
        raise ValueError("online dependencies require an installed legacy blocklist")
    target.write_file("/etc/pip.conf", "[global]\ndisable-pip-version-check = true\n")
