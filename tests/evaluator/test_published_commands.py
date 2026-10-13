"""Compatibility must preserve published pytest semantics, not normalize them."""
import json
import os
import shlex
import subprocess

import pytest

from autonl2repo_eval._canonical_runner import (
    ValidationError,
    _direct_pytest_argv,
    _write_direct_runner,
    published_pytest_chunks,
)


LEGACY = ["python", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
CURRENT = [*LEGACY, "-o", "addopts="]
PREFIXES = [LEGACY, CURRENT]


@pytest.mark.parametrize("prefix", PREFIXES, ids=["legacy", "current"])
def test_preserves_prefix_nodes_order_and_process_groups(prefix):
    groups = [["tests/test_b.py::test_b[x y]", "tests/test_a.py::test_a"],
              ["tests/test_b.py::test_b[$(literal);value]"]]
    nodes = [node for group in groups for node in group]
    commands = [shlex.join([*prefix, *group]) for group in groups]
    before = (commands[:], nodes[:])
    chunks = published_pytest_chunks(commands, nodes)
    assert chunks == [(prefix[3:], group) for group in groups]
    assert (commands, nodes) == before
    assert chunks[0][0] is not chunks[1][0]


def test_each_command_keeps_its_own_addopts_policy():
    nodes = ["tests/test_a.py::test_a", "tests/test_b.py::test_b"]
    commands = [shlex.join([*prefix, node]) for prefix, node in zip(PREFIXES, nodes)]
    assert published_pytest_chunks(commands, nodes) == [
        (LEGACY[3:], [nodes[0]]), (CURRENT[3:], [nodes[1]]),
    ]


@pytest.mark.parametrize("prefix", PREFIXES, ids=["legacy", "current"])
@pytest.mark.parametrize("case", ["unknown", "missing", "duplicate", "cross_duplicate",
                                  "extra_option", "shell_suffix", "empty_chunk"])
def test_rejects_invalid_denominators_and_extra_commands(prefix, case):
    a, b = "tests/test_a.py::test_a", "tests/test_b.py::test_b"
    groups = {
        "unknown": [[a, "tests/test_c.py::test_c"]],
        "missing": [[a]],
        "duplicate": [[a, a, b]],
        "cross_duplicate": [[a], [a, b]],
        "extra_option": [["--collect-only", a, b]],
        "shell_suffix": [[a, b, ";", "echo", "unexpected"]],
        "empty_chunk": [[], [a, b]],
    }[case]
    commands = [shlex.join([*prefix, *group]) for group in groups]
    with pytest.raises(ValidationError):
        published_pytest_chunks(commands, [a, b])


@pytest.mark.parametrize("prefix", [
    ["python3", *LEGACY[1:]],
    ["echo", *LEGACY],
    [*LEGACY, "-o", "addopts=--doctest-modules"],
    [*LEGACY, "--override-ini", "addopts="],
    ["python", "-m", "pytest", "--no-header", "-q", "-p", "no:cacheprovider"],
])
def test_only_the_two_known_prefixes_are_accepted(prefix):
    node = "tests/test_a.py::test_a"
    with pytest.raises(ValidationError):
        published_pytest_chunks([shlex.join([*prefix, node])], [node])


@pytest.mark.parametrize("prefix", PREFIXES, ids=["legacy", "current"])
def test_unterminated_quote_is_rejected(prefix):
    with pytest.raises(ValidationError, match="cannot be parsed"):
        published_pytest_chunks([shlex.join(prefix) + " 'unfinished"], ["a.py::test_a"])


@pytest.mark.parametrize("prefix,expected_verbose", [(LEGACY, 0), (CURRENT, -1)],
                         ids=["legacy_inherits_addopts", "current_clears_addopts"])
def test_real_pytest_preserves_configured_addopts(prefix, expected_verbose, tmp_path):
    """Run the consumer's JSON-argv runner and structured outcome plugin."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "pytest.ini").write_text("[pytest]\naddopts = -v\n", encoding="utf-8")
    (workspace / "test_options.py").write_text(
        "import os\n"
        "def test_options(pytestconfig):\n"
        "    assert pytestconfig.getoption('verbose') == int(os.environ['EXPECTED_VERBOSE'])\n",
        encoding="utf-8",
    )
    node = "test_options.py::test_options"
    [(options, nodes)] = published_pytest_chunks([shlex.join([*prefix, node])], [node])
    report_root = tmp_path / "reports"
    report_root.mkdir()
    runner = _write_direct_runner(report_root)
    arguments = report_root / "arguments.json"
    arguments.write_text(json.dumps([*options, *nodes]), encoding="utf-8")
    outcome = report_root / "outcomes.json"
    env = os.environ.copy()
    for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "TASKFORGE_RUNTIME_BEHAVIOR_CONFIG"):
        env.pop(key, None)
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPATH=os.pathsep.join([str(report_root), str(workspace)]),
               TASKFORGE_PYTEST_WORKSPACE=str(workspace),
               TASKFORGE_PYTEST_OUTCOME_FILE=str(outcome),
               EXPECTED_VERBOSE=str(expected_verbose))
    process = subprocess.run(_direct_pytest_argv(runner=runner, arguments_file=arguments),
                             cwd=workspace, env=env, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr
    recorded = json.loads(outcome.read_text(encoding="utf-8"))
    assert recorded["exitstatus"] == 0
    assert recorded["outcomes"] == {node: "passed"}
