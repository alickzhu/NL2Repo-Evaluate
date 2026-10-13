"""Only the single-task entry point is new; exercise its wiring and failures."""
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhands_research import local
from openhands_research.cli import build_parser, make_profile
from openhands_research.ledger import _identifier
from openhands_research.model_profile import ModelProfile
from openhands_research.runner import RunnerConfig, AGENT_ENVIRONMENT_PROTOCOL


def config(tmp_path):
    return RunnerConfig("single", tmp_path, network_policy="legacy_blocklist",
                        collection_protocol=AGENT_ENVIRONMENT_PROTOCOL,
                        dependency_downloads=False, dependency_cache=tmp_path / "cache")


def test_only_single_task_commands():
    parser = build_parser()
    for name in ("collect", "run-many", "run-config"):
        with pytest.raises(SystemExit):
            parser.parse_args([name])
    args = parser.parse_args(["rollout", "--task", "task.json", "--run-id", "one",
                              "--base-url", "http://localhost:1/v1", "--model", "qwen-test"])
    assert make_profile(args).api_model == "qwen-test"
    args.api_mode = "responses"
    profile = make_profile(args)
    assert profile.model == "qwen-test" and profile.seed is None and profile.temperature is None
    args.base_url = "https://secret@example.com/v1"
    with pytest.raises(ValueError, match="credentials"):
        make_profile(args)


def test_agent_prompt_omits_controller_identifiers():
    from openhands_research.task_prompt import agent_environment_instruction

    task = SimpleNamespace(public=SimpleNamespace(task_id="private-task-id"),
                           stable_key="private-task-key")
    prompt = agent_environment_instruction(task)
    assert prompt.startswith("According to README.md in /workspace,")
    for marker in ("Task:", "Task key:", task.public.task_id, task.stable_key):
        assert marker not in prompt
    assert "No project dependencies are prefetched" in prompt
    assert "implementation or private tests are provided." in prompt


def test_invalid_task_and_output_identifiers(tmp_path):
    with pytest.raises(ValueError, match="task .json"):
        local.load_one(tmp_path)
    for value in ("../x", "one/two", ".", "", "a" * 129):
        with pytest.raises(ValueError):
            _identifier(value, "run_id")


@pytest.mark.parametrize("ready", [True, False])
def test_single_task_is_sequential_and_retains_original_handoff(tmp_path, monkeypatch, ready):
    calls = []
    task = SimpleNamespace(public=SimpleNamespace(image="grading"))
    monkeypatch.setattr(local, "check_docker", lambda _: {"cli": "/docker", "server_version": "test"})
    monkeypatch.setattr(local, "prepare_generic_image", lambda _: calls.append("image-dev"))
    monkeypatch.setattr(local, "prepare_grading_image", lambda _: calls.append("image-grade") or {"digest": "pinned"})
    class Runner:
        def __init__(self, *_args, **_kwargs):
            pass
        def run(self, item, attempt, *, defer_grading):
            assert item is task and attempt == 0 and defer_grading
            calls.append("generation")
            return {"generation_ready": ready, "outcome": "failed"}
        def grade_generated(self, item, attempt, image):
            assert item is task and attempt == 0 and image == {"digest": "pinned"}
            calls.append("score")
            return {"outcome": "completed"}
    monkeypatch.setattr(local, "AttemptRunner", Runner)
    local.rollout(task, config(tmp_path), ModelProfile("http://localhost:1/v1"), api_key="not-needed")
    assert calls == ["image-dev", "image-grade", "generation"] + (["score"] if ready else [])
    with pytest.raises(FileExistsError):
        local.rollout(task, config(tmp_path), ModelProfile("http://localhost:1/v1"), api_key="not-needed")


def test_mandatory_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("NL2REPO_BLOCKLIST", "0")
    with pytest.raises(ValueError):
        config(tmp_path)


def test_wrapper_runs_on_this_python_and_preserves_docker_choice(tmp_path, monkeypatch):
    # A source shebang cannot assume /usr/bin/env python3 has the installed deps.
    from openhands_research.benchmarks.score_runtime import command
    monkeypatch.setenv("AUTONL2REPO_DOCKER", "/chosen/docker")
    monkeypatch.setenv("REPOGEN_SCORE_MEMORY", "8g")
    monkeypatch.setenv("REPOGEN_SCORE_CPUS", "4")
    assert command(["inspect", "image"]) == ["/chosen/docker", "inspect", "image"]
    created = command(["create", "--network", "none", "image"])
    assert created[0] == "/chosen/docker" and "--memory" in created
    assert created[-3:] == ["--network", "none", "image"]


def test_installed_docker_wrapper_is_executable(monkeypatch):
    wrapper = Path(sys.executable).with_name("autonl2repo-docker")
    assert os.access(wrapper, os.X_OK)
    monkeypatch.setenv("AUTONL2REPO_DOCKER", sys.executable)
    result = subprocess.run([str(wrapper), "--version"], capture_output=True,
                            text=True, check=True, timeout=30)
    assert result.stdout.startswith("Python ")


def test_public_scoring_handoff_preserves_task_and_submission(tmp_path, monkeypatch):
    from openhands_research.benchmarks import current_autonl2repo as bridge

    identity = bridge.evaluator_identity()
    row = dict(instance_id="local-scoring-test", repo="local-scoring-test",
               document="Implement solution.py.", image="example/image:tag",
               workdir="/workspace", parent_commit="a" * 40,
               passed_ptp=["tests/test_hidden.py::test_answer"], test_case_count=1,
               test_files=["tests/test_hidden.py"], test_patch="PRIVATE_TEST_MATERIAL",
               test_commands=["python -m pytest tests/test_hidden.py::test_answer"])
    task = bridge.task_from_row(row, identity)
    image = {"immutable_image": "example/image@sha256:" + "b" * 64,
             "image_digest": "sha256:" + "b" * 64}
    archive = io.BytesIO()
    payload = b"answer = 42\n"
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        member = tarfile.TarInfo("workspace/solution.py")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))

    def evaluate(received, *, submission_dir, pull, timeout_seconds, docker):
        assert received == {**row, "image": image["immutable_image"]}
        assert (Path(submission_dir) / "solution.py").read_bytes() == payload
        assert not (Path(submission_dir) / "tests").exists()
        assert pull is False and timeout_seconds == 60
        assert Path(docker) == Path(sys.executable).with_name("autonl2repo-docker")
        assert os.access(docker, os.X_OK)
        assert os.environ["REPOGEN_SCORE_MEMORY"] == "2g"
        assert os.environ["REPOGEN_SCORE_CPUS"] == "1.0"
        return dict(instance_id=row["instance_id"], scoring_valid=True, status="scored",
                    image_digest=image["image_digest"], expected=1, passed=1, score=1.0,
                    evaluator_version=identity["version"],
                    canonical_runner_sha256=identity["files"]["_canonical_runner.py"])

    monkeypatch.setattr(bridge.evaluator, "evaluate", evaluate)
    monkeypatch.setenv("REPOGEN_SCORE_MEMORY", "previous")
    monkeypatch.delenv("REPOGEN_SCORE_CPUS", raising=False)
    result = bridge.grade_current(task, archive.getvalue(), timeout=60,
        artifact_dir=tmp_path, image_identity=image, memory="2g", nano_cpus=1_000_000_000)
    assert result.reward == 1.0 and result.resolved
    assert row["image"] == "example/image:tag"
    assert (tmp_path / "canonical_result.json").is_file()
    assert os.environ["REPOGEN_SCORE_MEMORY"] == "previous"
    assert "REPOGEN_SCORE_CPUS" not in os.environ

