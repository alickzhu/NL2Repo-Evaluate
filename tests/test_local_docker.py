"""Real Docker generation/capture, scripted model; no scoring-result claim."""
import json
import os
import tarfile
import uuid

import docker
import pytest
from litellm import ModelResponse
from openhands.sdk import LLM
from openhands_research.benchmarks.current_autonl2repo import task_from_row, evaluator_identity
from openhands_research.local import check_docker
from openhands_research.model_profile import ModelProfile
from openhands_research.runner import RunnerConfig, AttemptRunner, AGENT_ENVIRONMENT_PROTOCOL


@pytest.mark.integration
def test_real_docker_generation_and_capture(tmp_path, monkeypatch):
    image = os.environ.get("AUTONL2REPO_TEST_IMAGE")
    if not image:
        pytest.skip("Set AUTONL2REPO_TEST_IMAGE to a cached generic Bash/Python image")
    check_docker(os.environ.get("AUTONL2REPO_TEST_DOCKER", "docker"))
    row = dict(instance_id="public-smoke-test", repo="public-smoke-test",
               document="Write solution.py with answer = 42.",
               image=image, workdir="/workspace", parent_commit="a" * 40,
               passed_ptp=["tests/test_hidden.py::test_answer"], test_case_count=1,
               test_files=["tests/test_hidden.py"], test_patch="PRIVATE_NEVER_SHOWN",
               test_commands=["python -m pytest tests/test_hidden.py::test_answer"])
    task = task_from_row(row, evaluator_identity())
    run_id = "smoke-" + uuid.uuid4().hex
    settings = RunnerConfig(run_id, tmp_path, max_iterations=3, trajectory_timeout=90,
        grading_timeout=30, task_memory="2g", solver_image=image,
        dependency_downloads=False, network_policy="legacy_blocklist",
        collection_protocol=AGENT_ENVIRONMENT_PROTOCOL, dependency_cache=tmp_path / "cache")
    requests = []
    async def transport(llm, *, messages, **kwargs):
        requests.append(messages)
        assert "PRIVATE_NEVER_SHOWN" not in json.dumps(messages)
        index = len(requests)
        function = {"name": "terminal", "arguments": json.dumps({
            "command": "printf 'answer = 42\\n' > /workspace/solution.py", "timeout": 5,
        })} if index == 1 else {"name": "finish", "arguments": '{"message":"Done"}'}
        return ModelResponse(id=f"smoke-{index}", choices=[{"index": 0, "finish_reason": "tool_calls",
            "message": {"role": "assistant", "content": "Working", "reasoning_content": "PUBLIC_REASONING",
                "tool_calls": [{"id": f"call_{index}", "type": "function", "function": function}]}}])
    monkeypatch.setattr(LLM, "_atransport_call", transport)
    runner = AttemptRunner(settings, ModelProfile("http://localhost:1/v1", condenser_enabled=False))
    result = runner.run(task, defer_grading=True)
    root = tmp_path / run_id / "tasks" / task.stable_key / "attempt-0000"
    assert result.get("generation_ready"), result
    with tarfile.open(root / "submission.tar") as archive:
        assert archive.extractfile("solution.py").read() == b"answer = 42\n"
        assert "tests/test_hidden.py" not in archive.getnames()
    assert len(requests) == 2
    assert "PUBLIC_REASONING" in (root / "trajectory.jsonl").read_text()
    assert (root / "blocklist-audit.jsonl").is_file()
    client = docker.from_env()
    try:
        assert client.containers.list(all=True, filters={"label": f"org.repogeneration.run={run_id}"}) == []
    finally:
        client.close()
