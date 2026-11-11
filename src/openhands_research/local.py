"""One task, sequential image preparation, generation, then private evaluation."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import docker

from .benchmarks.current_autonl2repo import load_current
from .ledger import atomic_json
from .runner import AttemptRunner
from .preparation import prepare_generic_image, prepare_grading_image
from .runner import SPEC_ONLY_CONTRACT
from .contracts import DatasetTask, PublicTaskSpec, PrivateEvalSpec
from .benchmarks.current_autonl2repo import _object, _blocked_names


def load_spec_one(path: Path, solver_image: str):
    """Load a static delivery without inventing an image, test set or score."""
    if path.is_symlink() or not path.is_file() or path.suffix != ".json":
        raise ValueError("--task must be one non-symlink task .json file")
    raw = path.read_bytes()
    row = json.loads(raw, object_pairs_hook=_object)
    if not isinstance(row, dict) or not isinstance(row.get("document"), str) or not row["document"].strip():
        raise ValueError("task document is required")
    digest = hashlib.sha256(raw).hexdigest()
    return DatasetTask(
        PublicTaskSpec(dataset_id="autonl2repo-spec-only", dataset_revision=digest,
                       task_id=row["instance_id"], prompt=row["document"], image=solver_image,
                       metadata={"source_sha256": digest, "blocked_packages": _blocked_names(row)}),
        PrivateEvalSpec(kind="spec_only", test_case_count=0, test_commands=(), test_files=(),
                        payload={"task": row, "evaluation_contract": SPEC_ONLY_CONTRACT,
                                 "source_sha256": digest}),
    )


def generate(task, config, profile, *, api_key, docker_executable="docker"):
    """Generate and capture from a Spec; never prepare or invoke scoring."""
    if task.private.kind != "spec_only":
        raise ValueError("generate requires a spec_only task")
    runtime = check_docker(docker_executable)
    os.environ["AUTONL2REPO_DOCKER"] = runtime["cli"]
    root = config.output_root / config.run_id
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)
    atomic_json(root / "local_runtime.json", {**runtime, "mode": "generation_only"})
    runner = AttemptRunner(config, profile, api_key=api_key)
    try:
        prepare_generic_image(config.solver_image)
    except Exception as error:
        return runner.preparation_failed(task, 0, error)
    return runner.run(task, 0, defer_grading=True)


def load_one(path: Path):
    if not path.is_file() or path.suffix != ".json" or path.is_symlink():
        raise ValueError("--task must be one complete, non-symlink task .json file")
    tasks = load_current(path)
    if len(tasks) != 1:
        raise ValueError("Exactly one task per invocation is required")
    return tasks[0]


def check_docker(executable="docker"):
    binary = shutil.which(executable)
    if not binary:
        raise ValueError("Docker CLI not found; install Docker Engine and CLI")
    result = subprocess.run([binary, "version", "--format", "{{json .Server}}"],
                            capture_output=True, text=True, timeout=20)
    if result.returncode or "podman" in (result.stdout + result.stderr).lower():
        raise ValueError("Docker Engine is required, not the Podman alias; select the real CLI with --docker")
    server = json.loads(result.stdout)
    client = docker.from_env(timeout=20)
    try:
        api_server = client.version()
        if "podman" in json.dumps(api_server).lower():
            raise ValueError("DOCKER_HOST must point to Docker Engine")
        if server.get("Version") != api_server.get("Version"):
            raise ValueError("Docker CLI and Python API server versions differ")
        # CLI contexts are not understood by docker-py: require equivalent info.
        info = client.info()
        cli = subprocess.run([binary, "info", "--format", "{{.ID}}"],
                             capture_output=True, text=True, check=True, timeout=20)
        if info.get("ID") != cli.stdout.strip():
            raise ValueError("Docker CLI and Python API refer to different daemons; set DOCKER_HOST explicitly")
        return {"cli": str(Path(binary).absolute()), "server_version": api_server["Version"]}
    finally:
        client.close()


def rollout(task, config, profile, *, api_key, docker_executable="docker"):
    """Same run()/grade_generated() handoff as collection, without worker pools."""
    runtime = check_docker(docker_executable)
    os.environ["AUTONL2REPO_DOCKER"] = runtime["cli"]
    root = config.output_root / config.run_id
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir(mode=0o700)  # Refuse existing output; never overwrite a prior attempt.
    atomic_json(root / "local_runtime.json", runtime)
    runner = AttemptRunner(config, profile, api_key=api_key)
    try:
        prepare_generic_image(config.solver_image)
        image = prepare_grading_image(task)
    except Exception as error:
        return runner.preparation_failed(task, 0, error)
    generated = runner.run(task, 0, defer_grading=True)
    if not generated.get("generation_ready"):
        return generated
    # Generation container has been cleaned up before this fresh scoring container.
    return runner.grade_generated(task, 0, image)

