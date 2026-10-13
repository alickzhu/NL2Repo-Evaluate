"""Local single-task rollout and scoring. No scheduler, cloud or batch commands."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

from .benchmarks.current_autonl2repo import evaluator_identity
from .ledger import atomic_json, _identifier
from .local import check_docker, load_one, rollout
from .model_profile import ModelProfile, MODEL_REVISIONS, QWEN38_FLASH_NEXT_MODEL, GLM52_FP8_MODEL
from .runner import AGENT_ENVIRONMENT_PROTOCOL, NL2REPO_SOLVER_IMAGE, RunnerConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("rollout", help="Generate one project then evaluate it locally")
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    run.add_argument("--model", default=os.getenv("OPENAI_MODEL"))
    run.add_argument("--api-mode", choices=("chat", "responses"), default="chat")
    run.add_argument("--profile", choices=("qwen", "glm"), default="qwen",
                     help="Existing Chat capability profile; ignored for Responses")
    run.add_argument("--run-id", required=True)
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--docker", default="docker")
    run.add_argument("--solver-image", default=NL2REPO_SOLVER_IMAGE)
    run.add_argument("--generation-timeout", type=int, default=14400)
    run.add_argument("--evaluation-timeout", type=int, default=3600)
    run.add_argument("--max-iterations", type=int, default=500)
    run.add_argument("--memory", default="8g")
    run.add_argument("--cpus", type=int, default=4)
    run.add_argument("--context-tokens", type=int, default=262144)
    run.add_argument("--max-output-tokens", type=int, default=32768)
    run.add_argument("--max-tokens-ceiling", type=int, default=65536)
    run.add_argument("--summary-trigger", type=int, default=196608)
    run.add_argument("--summary-output-tokens", type=int, default=32768)
    run.add_argument("--no-summary", action="store_true")
    run.add_argument("--tokenizer", type=Path)
    run.add_argument("--temperature", type=float)
    run.add_argument("--reasoning-effort", default="high")
    run.add_argument("--request-timeout", type=int, default=7200)
    run.add_argument("--dry-run", action="store_true", help="Validate inputs without Docker or model calls")
    score = commands.add_parser("evaluate", help="Score one existing implementation directory")
    score.add_argument("--task", type=Path, required=True)
    score.add_argument("--submission", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--docker", default="docker")
    score.add_argument("--timeout", type=int, default=3600)
    return parser


def make_profile(args):
    if not args.base_url or not args.model:
        raise ValueError("Provide --base-url and --model, or OPENAI_BASE_URL and OPENAI_MODEL")
    url = urlsplit(args.base_url)
    if url.username or url.password or url.query or url.fragment:
        raise ValueError("API URL must not contain credentials, query parameters or fragments")
    responses = args.api_mode == "responses"
    family = args.model if responses else {"qwen": QWEN38_FLASH_NEXT_MODEL, "glm": GLM52_FP8_MODEL}[args.profile]
    if responses and args.temperature is not None:
        raise ValueError("Responses mode omits sampling parameters")
    return ModelProfile(
        endpoint=args.base_url.rstrip("/"), model=family,
        model_revision=MODEL_REVISIONS.get(family, "endpoint-defined"),
        api_model=args.model, api_mode=args.api_mode,
        temperature=None if responses else (args.temperature if args.temperature is not None else 1.0),
        seed=None if responses else 42,
        reasoning_effort=args.reasoning_effort,
        max_input_tokens=args.context_tokens, max_output_tokens=args.max_output_tokens,
        max_tokens_ceiling=args.max_tokens_ceiling,
        condenser_enabled=not args.no_summary,
        condenser_max_tokens=args.summary_trigger,
        condenser_max_output_tokens=args.summary_output_tokens,
        request_timeout=args.request_timeout,
        custom_tokenizer=str(args.tokenizer.resolve()) if args.tokenizer else None,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        task = load_one(args.task)
        if args.command == "evaluate":
            if not args.submission.is_dir() or args.timeout <= 0:
                raise ValueError("Provide an existing implementation directory and positive timeout")
            if args.output.exists():
                raise ValueError("Output already exists; choose a new result path")
            runtime = check_docker(args.docker)
            from autonl2repo_eval import evaluate
            result = evaluate(task.private.payload["task"], submission_dir=args.submission,
                              timeout_seconds=args.timeout, docker=runtime["cli"])
            atomic_json(args.output, result)
            print(json.dumps({"result": str(args.output), "scoring_valid": result["scoring_valid"],
                              "score": result["score"]}, indent=2))
            raise SystemExit(0 if result["scoring_valid"] else 2)
        profile = make_profile(args)
        config = RunnerConfig(
            run_id=args.run_id, output_root=args.output.resolve(),
            max_iterations=args.max_iterations, trajectory_timeout=args.generation_timeout,
            grading_timeout=args.evaluation_timeout, task_memory=args.memory,
            task_nano_cpus=args.cpus * 1_000_000_000, solver_image=args.solver_image,
            collection_protocol=AGENT_ENVIRONMENT_PROTOCOL,
            dependency_downloads=False, network_policy="legacy_blocklist",
            dependency_cache=args.output.resolve() / args.run_id / "cache",
        )
        # Validate the same run-ID grammar before any filesystem or Docker action.
        _identifier(args.run_id, "run_id")
        if args.dry_run:
            print(json.dumps({"task_id": task.public.task_id, "evaluator": evaluator_identity(),
                              "profile": profile.public_dict(), "runner": asdict(config)}, default=str, indent=2))
            return
        result = rollout(task, config, profile, api_key=os.getenv("OPENAI_API_KEY", "not-needed"),
                         docker_executable=args.docker)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result.get("outcome") == "completed" else 2)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

