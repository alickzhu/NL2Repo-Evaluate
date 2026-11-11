"""Static inputs never invent frozen tests or enter the scoring path."""
import json
from types import SimpleNamespace

import pytest
from autonl2repo_eval.evaluator import EvaluationError

from openhands_research import local
from openhands_research.cli import build_parser
from openhands_research.contracts import PrivateEvalSpec
from openhands_research.runner import AttemptRunner, NL2REPO_SOLVER_IMAGE
from test_local import config


def spec(tmp_path):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"instance_id": "org__example--abc", "repo": "example",
                                "pypi_name": "example", "import_names": ["example"],
                                "document": "# Example\nImplement the documented API.\n"}))
    return path


def test_spec_loader_preserves_document_and_full_loader_remains_strict(tmp_path):
    path = spec(tmp_path)
    task = local.load_spec_one(path, NL2REPO_SOLVER_IMAGE)
    assert task.public.prompt == json.loads(path.read_text())["document"]
    assert task.private.kind == "spec_only" and task.private.test_case_count == 0
    assert task.private.test_commands == () and task.private.test_files == ()
    assert task.environment.grader_dependencies is False
    with pytest.raises(EvaluationError, match="missing task fields"):
        local.load_one(path)
    assert build_parser().parse_args(["generate", "--task", str(path), "--run-id", "one"]).command == "generate"


def test_generation_never_prepares_or_runs_scoring(tmp_path, monkeypatch):
    task = local.load_spec_one(spec(tmp_path), NL2REPO_SOLVER_IMAGE)
    calls = []
    monkeypatch.setattr(local, "check_docker", lambda _: {"cli": "/docker"})
    monkeypatch.setattr(local, "prepare_generic_image", lambda _: calls.append("dev-image"))
    def forbidden(*a, **k):
        pytest.fail("static generation entered scoring")
    monkeypatch.setattr(local, "prepare_grading_image", forbidden)
    class Runner:
        def __init__(self, *a, **k): pass
        def run(self, item, attempt, *, defer_grading):
            assert item is task and defer_grading
            calls.append("generate")
            return {"outcome": "completed", "evaluation_status": "not_run", "score": None}
        grade_generated = forbidden
    monkeypatch.setattr(local, "AttemptRunner", Runner)
    result = local.generate(task, config(tmp_path), None, api_key="not-needed")
    assert calls == ["dev-image", "generate"] and result["score"] is None


def test_no_scoring_material_in_spec_only_contract(tmp_path):
    with pytest.raises(ValueError, match="cannot contain"):
        PrivateEvalSpec("spec_only", 1, ("true",), ())
    task = local.load_spec_one(spec(tmp_path), NL2REPO_SOLVER_IMAGE)
    with pytest.raises(ValueError, match="no evaluator"):
        AttemptRunner.grade_generated(SimpleNamespace(), task, 0, {})
