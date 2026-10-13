from pathlib import Path

import pytest

from autonl2repo_eval._submission import materialize_implementation
from autonl2repo_eval.evaluator import EvaluationError, evaluate, _score_result


def test_real_clean_baseline_conflict_and_binary_modes(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "pyproject.toml").write_text("new metadata\n")
    (target / "pyproject.toml").write_text("retained original metadata\n")
    (target / "requirements.txt").write_text("retained dependency\n")
    (source / "templates").mkdir()
    (target / "templates").mkdir()
    (source / "templates/init.js").write_text("new JS\n")
    (target / "templates/init.js").write_text("old JS\n")
    (source / "binary").write_bytes(bytes(range(256)) + b"\r\n\x00")
    (source / "binary").chmod(0o755)
    (source / "start.md").write_text("spec")
    (source / ".git").mkdir()
    (source / ".git/config").write_text("untrusted config")
    result = materialize_implementation(source, target)
    assert (target / "pyproject.toml").read_text() == "new metadata\n"
    assert (target / "templates/init.js").read_text() == "new JS\n"
    assert (target / "requirements.txt").read_text() == "retained dependency\n"
    assert (target / "binary").read_bytes() == (source / "binary").read_bytes()
    assert (target / "binary").stat().st_mode & 0o777 == 0o755
    assert not (target / ".git").exists() and not (target / "start.md").exists()
    assert result["file_count"] == 3


def test_hidden_test_files_cannot_be_overwritten(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "tests").mkdir(); (target / "tests").mkdir()
    (source / "tests/test_hidden.py").write_text("raise SystemExit(0)")
    (target / "tests/test_hidden.py").write_text("authoritative")
    (source / "tests/own_test.py").write_text("own test")
    result = materialize_implementation(source, target, protected_paths=["tests/test_hidden.py"])
    assert result["excluded_test_paths"] == ["tests/test_hidden.py"]
    assert (target / "tests/test_hidden.py").read_text() == "authoritative"
    assert (target / "tests/own_test.py").read_text() == "own test"


def test_readme_can_be_frozen_test_material(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "README.md").write_text("agent readme")
    (target / "README.md").write_text("authoritative doctest")
    materialize_implementation(source, target, protected_paths=["README.md"])
    assert (target / "README.md").read_text() == "authoritative doctest"


def test_empty_excluded_test_directory_is_not_recreated(tmp_path):
    source = tmp_path / "agent"; source.mkdir()
    (source / "tests").mkdir(); (source / "tests/test_x.py").write_text("excluded")
    materialize_implementation(source, tmp_path / "out", protected_paths=["tests/test_x.py"])
    assert not (tmp_path / "out/tests").exists()


def test_submitted_packaging_does_not_execute_retained_setup(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='new'\n")
    (target / "setup.py").write_text("open('old-HISTORY.rst')")
    (target / "setup.cfg").write_text("obsolete package list")
    (target / "requirements.txt").write_text("dependency")
    result = materialize_implementation(source, target)
    assert result["removed_retained_packaging"] == ["setup.cfg", "setup.py"]
    assert not (target / "setup.py").exists()
    assert (target / "requirements.txt").exists()


def test_overlapping_delivery_roots_rejected(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        materialize_implementation(tmp_path, tmp_path / "subdir")


def test_budget_failure_happens_before_any_copy(tmp_path):
    source = tmp_path / "source"; source.mkdir()
    (source / "x").write_bytes(b"abc")
    with pytest.raises(ValueError, match="budget"):
        materialize_implementation(source, tmp_path / "out", max_bytes=2)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("kind", ["symlink", "file"])
def test_cannot_replace_hidden_test_parent(tmp_path, kind):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "real").mkdir()
    if kind == "symlink":
        (source / "tests").symlink_to("real")
    else:
        (source / "tests").write_text("bad")
    with pytest.raises(ValueError, match="hidden-test parent"):
        materialize_implementation(source, target, protected_paths=["tests/test_x.py"])


def test_destination_links_are_not_followed(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    outside = tmp_path / "untouched"
    outside.write_text("keep")
    (source / "module.py").write_text("new")
    (target / "module.py").symlink_to(outside)
    materialize_implementation(source, target)
    assert outside.read_text() == "keep"
    assert not (target / "module.py").is_symlink()


def test_destination_symlink_directory_rejected_before_copy(tmp_path):
    source, target = tmp_path / "agent", tmp_path / "clean"
    source.mkdir(); target.mkdir()
    (source / "first").write_text("must not copy")
    (source / "package").mkdir(); (source / "package/x.py").write_text("bad")
    (target / "package").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="directory conflict"):
        materialize_implementation(source, target)
    assert not (target / "first").exists()


def test_external_source_symlink_rejected(tmp_path):
    source = tmp_path / "agent"; source.mkdir()
    (source / "bad").symlink_to("../outside")
    with pytest.raises(ValueError, match="external submission symlink"):
        materialize_implementation(source, tmp_path / "clean")


def test_safe_link_and_delivery_hash_are_repeatable(tmp_path):
    source = tmp_path / "agent"; source.mkdir()
    (source / "x").write_bytes(b"source\x00")
    (source / "link").symlink_to("x")
    first = materialize_implementation(source, tmp_path / "one")
    second = materialize_implementation(source, tmp_path / "two")
    assert first == second
    assert (tmp_path / "one/link").read_bytes() == b"source\x00"


def test_empty_directory_is_explicit_not_reference(tmp_path):
    source = tmp_path / "empty"; source.mkdir()
    assert materialize_implementation(source, tmp_path / "clean")["file_count"] == 0
    with pytest.raises(EvaluationError, match="exactly one"):
        evaluate({})
    with pytest.raises(EvaluationError, match="exactly one"):
        evaluate({}, b"patch", submission_dir=source)


def scoreable(passed=1):
    return {"status": "validated" if passed else "parked", "expected": 1,
            "pipeline_stage": "pytest_finished", "passed": passed,
            "accepted_passed_ptp": ["t::x"] if passed else [],
            "non_passed": [] if passed else [{"nodeid": "t::x", "outcome": "failed"}],
            "outcome_counts": {"passed" if passed else "failed": 1},
            "timing": {"pytest_planned_command_count": 1, "pytest_completed_command_count": 1,
                       "pytest_commands": [{"status": "completed", "structured_report_available": True}]}}


@pytest.mark.parametrize("passed", [0, 1])
def test_real_test_failure_is_a_valid_zero(passed):
    result = scoreable(passed)
    _score_result(result, {"passed_ptp": ["t::x"]}, 0)
    assert result["scoring_valid"] and result["status"] == "scored"
    assert result["score"] == passed


@pytest.mark.parametrize("defect", ["before_tests", "denominator", "no_report", "no_execution", "timeout", "exit"])
def test_pipeline_failure_is_not_a_model_zero(defect):
    result = scoreable()
    if defect == "before_tests": result.pop("pipeline_stage")
    if defect == "denominator": result["expected"] = 999
    if defect == "no_report": result["timing"]["pytest_commands"][0]["structured_report_available"] = False
    if defect == "no_execution": result["outcome_counts"] = {"missing": 1}
    if defect == "timeout": result["timing"]["pytest_commands"][0]["timed_out"] = True
    _score_result(result, {"passed_ptp": ["t::x"]}, 1 if defect == "exit" else 0)
    assert not result["scoring_valid"] and result["score"] is None
    assert result["status"] == "evaluator_failed" and not result["accepted"]
