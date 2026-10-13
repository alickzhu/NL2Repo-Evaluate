"""Frozen, identity-bound repairs for omitted hidden-test support files.

These are reviewed task-data repairs, not an exception-driven fallback to
copy arbitrary modules from the original implementation. Nothing is fetched
at evaluation time and the input task object is never mutated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_REPAIRS = {
    "jaap3__django-richtextfield--t30--l196--c61ac7ae":
        "django_richtextfield_20260913.json",
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _addition_patch(files: list[dict]) -> str:
    chunks = []
    for file in files:
        path, text, mode = file["path"], file["text"], file["mode"]
        parts = Path(path).parts
        if not parts or Path(path).is_absolute() or ".." in parts or "\x00" in path:
            raise ValueError("unsafe frozen support path")
        if not path.startswith("testproject/") or path.startswith("testproject/tests/"):
            raise ValueError("repair must contain test support, not product or scored tests")
        data = text.encode("utf-8")
        if _digest(data) != file["sha256"]:
            raise ValueError(f"test support content hash mismatch: {path}")
        blob_oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob_oid != file["git_blob_oid"] or mode not in {"100644", "100755"}:
            raise ValueError(f"test support Git identity mismatch: {path}")
        header = f"diff --git a/{path} b/{path}\nnew file mode {mode}\n"
        header += f"index {'0' * 40}..{blob_oid}\n"
        if not text:
            chunks.append(header)
            continue
        lines = text.splitlines(keepends=True)
        header += f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{len(lines)} @@\n"
        body = "".join("+" + line for line in lines)
        if not text.endswith("\n"):
            body += "\n\\ No newline at end of file\n"
        chunks.append(header + body)
    return "".join(chunks)


def prepare_test_support(task: Mapping[str, Any]) -> tuple[dict, dict | None]:
    effective = dict(task)
    name = _REPAIRS.get(str(task.get("instance_id") or ""))
    if name is None:
        return effective, None
    raw = (Path(__file__).with_name("test_support") / name).read_bytes()
    repair = json.loads(raw)
    if repair["instance_id"] != task["instance_id"] or repair["parent_commit"] != task.get("parent_commit"):
        raise ValueError("frozen test support repair does not match task identity")
    nodes = task.get("passed_ptp") or []
    node_hash = _digest(("\n".join(sorted(nodes)) + "\n").encode())
    if node_hash != repair["passed_ptp_sha256"]:
        raise ValueError("frozen test support repair denominator mismatch")
    files = repair["files"]
    added_paths = [file["path"] for file in files]
    source_paths = repair["source_test_files"]
    if len(added_paths) != len(set(added_paths)) or set(added_paths) & set(source_paths):
        raise ValueError("frozen test support paths overlap")
    addition = _addition_patch(files)
    original = str(task.get("test_patch") or "")
    already_applied = original.endswith(addition)
    source_patch = original[:-len(addition)] if already_applied else original
    if _digest(source_patch.encode()) != repair["source_test_patch_sha256"]:
        raise ValueError("task test payload changed; frozen support repair must be reviewed")
    expected_paths = [*source_paths, *added_paths] if already_applied else source_paths
    if list(task.get("test_files") or []) != expected_paths:
        raise ValueError("task test inventory changed; frozen support repair must be reviewed")
    if source_patch and not source_patch.endswith("\n"):
        raise ValueError("source test patch lacks a final newline")
    effective["test_patch"] = source_patch + addition
    effective["test_files"] = [*source_paths, *added_paths]
    provenance = {
        "repair_id": repair["repair_id"],
        "repair_manifest_sha256": _digest(raw),
        "parent_commit": repair["parent_commit"],
        "required_image_digest": repair["image_digest"],
        "source_test_patch_sha256": repair["source_test_patch_sha256"],
        "effective_test_patch_sha256": _digest(effective["test_patch"].encode()),
        "passed_ptp_sha256": node_hash,
        "files": [{key: file[key] for key in ["path", "mode", "sha256", "git_blob_oid"]}
                  for file in files],
    }
    return effective, provenance
