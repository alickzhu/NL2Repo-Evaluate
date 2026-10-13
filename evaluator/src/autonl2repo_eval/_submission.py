"""Implementation-directory delivery shared by reference and agent scoring.

This module is also copied into the scoring container. No Git baseline is
assumed: submitted files replace retained files, while unrelated retained
resources stay in place. Frozen test paths are never supplied by a submission.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat


IGNORED = frozenset({".git", ".venv", "venv", "__pycache__", ".pytest_cache"})
PACKAGING = frozenset({"pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in"})


def materialize_implementation(source: Path, destination: Path, *,
                               protected_paths=(), max_files=100000,
                               max_bytes=200 * 1024 * 1024) -> dict:
    """Preflight, then copy an implementation without following destination links.

Never delete unrelated files, silently reinterpret patches, copy VCS metadata,
or let a submitted path replace a directory containing authoritative tests.
An empty implementation is valid input (and is useful as a negative control).
"""
    if source.is_symlink() or not source.is_dir():
        raise ValueError("submission source must be a real directory")
    source = source.resolve()
    if destination.is_symlink():
        raise ValueError("submission destination is a symlink")
    destination = destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("submission source and destination overlap")
    protected = set()
    for value in protected_paths:
        path = PurePosixPath(str(value))
        if not str(value) or path.is_absolute() or ".." in path.parts or path == PurePosixPath("."):
            raise ValueError("unsafe protected submission path")
        protected.add(path.as_posix())

    entries, excluded = [], []
    total = count = 0

    def scan(directory):
        nonlocal total, count
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(source)
            name = relative.as_posix()
            if path.name in IGNORED or path.name.endswith((".pyc", ".pyo")) or name == "start.md":
                continue
            if any(name == item or name.startswith(item + "/") for item in protected):
                excluded.append(name)
                continue
            mode = path.lstat().st_mode
            kind = "link" if stat.S_ISLNK(mode) else "dir" if stat.S_ISDIR(mode) else "file"
            if kind == "file" and not stat.S_ISREG(mode):
                raise ValueError(f"special submission file: {name}")
            if kind != "dir" and any(item.startswith(name + "/") for item in protected):
                raise ValueError(f"submission replaces hidden-test parent: {name}")
            target = destination / relative
            for parent in [target, *target.parents]:
                if parent == destination:
                    break
                if parent != target and parent.is_symlink():
                    raise ValueError(f"destination symlink ancestor: {name}")
            if kind == "dir":
                if target.is_symlink() or (target.exists() and not target.is_dir()):
                    raise ValueError(f"submission directory conflict: {name}")
            elif target.is_dir() and not target.is_symlink():
                raise ValueError(f"submission would replace retained directory: {name}")
            if kind == "link":
                link = os.readlink(path)
                if os.path.isabs(link) or not path.resolve().is_relative_to(source):
                    raise ValueError(f"external submission symlink: {name}")
                # Validate the link in its destination context too: a retained
                # symlink must not redirect a seemingly relative submitted link.
                if not (target.parent / link).resolve().is_relative_to(destination):
                    raise ValueError(f"external destination symlink: {name}")
            count += 1
            if kind == "file":
                total += path.stat().st_size
            if count > max_files or total > max_bytes:
                raise ValueError("submission exceeds delivery budget")
            entries.append((path, target, name, kind, mode & 0o777))
            if kind == "dir":
                scan(path)
    scan(source)
    # Git/workspace delivery carries files, not empty directory scaffolding.
    # In particular, do not recreate an empty hidden-test directory removed by
    # clean: nested overlay storage can expose its old lower-layer contents.
    needed_dirs = {parent.as_posix() for _, _, name, kind, _ in entries if kind != "dir"
                   for parent in PurePosixPath(name).parents}
    entries = [entry for entry in entries if entry[3] != "dir" or entry[2] in needed_dirs]
    destination.mkdir(parents=True, exist_ok=True)
    # Packaging configuration is a coherent unit supplied by the author.
    # A new pyproject must not accidentally execute an old retained setup.py.
    packaging_roots = {path.parent for path, _, _, kind, _ in entries if kind == "file"
                       and path.name in PACKAGING - {"MANIFEST.in"}}
    removed_packaging = []
    for root in sorted(packaging_roots):
        for filename in sorted(PACKAGING):
            relative = (root / filename).relative_to(source)
            name = relative.as_posix()
            if (root / filename).exists() or (root / filename).is_symlink() or name in protected:
                continue
            target = destination / relative
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    raise ValueError(f"retained packaging path is a directory: {name}")
                target.unlink()
                removed_packaging.append(name)
    identity = []
    for path, target, name, kind, mode in entries:
        if kind == "dir":
            target.mkdir(exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        # Unlink first: copy2 must not follow a retained symlink or hardlink.
        target.unlink(missing_ok=True)
        if kind == "link":
            link = os.readlink(path)
            target.symlink_to(link)
            identity.append([name, "symlink", link])
        else:
            shutil.copyfile(path, target)
            target.chmod(mode)
            identity.append([name, mode, hashlib.sha256(target.read_bytes()).hexdigest()])
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return {"contract": "implementation_directory_v1", "file_count": len(identity),
            "bytes": total, "sha256": digest, "excluded_test_paths": excluded,
            "removed_retained_packaging": removed_packaging}
