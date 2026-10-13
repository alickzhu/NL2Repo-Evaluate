"""Dependency-only wheel artifacts for network-isolated task environments."""

from __future__ import annotations

import ast
import configparser
import copy
import hashlib
import inspect
import io
import json
import re
import tarfile
import threading
import time
import tomllib
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from packaging.markers import Environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag, parse_tag
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version
from resolvelib import AbstractProvider, BaseReporter, Resolver

from .grading import TaskTarget
from .ledger import append_jsonl, utc_now


MAX_WHEEL_BYTES = 32 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 128 * 1024 * 1024
WHEELHOUSE_PATH = "/opt/openhands-research-wheels"


def checked_requirement(value: str, blocked_names: set[str]) -> Requirement:
    """Accept named index requirements only, never target packages or URLs."""
    requirement = Requirement(value)
    if requirement.url is not None:
        raise ValueError("URL and VCS dependencies are forbidden")
    blocked = {str(canonicalize_name(name)) for name in blocked_names}
    if canonicalize_name(requirement.name) in blocked:
        raise ValueError(f"target package is forbidden: {requirement.name}")
    return requirement


def checked_wheel(
    filename: str,
    content: bytes,
    allowed_names: set[str],
    blocked_names: set[str],
) -> str:
    """Validate distribution identity and dependencies before publishing a wheel."""
    name, version, _build, _tags = parse_wheel_filename(filename)
    allowed = {canonicalize_name(value) for value in allowed_names}
    blocked = {str(canonicalize_name(value)) for value in blocked_names}
    if name not in allowed or name in blocked:
        raise ValueError(f"wheel distribution is not allowed: {name}")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for entry in archive.infolist():
            path = PurePosixPath(entry.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in entry.filename:
                raise ValueError("unsafe wheel member path")
            for part in path.parts:
                module = (
                    part.split(".", 1)[0]
                    if part.endswith((".py", ".pyc", ".so", ".pyd"))
                    else part
                )
                if canonicalize_name(module) in blocked:
                    raise ValueError("wheel contains a forbidden target namespace")
            if entry.filename.endswith(".dist-info/METADATA"):
                if entry.file_size > 1024 * 1024:
                    raise ValueError("wheel metadata exceeds its byte limit")
                bundled = BytesParser().parsebytes(archive.read(entry))
                if canonicalize_name(bundled.get("Name", "")) in blocked:
                    raise ValueError("wheel vendors a forbidden target distribution")
        metadata_files = [
            entry
            for entry in archive.infolist()
            if entry.filename.endswith(".dist-info/METADATA")
            and len(PurePosixPath(entry.filename).parts) == 2
        ]
        if len(metadata_files) != 1 or metadata_files[0].file_size > 1024 * 1024:
            raise ValueError("wheel must contain one bounded METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    if (
        canonicalize_name(metadata.get("Name", "")) != name
        or Version(metadata.get("Version", "0")) != version
    ):
        raise ValueError("wheel filename and metadata identity disagree")
    for dependency in metadata.get_all("Requires-Dist", []):
        checked_requirement(dependency, blocked)
    return str(name)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        raise ValueError("package download redirects are forbidden")


@dataclass(frozen=True)
class WheelCandidate:
    name: str
    version: Version
    filename: str
    url: str
    sha256: str
    extras: frozenset[str] = frozenset()


class WheelProvider(AbstractProvider[Requirement, WheelCandidate, str]):
    """Resolve wheel metadata only; never run a build backend or package code."""

    def __init__(
        self,
        blocked_names: set[str],
        tags: set[Tag],
        environment: Environment | dict[str, str],
        timeout: float = 300,
        cancelled: threading.Event | None = None,
        byte_limit: int = MAX_DOWNLOAD_BYTES,
    ) -> None:
        self.blocked = {str(canonicalize_name(name)) for name in blocked_names}
        self.tags = tags
        self.environment: dict[str, str] = {}
        for name, value in environment.items():
            if not isinstance(value, str):
                raise ValueError("package marker environment must contain strings")
            self.environment[name] = value
        self.deadline = time.monotonic() + timeout
        self.cancelled = cancelled
        self.byte_limit = byte_limit
        self.downloaded_bytes = 0
        self.index: dict[str, dict[str, Any]] = {}
        self.wheels: dict[str, bytes] = {}
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )

    def check_active(self) -> None:
        if self.cancelled is not None and self.cancelled.is_set():
            raise InterruptedError("package request interrupted")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("dependency preparation exceeded its deadline")

    def _fetch(self, url: str, limit: int) -> bytes:
        self.check_active()
        parsed = urlsplit(url)
        allowed = (
            parsed.hostname == "pypi.org"
            and re.fullmatch(r"/pypi/[a-z0-9-]+/json", parsed.path)
        ) or (
            parsed.hostname == "files.pythonhosted.org"
            and parsed.path.startswith("/packages/")
            and parsed.path.endswith(".whl")
        )
        if (
            not allowed
            or parsed.scheme != "https"
            or parsed.port is not None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("only fixed PyPI metadata and wheel URLs are permitted")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("dependency preparation exceeded its deadline")
        request = urllib.request.Request(
            url, headers={"User-Agent": "openhands-research"}
        )
        content = bytearray()
        with self.opener.open(request, timeout=min(remaining, 30)) as response:
            while chunk := response.read(64 * 1024):
                content.extend(chunk)
                self.downloaded_bytes += len(chunk)
                if len(content) > limit or self.downloaded_bytes > self.byte_limit:
                    raise ValueError("dependency download exceeds its byte limit")
                self.check_active()
        return bytes(content)

    def identify(self, requirement_or_candidate: Requirement | WheelCandidate) -> str:
        return str(canonicalize_name(requirement_or_candidate.name))

    def get_preference(
        self, identifier, resolutions, candidates, information, backtrack_causes
    ):
        del resolutions, candidates, information, backtrack_causes
        return identifier

    def find_matches(self, identifier, requirements, incompatibilities):
        self.check_active()
        if identifier in self.blocked:
            raise ValueError(f"target package is forbidden: {identifier}")
        constraints = list(requirements[identifier])
        rejected = set(incompatibilities[identifier])
        extras = frozenset(extra for req in constraints for extra in req.extras)
        if identifier not in self.index:
            if len(self.index) >= 64:
                raise ValueError("dependency closure exceeds 64 distributions")
            self.index[identifier] = json.loads(
                self._fetch(
                    f"https://pypi.org/pypi/{identifier}/json", 16 * 1024 * 1024
                )
            )
        matches = []
        for release, files in self.index[identifier]["releases"].items():
            self.check_active()
            try:
                version = Version(release)
            except InvalidVersion:
                continue
            if not all(req.specifier.contains(version) for req in constraints):
                continue
            for artifact in files:
                if artifact.get("packagetype") != "bdist_wheel" or artifact.get(
                    "yanked"
                ):
                    continue
                filename = artifact["filename"]
                if PurePosixPath(filename).name != filename:
                    raise ValueError("wheel filename is not a basename")
                name, wheel_version, _build, tags = parse_wheel_filename(filename)
                if name != identifier or version != wheel_version:
                    raise ValueError("PyPI release and wheel identity disagree")
                if not self.tags.intersection(tags):
                    continue
                python_requirement = artifact.get("requires_python") or ""
                if not SpecifierSet(python_requirement).contains(
                    self.environment["python_full_version"], prereleases=True
                ):
                    continue
                candidate = WheelCandidate(
                    identifier,
                    version,
                    filename,
                    artifact["url"],
                    artifact["digests"]["sha256"],
                    extras,
                )
                if candidate not in rejected:
                    matches.append(candidate)
        return sorted(matches, key=lambda candidate: candidate.version, reverse=True)

    def is_satisfied_by(
        self, requirement: Requirement, candidate: WheelCandidate
    ) -> bool:
        return (
            canonicalize_name(requirement.name) == candidate.name
            and requirement.specifier.contains(candidate.version)
            and requirement.extras <= candidate.extras
        )

    def get_dependencies(self, candidate: WheelCandidate) -> list[Requirement]:
        self.check_active()
        if candidate.filename not in self.wheels:
            content = self._fetch(candidate.url, MAX_WHEEL_BYTES)
            if hashlib.sha256(content).hexdigest() != candidate.sha256:
                raise ValueError("wheel SHA256 does not match PyPI metadata")
            checked_wheel(candidate.filename, content, {candidate.name}, self.blocked)
            self.wheels[candidate.filename] = content
        with zipfile.ZipFile(io.BytesIO(self.wheels[candidate.filename])) as archive:
            metadata_path = next(
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA")
                and len(PurePosixPath(name).parts) == 2
            )
            metadata = BytesParser().parsebytes(archive.read(metadata_path))
        dependencies = []
        for value in metadata.get_all("Requires-Dist", []):
            requirement = checked_requirement(value, self.blocked)
            if requirement.marker is None or any(
                requirement.marker.evaluate({**self.environment, "extra": extra})
                for extra in candidate.extras | {""}
            ):
                dependencies.append(requirement)
        return dependencies


def resolve_wheels(
    requirements: list[str], provider: WheelProvider
) -> tuple[dict[str, bytes], dict[str, Any]]:
    roots = [checked_requirement(value, provider.blocked) for value in requirements]
    roots = [
        req
        for req in roots
        if req.marker is None
        or req.marker.evaluate({**provider.environment, "extra": ""})
    ]
    result = Resolver(provider, BaseReporter()).resolve(roots, max_rounds=200)
    wheels = {}
    manifest = {
        "schema_version": 1,
        "requirements": requirements,
        "blocked_packages": sorted(provider.blocked),
        "environment": provider.environment,
        "wheels": [],
    }
    for name, candidate in sorted(result.mapping.items()):
        wheels[candidate.filename] = provider.wheels[candidate.filename]
        manifest["wheels"].append(
            {
                "name": name,
                "version": str(candidate.version),
                "filename": candidate.filename,
                "sha256": candidate.sha256,
                "url": candidate.url,
            }
        )
    return wheels, manifest


def _image_metadata(root="/workspace"):
    import importlib.metadata
    import json
    from pathlib import Path

    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.tags import sys_tags

    root = Path(root)
    files = {}
    for filename in ("pyproject.toml", "setup.cfg", "setup.py"):
        path = root / filename
        if path.is_file():
            if path.is_symlink() or path.stat().st_size > 128 * 1024:
                raise ValueError("packaging metadata must be a bounded regular file")
            files[filename] = path.read_text(encoding="utf-8")
    installed = []
    for distribution in importlib.metadata.distributions():
        installed.append(
            {
                "name": distribution.metadata.get("Name", ""),
                "version": distribution.version,
                "requires": distribution.requires or [],
                "direct_url": distribution.read_text("direct_url.json"),
            }
        )
    print(
        json.dumps(
            {
                "files": files,
                "installed": installed,
                "environment": default_environment(),
                "tags": [str(tag) for tag in sys_tags()],
            }
        )
    )


def declared_dependencies(
    metadata: dict[str, Any], task_id: str, extra_blocked: tuple[str, ...] = ()
) -> tuple[set[str], list[str], list[str]]:
    """Read packaging declarations as data; setup.py is never executed."""
    blocked = {task_id.rsplit("/", 1)[-1], *extra_blocked}
    builds = ["setuptools>=40.8", "wheel"]
    requirements: list[str] = []
    files = metadata["files"]

    def add_name(value):
        if isinstance(value, str):
            blocked.add(str(canonicalize_name(value, validate=True)))

    def add_requirements(value):
        if isinstance(value, (tuple, list)) and all(
            isinstance(item, str) for item in value
        ):
            requirements.extend(value)

    if "pyproject.toml" in files:
        project = tomllib.loads(files["pyproject.toml"])
        if "build-system" in project:
            builds = list(project["build-system"]["requires"])
        package = project.get("project", {})
        add_name(package.get("name"))
        add_name(project.get("tool", {}).get("poetry", {}).get("name"))
        add_requirements(package.get("dependencies"))
    if "setup.cfg" in files:
        parser = configparser.ConfigParser(interpolation=None)
        parser.read_string(files["setup.cfg"])
        add_name(parser.get("metadata", "name", fallback=None))
        for key in ("install_requires", "setup_requires", "tests_require"):
            value = parser.get("options", key, fallback="")
            requirements.extend(
                line.strip() for line in value.splitlines() if line.strip()
            )
    if "setup.py" in files:
        try:
            tree = ast.parse(files["setup.py"])
        except SyntaxError:
            tree = ast.Module(body=[], type_ignores=[])
        bindings = {}

        def literal(node):
            if isinstance(node, ast.Name):
                return bindings.get(node.id)
            try:
                return ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError):
                return None

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        bindings[target.id] = literal(node.value)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not (
                isinstance(function, ast.Name)
                and function.id == "setup"
                or isinstance(function, ast.Attribute)
                and function.attr == "setup"
            ):
                continue
            for keyword in node.keywords:
                value = literal(keyword.value)
                if keyword.arg == "name":
                    add_name(value)
                elif keyword.arg in {"install_requires", "tests_require"}:
                    add_requirements(value)
                elif keyword.arg == "setup_requires" and isinstance(
                    value, (list, tuple)
                ):
                    builds.extend(value)
    for distribution in metadata["installed"]:
        if distribution.get("direct_url"):
            direct = json.loads(distribution["direct_url"])
            location = urlsplit(direct.get("url", ""))
            if location.scheme == "file" and (
                location.path == "/workspace" or location.path.startswith("/workspace/")
            ):
                add_name(distribution["name"])
    blocked = {str(canonicalize_name(name, validate=True)) for name in blocked}
    for distribution in metadata["installed"]:
        if canonicalize_name(distribution["name"]) in blocked:
            add_requirements(distribution["requires"])
    for requirement in builds + requirements:
        checked_requirement(requirement, blocked)
    return blocked, builds, requirements


def _sanitize_package_sources(root="/", remove_git=True):
    import os
    import shutil
    from pathlib import Path

    base = Path(root)

    def remove(path):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    homes = [base / "root"]
    if (base / "home").is_dir():
        homes.extend(
            path
            for path in (base / "home").iterdir()
            if not path.is_symlink() and os.access(path, os.R_OK | os.X_OK)
        )
    for home in homes:
        for relative in (
            ".cache",
            ".pip",
            ".config/pip",
            ".gitconfig",
            ".git-credentials",
        ):
            remove(home / relative)
    for relative in ("tmp", "var/tmp"):
        directory = base / relative
        if directory.is_dir() and not directory.is_symlink():
            for path in directory.iterdir():
                remove(path)
    remove(base / "opt/openhands-research-wheels")
    if remove_git:
        for directory, subdirs, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            if parent == base:
                subdirs[:] = [
                    name for name in subdirs if name not in {"proc", "sys", "dev"}
                ]
            if ".git" in subdirs or ".git" in files:
                if parent == base:
                    raise ValueError(
                        "root-level Git checkout requires a clean task image"
                    )
                if not parent.is_relative_to(base / "workspace"):
                    remove(parent)
                    subdirs.clear()
                    continue
            for name in (".git", ".gitconfig", ".git-credentials"):
                if name in subdirs or name in files:
                    remove(parent / name)
                    if name in subdirs:
                        subdirs.remove(name)
            if "HEAD" in files and {"objects", "refs"} <= set(subdirs):
                remove(parent)
                subdirs.clear()


@dataclass
class DependencyWheelhouse:
    wheels: dict[str, bytes]
    manifest: dict[str, Any]

    def merge(self, *others: DependencyWheelhouse) -> DependencyWheelhouse:
        """Combine independently verified wheel sets without replacing artifacts."""
        components = [self, *others]
        wheels: dict[str, bytes] = {}
        artifacts: dict[str, dict[str, Any]] = {}
        requirements: list[str] = []
        blocked: set[str] = set()
        sources: list[str] = []
        for component in components:
            source = component.manifest.get("source")
            if isinstance(source, str) and source and source not in sources:
                sources.append(source)
            blocked.update(component.manifest.get("blocked_packages", []))
            for requirement in component.manifest.get("requirements", []):
                if requirement not in requirements:
                    requirements.append(requirement)
            for filename, content in component.wheels.items():
                existing = wheels.get(filename)
                if existing is not None and existing != content:
                    raise ValueError(f"conflicting wheel artifact: {filename}")
                wheels[filename] = content
            for artifact in component.manifest.get("wheels", []):
                filename = artifact.get("filename")
                if not isinstance(filename, str):
                    raise ValueError("wheel manifest entry requires a filename")
                existing = artifacts.get(filename)
                if existing is not None and existing != artifact:
                    raise ValueError(f"conflicting wheel manifest: {filename}")
                artifacts[filename] = copy.deepcopy(artifact)
        return DependencyWheelhouse(
            wheels,
            {
                "schema_version": 1,
                "source": "+".join(sources) or "empty",
                "requirements": requirements,
                "blocked_packages": sorted(blocked),
                "wheels": [artifacts[name] for name in sorted(artifacts)],
                "components": [copy.deepcopy(item.manifest) for item in components],
            },
        )

    def stage(self, target: TaskTarget) -> None:
        source = (
            inspect.getsource(_sanitize_package_sources)
            + "\n_sanitize_package_sources()\n"
        )
        code, _, stderr = target.exec(
            ["python3", "-I", "-S", "-c", source],
            timeout=120,
            workdir="/",
            user="root",
            privileged=True,
        )
        if code:
            raise RuntimeError(f"package source cleanup failed: {stderr[-1000:]}")
        self.publish(target)

    def publish(self, target: TaskTarget) -> None:
        """Add verified wheels without cleaning an active solver's files or caches."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            directory = tarfile.TarInfo(WHEELHOUSE_PATH.lstrip("/"))
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o755
            archive.addfile(directory)
            for filename, content in self.wheels.items():
                member = tarfile.TarInfo(directory.name + "/" + filename)
                member.size = len(content)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(content))
        target.restore(buffer.getvalue(), path="/")
        target.write_file(
            "/etc/pip.conf",
            "[global]\nno-index = true\n"
            f"find-links = {WHEELHOUSE_PATH}\n"
            "disable-pip-version-check = true\n",
        )


def _package_environment():
    import json

    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.tags import sys_tags

    print(
        json.dumps(
            {
                "environment": default_environment(),
                "tags": [str(tag) for tag in sys_tags()],
            }
        )
    )


class PackageBroker:
    """Model-initiated wheel access using only the solver's runtime and denylist."""

    def __init__(
        self, target: TaskTarget, blocked_names: set[str], journal: Path
    ) -> None:
        if target._labels.get("org.repogeneration.role") != "work":
            raise ValueError("package broker requires a solver container")
        self.target = target
        if any(
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]{1,512}", name) is None
            for name in blocked_names
        ):
            raise ValueError("invalid forbidden package/import namespace")
        self.blocked = {str(canonicalize_name(name)) for name in blocked_names}
        self.journal = journal
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self.downloaded_bytes = 0
        self.request_count = 0
        self._wheels: dict[str, bytes] = {}
        self._artifacts: dict[str, dict[str, Any]] = {}
        source = inspect.getsource(_package_environment) + "\n_package_environment()\n"
        code, stdout, _stderr = target.exec(
            ["python3", "-I", "-c", source], timeout=30, workdir="/", user="root"
        )
        if code:
            raise RuntimeError("solver Python/pip compatibility inspection failed")
        runtime = json.loads(stdout)
        self.tags = {tag for value in runtime["tags"] for tag in parse_tag(value)}
        self.environment = runtime["environment"]

    def interrupt(self) -> None:
        self.cancelled.set()

    def snapshot(self) -> DependencyWheelhouse:
        """Return controller-verified bytes, never files read back from a solver."""
        with self._lock:
            return DependencyWheelhouse(
                dict(self._wheels),
                {
                    "schema_version": 1,
                    "source": "model-requested",
                    "blocked_packages": sorted(self.blocked),
                    "wheels": copy.deepcopy(list(self._artifacts.values())),
                },
            )

    def request(self, requirements: list[str]) -> dict[str, Any]:
        with self._lock:
            request_id = uuid.uuid4().hex
            entry = {
                "request_id": request_id,
                "source": "solver",
                "requirements": requirements,
            }
            append_jsonl(self.journal, {**entry, "utc": utc_now(), "status": "started"})
            provider = None
            try:
                if (
                    not isinstance(requirements, list)
                    or not 1 <= len(requirements) <= 32
                ):
                    raise ValueError(
                        "request between 1 and 32 named Python requirements"
                    )
                for value in requirements:
                    if not isinstance(value, str) or len(value) > 1024:
                        raise ValueError(
                            "requirements must be strings of at most 1024 characters"
                        )
                    checked_requirement(value, self.blocked)
                if (
                    self.request_count >= 64
                    or self.downloaded_bytes >= MAX_DOWNLOAD_BYTES
                ):
                    raise ValueError("solver package-request allowance exhausted")
                self.request_count += 1
                provider = WheelProvider(
                    self.blocked,
                    self.tags,
                    self.environment,
                    cancelled=self.cancelled,
                    byte_limit=MAX_DOWNLOAD_BYTES - self.downloaded_bytes,
                )
                provider.check_active()
                wheels, manifest = resolve_wheels(requirements, provider)
                provider.check_active()
                DependencyWheelhouse(wheels, manifest).publish(self.target)
                self._wheels.update(wheels)
                self._artifacts.update(
                    {row["filename"]: row for row in manifest["wheels"]}
                )
                append_jsonl(
                    self.journal,
                    {
                        **entry,
                        "utc": utc_now(),
                        "status": "staged",
                        **manifest,
                    },
                )
                return manifest
            except Exception as exc:
                append_jsonl(
                    self.journal,
                    {
                        **entry,
                        "utc": utc_now(),
                        "status": "failed",
                        "exception_type": type(exc).__name__,
                    },
                )
                raise
            finally:
                if provider is not None:
                    self.downloaded_bytes += provider.downloaded_bytes


def prepare_dependencies(
    target: TaskTarget,
    task_id: str,
    extra_requirements: tuple[str, ...] = (),
    blocked_packages: tuple[str, ...] = (),
) -> DependencyWheelhouse:
    if target._labels.get("org.repogeneration.role") != "grading":
        raise ValueError("reference dependency preparation is restricted to grading")
    source = inspect.getsource(_image_metadata) + "\n_image_metadata()\n"
    code, stdout, stderr = target.exec(
        ["python3", "-I", "-c", source], timeout=60, workdir="/", user="root"
    )
    if code:
        raise RuntimeError(f"package metadata inspection failed: {stderr[-1000:]}")
    metadata = json.loads(stdout)
    blocked, builds, runtime = declared_dependencies(
        metadata, task_id, blocked_packages
    )
    requirements = list(builds) + list(extra_requirements)
    installed = {
        str(canonicalize_name(item["name"])): item["version"]
        for item in metadata["installed"]
        if canonicalize_name(item["name"]) not in blocked
    }
    for value in runtime:
        requirement = checked_requirement(value, blocked)
        version = installed.get(str(canonicalize_name(requirement.name)))
        if version is None or not requirement.specifier.contains(version):
            requirements.append(value)
    wheelhouse = _resolve_declared_wheels(
        metadata,
        blocked,
        requirements,
        source="grader-declared",
    )
    wheelhouse.manifest["declared_dependencies"] = runtime
    return wheelhouse


def _submission_packaging_files(archive: bytes) -> dict[str, str]:
    if not isinstance(archive, bytes):
        raise ValueError("submission must be tar bytes")
    files: dict[str, str] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            for member in bundle:
                if member.name not in {"pyproject.toml", "setup.cfg", "setup.py"}:
                    continue
                if (
                    not member.isfile()
                    or member.size < 0
                    or member.size > 128 * 1024
                    or member.name in files
                ):
                    raise ValueError("invalid submission packaging metadata")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("submission packaging metadata is missing")
                with stream:
                    content = stream.read(member.size + 1)
                if len(content) != member.size:
                    raise ValueError("submission packaging metadata is truncated")
                files[member.name] = content.decode("utf-8")
    except (tarfile.TarError, OSError, EOFError, UnicodeError) as exc:
        raise ValueError("invalid normalized submission tar") from exc
    return files


def _resolve_declared_wheels(
    metadata: dict[str, Any],
    blocked: set[str],
    requirements: list[str],
    *,
    source: str,
    cache_dir: Path | None = None,
) -> DependencyWheelhouse:
    installed = {
        str(canonicalize_name(item["name"])): item["version"]
        for item in metadata["installed"]
        if canonicalize_name(item["name"]) not in blocked
    }
    parsed = [checked_requirement(value, blocked) for value in requirements]
    for index, requirement in enumerate(parsed):
        version = installed.get(str(canonicalize_name(requirement.name)))
        if version is not None and all(
            other.specifier.contains(version)
            for other in parsed
            if canonicalize_name(other.name) == canonicalize_name(requirement.name)
        ):
            requirement.specifier &= SpecifierSet("==" + version)
            requirements[index] = str(requirement)
    tags = {tag for value in metadata["tags"] for tag in parse_tag(value)}
    if cache_dir is None:
        provider = WheelProvider(blocked, tags, metadata["environment"])
    else:
        from .prepared_dependencies import CachedWheelProvider

        provider = CachedWheelProvider(
            blocked,
            tags,
            metadata["environment"],
            cache_dir=cache_dir,
        )
    wheels, manifest = resolve_wheels(requirements, provider)
    manifest["source"] = source
    manifest["already_installed"] = installed
    return DependencyWheelhouse(wheels, manifest)


def prepare_build_dependencies(
    target: TaskTarget,
    task_id: str,
    *,
    submission: bytes,
    blocked_packages: tuple[str, ...] = (),
    cache_dir: Path | None = None,
) -> DependencyWheelhouse:
    """Resolve only the submitted PEP 517 build requirements for offline grading."""
    if target._labels.get("org.repogeneration.role") != "grading":
        raise ValueError("submission build preparation is restricted to grading")
    source = inspect.getsource(_image_metadata) + "\n_image_metadata()\n"
    code, stdout, stderr = target.exec(
        ["python3", "-I", "-c", source], timeout=60, workdir="/", user="root"
    )
    if code:
        raise RuntimeError(f"package metadata inspection failed: {stderr[-1000:]}")
    metadata = json.loads(stdout)
    metadata["files"] = _submission_packaging_files(submission)
    blocked, builds, _runtime = declared_dependencies(
        metadata, task_id, blocked_packages
    )
    if len(builds) > 32:
        raise ValueError("build-system requirements exceed 32 packages")
    wheelhouse = _resolve_declared_wheels(
        metadata,
        blocked,
        builds,
        source="submission-build-system",
        cache_dir=cache_dir,
    )
    wheelhouse.manifest["declared_build_dependencies"] = builds
    return wheelhouse
