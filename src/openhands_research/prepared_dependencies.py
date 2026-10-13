"""Cached dependency policy isolated from live legacy helper source."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

from resolvelib import BaseReporter, Resolver

from .dependencies import (
    DependencyWheelhouse,
    PackageBroker,
    WheelProvider,
    checked_requirement,
    checked_wheel,
)
from .ledger import append_jsonl, utc_now


PREPARATION_BYTES = 2 * 1024 * 1024 * 1024
PREPARATION_WHEEL_BYTES = 512 * 1024 * 1024


class CachedWheelProvider(WheelProvider):
    def __init__(self, *args, cache_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_dir = cache_dir
        self.cached_bytes = 0

    def find_matches(self, identifier, requirements, incompatibilities):
        self.check_active()
        if identifier in self.blocked:
            raise ValueError(f"target package is forbidden: {identifier}")
        if identifier not in self.index:
            if len(self.index) >= 1024:
                raise ValueError("dependency closure exceeds 1024 distributions")
            self.index[identifier] = json.loads(
                self._fetch(
                    f"https://pypi.org/pypi/{identifier}/json", 16 * 1024 * 1024
                )
            )
        return super().find_matches(identifier, requirements, incompatibilities)

    def get_dependencies(self, candidate):
        self.check_active()
        if candidate.filename not in self.wheels:
            if re.fullmatch(r"[0-9a-f]{64}", candidate.sha256) is None:
                raise ValueError("invalid wheel checksum")
            cached = self.cache_dir / candidate.sha256
            if cached.exists():
                if (
                    cached.is_symlink()
                    or not cached.is_file()
                    or cached.stat().st_size > PREPARATION_WHEEL_BYTES
                ):
                    raise ValueError("invalid wheel cache entry")
                content = cached.read_bytes()
                self.cached_bytes += len(content)
            else:
                content = self._fetch(candidate.url, PREPARATION_WHEEL_BYTES)
            if self.cached_bytes + self.downloaded_bytes > self.byte_limit:
                raise ValueError("dependency artifacts exceed their byte limit")
            if hashlib.sha256(content).hexdigest() != candidate.sha256:
                raise ValueError("wheel SHA256 does not match PyPI metadata")
            checked_wheel(candidate.filename, content, {candidate.name}, self.blocked)
            if not cached.exists():
                cached.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                temporary = cached.with_name(
                    f".{candidate.sha256}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("xb") as output:
                        output.write(content)
                    os.replace(temporary, cached)
                finally:
                    temporary.unlink(missing_ok=True)
            self.wheels[candidate.filename] = content
        return super().get_dependencies(candidate)


class CachedPackageBroker(PackageBroker):
    def __init__(self, *args, cache_dir: Path, source="solver", root_limit=32):
        super().__init__(*args)
        self.cache_dir = cache_dir
        self.source = source
        self.root_limit = root_limit

    def request(self, requirements):
        with self._lock:
            entry = {
                "request_id": uuid.uuid4().hex,
                "source": self.source,
                "requirements": requirements,
            }
            append_jsonl(self.journal, {**entry, "utc": utc_now(), "status": "started"})
            provider = None
            try:
                if (
                    not isinstance(requirements, list)
                    or not 1 <= len(requirements) <= self.root_limit
                ):
                    raise ValueError("invalid dependency requirement count")
                roots = []
                for value in requirements:
                    if not isinstance(value, str) or len(value) > 1024:
                        raise ValueError("invalid dependency requirement")
                    requirement = checked_requirement(value, self.blocked)
                    if requirement.marker is None or requirement.marker.evaluate(
                        {**self.environment, "extra": ""}
                    ):
                        roots.append(requirement)
                if (
                    self.request_count >= 64
                    or self.downloaded_bytes >= PREPARATION_BYTES
                ):
                    raise ValueError("solver package-request allowance exhausted")
                self.request_count += 1
                provider = CachedWheelProvider(
                    self.blocked,
                    self.tags,
                    self.environment,
                    cancelled=self.cancelled,
                    byte_limit=PREPARATION_BYTES - self.downloaded_bytes,
                    cache_dir=self.cache_dir,
                )
                resolved = Resolver(provider, BaseReporter()).resolve(
                    roots, max_rounds=4096
                )
                wheels, records = {}, []
                for name, candidate in sorted(resolved.mapping.items()):
                    wheels[candidate.filename] = provider.wheels[candidate.filename]
                    records.append(
                        {
                            "name": name,
                            "version": str(candidate.version),
                            "filename": candidate.filename,
                            "sha256": candidate.sha256,
                            "url": candidate.url,
                        }
                    )
                manifest = {
                    "schema_version": 1,
                    "requirements": requirements,
                    "blocked_packages": sorted(self.blocked),
                    "environment": self.environment,
                    "wheels": records,
                }
                provider.check_active()
                DependencyWheelhouse(wheels, manifest).publish(self.target)
                self._wheels.update(wheels)
                self._artifacts.update({row["filename"]: row for row in records})
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
                    self.downloaded_bytes += (
                        provider.downloaded_bytes + provider.cached_bytes
                    )
