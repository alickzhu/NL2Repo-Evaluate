"""Controller-owned Docker containers; solver actions only use the task RPC.

Images must already be present and contain Bash and Python >= 3.9. No image
pulls, host mounts, published ports, proxy environment forwarding or model
credentials are introduced here. Mutating execution is never retried after a
timeout/disconnect. Only pre-create slot waits and read-only daemon handshakes
may be retried.
"""

from __future__ import annotations

import base64
import copy
import fcntl
import io
import ipaddress
import json
import logging
import math
import os
import posixpath
import select
import socket
import struct
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, cast

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from requests.exceptions import RequestException, Timeout as RequestTimeout


MAX_REQUEST_BYTES = 64 * 1024
MAX_EXEC_OUTPUT_BYTES = 1024 * 1024
MAX_RPC_OUTPUT_BYTES = 16 * 1024 * 1024
_IDLE_COMMAND = (
    "exec python3 -I -S -c 'import signal, sys; "
    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); signal.pause()'"
)
_IO_ERRORS = (DockerException, RequestException, OSError)
_CLEANUP_ATTEMPTS = 2
_STARTUP_WAIT_ATTEMPTS = 3
_STARTUP_WAIT_SECONDS = 1800
_DAEMON_CONNECT_ATTEMPTS = 8
_STARTUP_RETRY_DELAY = 16
_LOGGER = logging.getLogger(__name__)


@contextmanager
def startup_slot():
    """Bound simultaneous daemon creates across controller worker processes."""
    root = Path(tempfile.gettempdir()) / "repogeneration-startup-slots"
    root.mkdir(mode=0o700, exist_ok=True)
    wait_attempt = 1
    deadline = time.monotonic() + _STARTUP_WAIT_SECONDS
    while True:
        for index in range(2):
            stream = (root / str(index)).open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            try:
                yield
            finally:
                stream.close()
            return
        if time.monotonic() >= deadline:
            if wait_attempt >= _STARTUP_WAIT_ATTEMPTS:
                raise TaskRuntimeError("Docker startup slot deadline exceeded")
            wait_attempt += 1
            _LOGGER.warning("Container startup slot still busy; retrying wait %s/%s",
                            wait_attempt, _STARTUP_WAIT_ATTEMPTS)
            time.sleep(_STARTUP_RETRY_DELAY)
            deadline = time.monotonic() + _STARTUP_WAIT_SECONDS
        time.sleep(0.1)


def _connect_startup_client():
    """Limit and retry read-only version negotiation before any container exists."""
    with startup_slot():
        for attempt in range(1, _DAEMON_CONNECT_ATTEMPTS + 1):
            try:
                return docker.from_env(timeout=60)
            except (DockerException, RequestException) as exc:
                # DockerException also includes non-transient configuration
                # failures. Only the known version-read wrapper is eligible.
                if isinstance(exc, DockerException) and not str(exc).startswith(
                    "Error while fetching server API version:"
                ):
                    raise
                if attempt == _DAEMON_CONNECT_ATTEMPTS:
                    raise
                _LOGGER.warning("Read-only container daemon handshake failed; retry %s/%s",
                                attempt + 1, _DAEMON_CONNECT_ATTEMPTS)
                time.sleep(_STARTUP_RETRY_DELAY)


class TaskRuntimeError(RuntimeError):
    """Safe controller error, without raw commands, daemon URLs or credentials."""


class OwnershipError(TaskRuntimeError):
    """The immutable task container identity could not be verified."""


class TaskExecutionTimeout(TaskRuntimeError, TimeoutError):
    """Execution may still be running inside the task; it was not retried."""


class TaskCleanupError(TaskRuntimeError):
    """A bounded, ownership-checked cleanup left a task resource behind."""

    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = copy.deepcopy(diagnostics)


def _attach_cleanup_diagnostics(
    error: BaseException, diagnostics: list[dict[str, Any]]
) -> None:
    """Keep the primary exception while making secondary cleanup failures inspectable."""
    if not diagnostics:
        return
    existing = list(getattr(error, "cleanup_diagnostics", ()))
    try:
        error.cleanup_diagnostics = existing + copy.deepcopy(diagnostics)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass


def normalize_task_path(path: str, workdir: str = "/workspace") -> str:
    """Lexical POSIX normalization only; never inspect the controller filesystem."""
    if not isinstance(path, str) or not path or "\0" in path:
        raise ValueError("A nonempty task path without NUL is required")
    base = PurePosixPath(workdir)
    if not base.is_absolute():
        raise ValueError("workdir must be an absolute task path")
    normalized = posixpath.normpath(str(base / PurePosixPath(path)))
    return "/" + normalized.lstrip("/")


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TaskExecutionTimeout(
            "Docker execution timed out; task-side effects may have occurred. "
            "The command was not retried or automatically restarted."
        )
    return remaining


def _tar_file(name: str, content: bytes, directory: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        if directory is not None:
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            info.mode = 0o700
            archive.addfile(info)
        info = tarfile.TarInfo(name)
        info.size = len(content)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


class TaskContainer:
    """One immutable ownership generation, with explicit capture-before-close.

    ``exec`` is for trusted setup, not serialized solver actions. Its hard
    timeout closes the transport, not an arbitrary process in the container;
    the outcome is then unknown until inspected or the container is closed.
    The terminal tool instead has a resumable soft timeout and explicit reset.
    ``capture``/``restore`` only move tar bytes, never extract them on the host.
    """

    def __init__(
        self,
        image: str,
        run_id: str,
        task_id: str,
        role: str = "work",
        workdir: str = "/workspace",
        network_disabled: bool = True,
        mem_limit: str = "4g",
        nano_cpus: int = 2_000_000_000,
        client: docker.DockerClient | None = None,
    ) -> None:
        if not PurePosixPath(workdir).is_absolute():
            raise ValueError("workdir must be absolute inside the task")
        if not all(
            isinstance(value, str) and value for value in (image, run_id, task_id, role)
        ):
            raise ValueError("image, run_id, task_id and role must be nonempty strings")
        if client is None and os.environ.get("DOCKER_HOST", "").startswith("ssh:"):
            raise ValueError("SSH Docker transports are not supported")
        self._client = client if client is not None else _connect_startup_client()
        if self._client.api.base_url.startswith("http+docker://ssh"):
            raise ValueError("SSH Docker transports are not supported")
        self._owns_client = client is None
        self._image = image
        self._workdir = normalize_task_path(workdir)
        self._owner_id = uuid.uuid4().hex
        self._labels = MappingProxyType(
            {
                "org.repogeneration.run": run_id,
                "org.repogeneration.task": task_id,
                "org.repogeneration.role": role,
                "org.repogeneration.owner": self._owner_id,
            }
        )
        self._network_disabled = network_disabled
        self._dns = [str(ipaddress.ip_address(address.strip()))
                     for address in os.environ.get("NL2REPO_CONTAINER_DNS", "").split(",")
                     if address.strip()] if not network_disabled else []
        self._mem_limit = mem_limit
        self._nano_cpus = nano_cpus
        self._container: Container | None = None
        self._container_id: str | None = None
        self._entered = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._helper_lock = threading.Lock()
        self._helper_state = "new"
        self._helper_dir = f"/opt/rgoh-{self._owner_id}"
        self._helper_path = f"{self._helper_dir}/task_runtime.py"
        self._socket_path = f"{self._helper_dir}/rpc.sock"
        self._cleanup_diagnostics: list[dict[str, Any]] = []

    @property
    def client(self) -> docker.DockerClient:
        return self._client

    @property
    def container(self) -> Container | None:
        return self._container

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def workdir(self) -> str:
        return self._workdir

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def generation(self) -> str:
        return self._owner_id

    @property
    def cleanup_diagnostics(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._cleanup_diagnostics)

    def _api(self, timeout: float = 15.0):
        # Per-call timeout without changing the shared caller's APIClient. The
        # connection adapters are shared; only the response/socket is closed.
        api = copy.copy(self.client.api)
        cast(Any, api).timeout = max(0.001, timeout)
        return api

    def _owned(self, timeout: float = 15.0) -> Container:
        if self._closed or self._container_id is None or self._container is None:
            raise TaskRuntimeError("Task container is not open")
        if self._container.id != self._container_id:
            raise OwnershipError("Container object identity changed; operation refused")
        attrs = self._api(timeout).inspect_container(self._container_id)
        labels = attrs.get("Config", {}).get("Labels") or {}
        if attrs.get("Id") != self._container_id or any(
            labels.get(key) != value for key, value in self._labels.items()
        ):
            raise OwnershipError("Task container ownership mismatch; operation refused")
        return self.client.containers.prepare_model(attrs)

    def __enter__(self) -> TaskContainer:
        with startup_slot():
            # Each target owns its client. Restore the ordinary RPC timeout;
            # this does not change tool-execution or model request budgets.
            old_timeout = self.client.api.timeout
            self.client.api.timeout = 300
            try:
                return self._start()
            finally:
                self.client.api.timeout = old_timeout

    def _start(self) -> TaskContainer:
        if self._entered or self._closed:
            raise TaskRuntimeError(
                "A TaskContainer generation can only be entered once"
            )
        self._entered = True
        try:
            image = self.client.images.get(self._image)
            if not isinstance(image.id, str):
                raise TaskRuntimeError("Docker image has no immutable ID")
            if image.attrs.get("Config", {}).get("Volumes"):
                raise TaskRuntimeError("Images declaring volumes are not supported")
            container = self.client.containers.create(
                image=image.id,
                name=f"rgoh-{self.owner_id}",
                entrypoint=["/bin/bash", "--noprofile", "--norc", "-c"],
                command=[_IDLE_COMMAND],
                working_dir=self.workdir,
                user="root",
                labels=dict(self._labels),
                network_mode="none" if self._network_disabled else "bridge",
                **({"dns": self._dns} if self._dns else {}),
                mem_limit=self._mem_limit,
                nano_cpus=self._nano_cpus,
                pids_limit=256,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges:true"],
                privileged=False,
                read_only=False,
                auto_remove=False,
                init=True,
                ports=None,
                publish_all_ports=False,
                volumes=None,
                mounts=[],
                use_config_proxy=False,
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
            )
            self._container = container
            self._container_id = container.id
            if self._container_id is None:
                raise TaskRuntimeError("Docker created no identifiable container")
            owned = self._owned(timeout=300)
            if owned.attrs.get("Mounts"):
                raise TaskRuntimeError("Unexpected container mounts; refusing to start")
            self._api(300).start(self._container_id)
            if not self._owned(timeout=300).attrs.get("State", {}).get("Running"):
                raise TaskRuntimeError("Task image did not stay running")
            return self
        except BaseException:
            # A timed-out POST may have created the container. Never repeat it.
            # Resolve the unique name, verify owner labels, then clean by ID.
            if self._container_id is None:
                try:
                    attrs = self._api(60).inspect_container(f"rgoh-{self.owner_id}")
                    labels = attrs.get("Config", {}).get("Labels", {})
                    if all(labels.get(k) == v for k, v in self._labels.items()):
                        self._container_id = attrs["Id"]
                        self._container = self.client.containers.prepare_model(attrs)
                except _IO_ERRORS:
                    pass
            self.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except TaskRuntimeError:
            if exc_value is None:
                raise
            _attach_cleanup_diagnostics(exc_value, self.cleanup_diagnostics)

    @staticmethod
    def _read_exec_output(stream, deadline: float, limit: int) -> tuple[bytes, bytes]:
        outputs = [bytearray(), bytearray()]
        truncated = [False, False]
        wire = bytearray()
        channel, frame_left = 0, 0
        while True:
            remaining = _remaining(deadline)
            if not select.select([stream], [], [], remaining)[0]:
                _remaining(deadline)
                continue
            if isinstance(stream, socket.socket):
                stream.settimeout(remaining)
                chunk = stream.recv(65536)
            elif isinstance(stream, socket.SocketIO):
                chunk = stream.read(65536)
            else:
                raise TaskRuntimeError("Unsupported Docker exec socket transport")
            if not chunk:
                if wire or frame_left:
                    raise TaskRuntimeError(
                        "Docker exec disconnected mid-frame; not retried"
                    )
                break
            wire.extend(chunk)
            while wire:
                if not frame_left:
                    if len(wire) < 8:
                        break
                    channel, frame_left = struct.unpack("!BxxxI", wire[:8])
                    del wire[:8]
                    if channel not in (1, 2):
                        raise TaskRuntimeError("Invalid Docker exec output frame")
                    if not frame_left:
                        continue
                size = min(frame_left, len(wire))
                output = outputs[channel - 1]
                available = max(0, limit - len(output))
                output.extend(wire[: min(size, available)])
                truncated[channel - 1] |= size > available
                del wire[:size]
                frame_left -= size
        for index, was_truncated in enumerate(truncated):
            if was_truncated:
                outputs[index].extend(b"\n<response clipped>\n")
        return bytes(outputs[0]), bytes(outputs[1])

    def _exec_bytes(
        self,
        command: list[str],
        timeout: float,
        workdir: str | None = None,
        user: str | None = None,
        privileged: bool = False,
        output_limit: int = MAX_EXEC_OUTPUT_BYTES,
    ) -> tuple[int, bytes, bytes]:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("exec timeout must be finite and positive")
        if type(privileged) is not bool:
            raise ValueError("privileged must be a boolean")
        deadline = time.monotonic() + timeout
        stream = None
        try:
            self._owned(_remaining(deadline))
            created = self._api(_remaining(deadline)).exec_create(
                self._container_id,
                cmd=command,
                stdout=True,
                stderr=True,
                stdin=False,
                tty=False,
                privileged=privileged,
                user=user or "root",
                workdir=normalize_task_path(workdir or self.workdir, self.workdir),
            )
            exec_id = created["Id"]
            stream = self._api(_remaining(deadline)).exec_start(exec_id, socket=True)
            stdout, stderr = self._read_exec_output(stream, deadline, output_limit)
            while True:
                state = self._api(_remaining(deadline)).exec_inspect(exec_id)
                if not state["Running"] and state["ExitCode"] is not None:
                    return int(state["ExitCode"]), stdout, stderr
                threading.Event().wait(min(0.01, _remaining(deadline)))
        except (TimeoutError, RequestTimeout):
            raise TaskExecutionTimeout(
                "Docker execution timed out; outcome may be unknown. "
                "The command was not retried."
            ) from None
        except _IO_ERRORS:
            raise TaskRuntimeError(
                "Docker execution transport failed; outcome may be unknown. "
                "The command was not retried."
            ) from None
        finally:
            if stream is not None:
                # Docker retains the HTTP response on its raw socket. Closing
                # raw first breaks HTTPResponse.flush on Python 3.13.
                try:
                    stream._response.close()
                finally:
                    stream.close()

    def exec(
        self,
        command: str | list[str],
        timeout: float = 300,
        workdir: str | None = None,
        user: str | None = None,
        privileged: bool = False,
    ) -> tuple[int, str, str]:
        """Run trusted setup once; strings use task Bash, lists use direct argv."""
        if isinstance(command, str):
            argv = ["/bin/bash", "--noprofile", "--norc", "-c", command]
        elif isinstance(command, list) and command:
            argv = list(command)
        else:
            raise ValueError("command must be a shell string or nonempty argv list")
        if any(not isinstance(arg, str) or "\0" in arg for arg in argv):
            raise ValueError("command arguments must be strings without NUL")
        code, stdout, stderr = self._exec_bytes(
            argv,
            timeout,
            workdir,
            user,
            privileged,
        )
        return (
            code,
            stdout.decode("utf-8", "replace"),
            stderr.decode("utf-8", "replace"),
        )

    def _put_archive(self, archive: bytes, path: str) -> None:
        try:
            self._owned()
            assert self._container_id is not None
            if not self._api().put_archive(self._container_id, path, archive):
                raise TaskRuntimeError("Docker did not accept the task archive")
        except _IO_ERRORS:
            raise TaskRuntimeError(
                "Task archive write failed; it was not retried"
            ) from None

    def write_file(self, path: str, content: str | bytes) -> None:
        destination = PurePosixPath(normalize_task_path(path, self.workdir))
        data = content.encode("utf-8") if isinstance(content, str) else content
        if not isinstance(data, bytes):
            raise TypeError("content must be str or bytes")
        code, _, _ = self.exec(["mkdir", "-p", "--", str(destination.parent)])
        if code:
            raise TaskRuntimeError("Could not prepare the task file parent directory")
        temporary = destination.parent / f".rgoh-write-{uuid.uuid4().hex}"
        self._put_archive(_tar_file(temporary.name, data), str(destination.parent))
        code, _, _ = self.exec(
            [
                "python3",
                "-I",
                "-S",
                "-c",
                "import os, sys; os.replace(sys.argv[1], sys.argv[2])",
                str(temporary),
                str(destination),
            ]
        )
        if code:
            raise TaskRuntimeError("Task file replacement failed; it was not retried")

    def _get_archive(self, path: str, limit: int | None = None) -> bytes:
        try:
            self._owned()
            assert self._container_id is not None
            chunks, _ = self._api().get_archive(self._container_id, path)
            result = bytearray()
            try:
                for chunk in chunks:
                    result.extend(chunk)
                    if limit is not None and len(result) > limit:
                        raise TaskRuntimeError(
                            "Task file archive exceeds the 64 MiB read limit"
                        )
            finally:
                chunks.close()
            return bytes(result)
        except _IO_ERRORS:
            raise TaskRuntimeError("Task archive read failed") from None

    def read_file(self, path: str, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
        data = self._get_archive(normalize_task_path(path, self.workdir), max_bytes)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isfile():
                raise TaskRuntimeError("read_file requires a single regular task file")
            file = archive.extractfile(members[0])
            if file is None:
                raise TaskRuntimeError("Task archive has no file contents")
            return file.read()

    def capture(self, path: str = "/workspace", *, directory: bool = False) -> bytes:
        """Return the Docker tar tree without any controller-side extraction."""
        if directory:
            root = PurePosixPath(normalize_task_path(path, self.workdir))
            temporary = f"/tmp/rgoh-submission-{uuid.uuid4().hex}.tar"
            command = ["tar", "--hard-dereference", "-cf", temporary]
            command.extend(
                f"--exclude={value}"
                for value in (
                    ".git",
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".pytest_cache",
                    "*.pyc",
                    "*.pyo",
                    f"{root.name}/start.md",
                )
            )
            command.extend(["-C", str(root.parent), "--", root.name])
            primary_error: BaseException | None = None
            result: bytes | None = None
            try:
                code, _, error = self.exec(command, timeout=120, workdir="/")
                if code:
                    raise TaskRuntimeError(
                        f"implementation capture failed: {error[-1000:]}"
                    )
                result = self.read_file(temporary, max_bytes=256 * 1024 * 1024)
            except BaseException as exc:
                primary_error = exc
            try:
                cleanup_diagnostic = self._remove_capture_temporary(temporary)
            except OwnershipError as cleanup_error:
                cleanup_diagnostic = {
                    "status": "residual_ownership_unverified",
                    "resource": "task_capture_temporary_archive",
                    "path": temporary,
                    "container_id": self._container_id,
                    "owner_id": self._owner_id,
                    "operation": "remove_exact_temporary_archive",
                    "attempts": 1,
                    "error_type": type(cleanup_error).__name__,
                }
                self._cleanup_diagnostics.append(cleanup_diagnostic)
                if primary_error is not None:
                    _attach_cleanup_diagnostics(primary_error, [cleanup_diagnostic])
                    raise primary_error
                raise
            if cleanup_diagnostic is not None:
                self._cleanup_diagnostics.append(cleanup_diagnostic)
            if primary_error is not None:
                _attach_cleanup_diagnostics(
                    primary_error,
                    [cleanup_diagnostic] if cleanup_diagnostic is not None else [],
                )
                raise primary_error
            assert result is not None
            return result
        return self._get_archive(normalize_task_path(path, self.workdir))

    def _remove_capture_temporary(self, path: str) -> dict[str, Any] | None:
        """Remove one task-side archive, retrying only its exact generated path."""
        last_error: Exception | None = None
        for attempt in range(1, _CLEANUP_ATTEMPTS + 1):
            try:
                code, _, _ = self.exec(
                    ["rm", "-f", "--", path], timeout=15, workdir="/"
                )
                if code:
                    raise TaskRuntimeError(
                        "Task capture temporary cleanup command failed"
                    )
                return None
            except OwnershipError:
                # A changed identity is a hard stop; never retry or hide it.
                raise
            except Exception as exc:
                last_error = exc
        return {
            "status": "residual_pending_container_close",
            "resource": "task_capture_temporary_archive",
            "path": path,
            "container_id": self._container_id,
            "owner_id": self._owner_id,
            "operation": "remove_exact_temporary_archive",
            "attempts": _CLEANUP_ATTEMPTS,
            "error_type": type(last_error).__name__ if last_error is not None else None,
        }

    def restore(self, archive: bytes, path: str = "/") -> None:
        """Restore into the owned task namespace; does not wipe existing paths."""
        self._put_archive(archive, normalize_task_path(path, self.workdir))

    def _ensure_helper(self) -> None:
        with self._helper_lock:
            if self._helper_state == "ready":
                return
            if self._helper_state != "new":
                raise TaskRuntimeError(
                    "Task helper bootstrap failed; automatic restart refused"
                )
            self._helper_state = "starting"
            try:
                source = Path(__file__).with_name("task_runtime.py").read_bytes()
                self._put_archive(
                    _tar_file(
                        self._helper_path.lstrip("/"),
                        source,
                        self._helper_dir.lstrip("/"),
                    ),
                    "/",
                )
                code, _, _ = self._exec_bytes(
                    [
                        "python3",
                        "-I",
                        "-S",
                        self._helper_path,
                        "launch",
                        self._socket_path,
                        self.workdir,
                    ],
                    timeout=30,
                )
                if code:
                    raise TaskRuntimeError(
                        "Task helper startup failed; it was not restarted"
                    )
                self._helper_state = "ready"
            except BaseException:
                self._helper_state = "failed"
                raise

    def _rpc(self, request: dict[str, Any], timeout: float) -> dict[str, Any]:
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("Task tool arguments exceed the 64 KiB JSON limit")
        encoded = base64.b64encode(payload).decode("ascii")
        code, stdout, _ = self._exec_bytes(
            [
                "python3",
                "-I",
                "-S",
                self._helper_path,
                "rpc",
                self._socket_path,
                encoded,
            ],
            timeout=timeout,
            output_limit=MAX_RPC_OUTPUT_BYTES,
        )
        if code:
            raise TaskRuntimeError(
                "Task RPC failed; action outcome is unknown; not retried"
            )
        try:
            result = json.loads(stdout)
        except (ValueError, UnicodeError):
            raise TaskRuntimeError(
                "Invalid task RPC response; action was not retried"
            ) from None
        if not isinstance(result, dict) or "error" in result:
            raise TaskRuntimeError("Task RPC reported failure; action was not retried")
        return result

    def call_tool(self, tool_name: str, action: dict) -> dict:
        """Invoke one task-side action, never a controller-local tool executor."""
        if tool_name not in ("terminal", "file_editor") or not isinstance(action, dict):
            raise ValueError("Only terminal and file_editor task actions are supported")
        # Validate the wire envelope before any bootstrap or side effects.
        request = {"operation": "tool", "tool": tool_name, "action": action}
        payload = json.dumps(request, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("Task tool arguments exceed the 64 KiB JSON limit")
        timeout = 30.0
        if tool_name == "terminal":
            duration = action.get("timeout")
            duration = 10.0 if duration is None else duration
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or not 0 <= duration <= 300
            ):
                raise ValueError("Terminal timeout must be between 0 and 300 seconds")
            timeout = float(duration) + 15.0
        self._owned()
        self._ensure_helper()
        return self._rpc(request, timeout)

    def interrupt(self) -> None:
        self._owned()
        if self._helper_state == "ready":
            self._rpc({"operation": "interrupt"}, timeout=10)

    def close(self) -> None:
        """Stop/remove only the verified immutable ID, never by name or listing."""
        with self._close_lock:
            if self._closed:
                return
            operation = "inspect_owned_container"
            last_error: Exception | None = None
            for attempt in range(1, _CLEANUP_ATTEMPTS + 1):
                operation = "inspect_owned_container"
                try:
                    if self._container_id is not None:
                        owned = self._owned()
                        if owned.attrs.get("State", {}).get("Running"):
                            operation = "stop_owned_container"
                            self._api().stop(self._container_id, timeout=5)
                        operation = "reverify_owned_container"
                        self._owned()
                        operation = "remove_owned_container"
                        self._api().remove_container(
                            self._container_id, v=True, force=False
                        )
                    operation = "close_owned_docker_client"
                    if self._owns_client:
                        self.client.close()
                    self._closed = True
                    for diagnostic in self._cleanup_diagnostics:
                        if diagnostic.get("resource") == "task_capture_temporary_archive":
                            diagnostic["status"] = "removed_with_owned_container"
                    self._cleanup_diagnostics.append(
                        {
                            "status": "cleaned",
                            "resource": "owned_task_container",
                            "container_id": self._container_id,
                            "owner_id": self._owner_id,
                            "operation": "stop_and_remove_owned_container",
                            "attempts": attempt,
                        }
                    )
                    return
                except NotFound:
                    # The exact immutable ID is already absent, including the
                    # ambiguous case where a prior remove completed remotely.
                    if self._owns_client:
                        self.client.close()
                    self._closed = True
                    for diagnostic in self._cleanup_diagnostics:
                        if diagnostic.get("resource") == "task_capture_temporary_archive":
                            diagnostic["status"] = "removed_with_owned_container"
                    self._cleanup_diagnostics.append(
                        {
                            "status": "cleaned",
                            "resource": "owned_task_container",
                            "container_id": self._container_id,
                            "owner_id": self._owner_id,
                            "operation": "owned_container_already_absent",
                            "attempts": attempt,
                        }
                    )
                    return
                except OwnershipError as exc:
                    diagnostic = {
                        "status": "residual_unverified_ownership",
                        "resource": "owned_task_container",
                        "container_id": self._container_id,
                        "owner_id": self._owner_id,
                        "operation": operation,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                    }
                    self._cleanup_diagnostics.append(diagnostic)
                    raise
                except _IO_ERRORS as exc:
                    last_error = exc
                    if attempt == _CLEANUP_ATTEMPTS:
                        break
            diagnostic = {
                "status": "residual_cleanup_failed",
                "resource": "owned_task_container",
                "container_id": self._container_id,
                "owner_id": self._owner_id,
                "operation": operation,
                "attempts": _CLEANUP_ATTEMPTS,
                "error_type": type(last_error).__name__ if last_error is not None else None,
            }
            self._cleanup_diagnostics.append(diagnostic)
            raise TaskCleanupError(
                "Owned task container cleanup failed", diagnostic
            ) from None
