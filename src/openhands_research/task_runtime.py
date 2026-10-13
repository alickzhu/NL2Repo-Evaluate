"""Task-only, stdlib RPC service. Run with Python >= 3.9, never as a host tool.

The Unix socket, shell, command files and editor history all live in the task
container. This is a transport boundary, not a sandbox within that container.
"""

from __future__ import annotations

import base64
import collections
import errno
import fcntl
import json
import math
import os
import pty
import re
import select
import shlex
import signal
import socket
import socketserver
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path


MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MAX_HISTORY_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT = 10.0
MAX_TIMEOUT = 300.0
CLIPPED = "\n<response clipped>\n"
KEYS = {
    "ENTER": b"\n",
    "TAB": b"\t",
    "ESC": b"\x1b",
    "BS": b"\x7f",
    "UP": b"\x1b[A",
    "DOWN": b"\x1b[B",
    "RIGHT": b"\x1b[C",
    "LEFT": b"\x1b[D",
    "HOME": b"\x1b[H",
    "END": b"\x1b[F",
    "PGUP": b"\x1b[5~",
    "PGDN": b"\x1b[6~",
}


def wait_seconds(value):
    if value is None:
        return DEFAULT_TIMEOUT
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("timeout must be a number")
    if not math.isfinite(value) or not 0 <= value <= MAX_TIMEOUT:
        raise ValueError("timeout must be between 0 and 300 seconds")
    return float(value)


def clipped(text):
    if len(text) <= MAX_OUTPUT_BYTES:
        return text
    return text[: MAX_OUTPUT_BYTES // 2] + CLIPPED + text[-MAX_OUTPUT_BYTES // 2 :]


class PersistentShell:
    """One real interactive Bash, including its controlling PTY and job state."""

    def __init__(self, workdir, runtime_dir):
        self.workdir = workdir
        self.runtime_dir = runtime_dir
        self._lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._process = None
        self._master = None
        self._command_file = None
        self._pending = False
        self._buffer = b""
        self._cwd = workdir
        self._exit_code = 0
        self._start()

    def _start(self):
        marker = uuid.uuid4().hex.encode("ascii")
        self._begin = b"\x1e" + marker + b":"
        self._end = b"\x1f" + marker + b"\x1e"
        self._pattern = re.compile(
            re.escape(self._begin) + rb"([0-9]+):(.*?)" + re.escape(self._end),
            re.DOTALL,
        )
        env = dict(os.environ)
        env.pop("BASH_ENV", None)
        env.pop("ENV", None)
        env.update(
            TERM="dumb",
            PS1="",
            PS2="",
            PS0="",
            HISTFILE="/dev/null",
            PROMPT_COMMAND=(
                "builtin printf '\\036"
                + marker.decode()
                + ":%s:%s\\037"
                + marker.decode()
                + '\\036\' "$?" "$PWD"'
            ),
        )
        master, slave = pty.openpty()
        attrs = termios.tcgetattr(slave)
        attrs[3] &= ~(termios.ECHO | termios.ECHONL)
        attrs[1] &= ~termios.ONLCR
        termios.tcsetattr(slave, termios.TCSANOW, attrs)
        try:
            # A tiny exec trampoline establishes the controlling tty without
            # preexec_fn (which is unsafe in the threaded RPC server).
            process = subprocess.Popen(
                [sys.executable, "-I", "-S", __file__, "shell"],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                cwd=self.workdir,
                env=env,
                start_new_session=True,
                close_fds=True,
            )
        except BaseException:
            os.close(master)
            raise
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        with self._lifecycle:
            self._master = master
            self._process = process
        self._buffer = b""
        self._cwd = self.workdir
        self._pending = True
        _, done, _ = self._collect(5.0)
        if not done or process.poll() is not None:
            self.close()
            raise RuntimeError("Bash did not become ready")

    def _write(self, data):
        deadline = time.monotonic() + 5.0
        while data:
            with self._lifecycle:
                master = self._master
            if master is None:
                raise ValueError("Terminal is closed; use reset=true")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([], [master], [], remaining)[1]:
                raise ValueError("Terminal input stalled; input was not retried")
            try:
                count = os.write(master, data[:4096])
            except BlockingIOError:
                continue
            data = data[count:]

    def interrupt(self):
        with self._lifecycle:
            if self._master is not None and self._process.poll() is None:
                try:
                    os.write(self._master, b"\x03")
                except OSError as exc:
                    if exc.errno not in (errno.EIO, errno.EBADF):
                        raise

    def _finished(self):
        self._pending = False
        if self._command_file is not None:
            Path(self._command_file).unlink(missing_ok=True)
            self._command_file = None

    def _collect(self, timeout):
        deadline = time.monotonic() + timeout
        output = bytearray()
        dropped = False
        exhausted = False

        def append(data):
            nonlocal dropped
            output.extend(data)
            if len(output) > MAX_OUTPUT_BYTES:
                del output[:-MAX_OUTPUT_BYTES]
                dropped = True

        while True:
            match = self._pattern.search(self._buffer)
            if match:
                append(self._buffer[: match.start()])
                self._buffer = self._buffer[match.end() :]
                self._exit_code = int(match.group(1))
                self._cwd = match.group(2).decode("utf-8", "replace")
                self._finished()
                return bytes(output), True, dropped

            # Preserve a possibly split sentinel, but never retain an unbounded
            # command log just to detect completion.
            start = self._buffer.find(self._begin)
            keep = len(self._begin) if start < 0 else len(self._buffer) - start
            keep = min(keep, 8192)
            if len(self._buffer) > keep:
                append(self._buffer[:-keep])
                self._buffer = self._buffer[-keep:]

            remaining = max(0.0, deadline - time.monotonic())
            master = self._master
            ready = (
                not exhausted
                and master is not None
                and select.select([master], [], [], remaining)[0]
            )
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except BlockingIOError:
                    chunk = None
                except OSError as exc:
                    if exc.errno != errno.EIO:
                        raise
                    chunk = b""
                if chunk:
                    self._buffer += chunk
                    exhausted = time.monotonic() >= deadline
                    continue
                if chunk == b"":
                    self._process.wait(timeout=1)

            if self._process.poll() is not None:
                append(self._buffer)
                self._buffer = b""
                self._exit_code = self._process.returncode
                self._finished()
                return bytes(output), True, dropped

            if time.monotonic() >= deadline:
                # Emit everything except an incomplete sentinel prefix.
                start = self._buffer.find(self._begin)
                keep = len(self._buffer) - start if start >= 0 else 0
                if not keep:
                    for size in range(min(len(self._buffer), len(self._begin)), 0, -1):
                        if self._begin.startswith(self._buffer[-size:]):
                            keep = size
                            break
                if keep:
                    append(self._buffer[:-keep])
                    self._buffer = self._buffer[-keep:]
                else:
                    append(self._buffer)
                    self._buffer = b""
                return bytes(output), False, dropped

    def _observation(self, command, text, done=True, error=False):
        code = self._exit_code if done else -1
        return {
            "command": command,
            "text": text,
            "is_error": error,
            "exit_code": code,
            "timeout": not done,
            "metadata": {
                "exit_code": code,
                "pid": self._process.pid,
                "working_dir": self._cwd,
            },
            "full_output_save_dir": None,
        }

    def execute(self, action):
        command = action.get("command", "")
        with self._lock:
            try:
                if not isinstance(command, str) or "\x00" in command:
                    raise ValueError("command must be a string without NUL bytes")
                timeout = wait_seconds(action.get("timeout"))
                is_input = action.get("is_input", False)
                reset = action.get("reset", False)
                if not isinstance(is_input, bool) or not isinstance(reset, bool):
                    raise ValueError("is_input and reset must be booleans")
                if reset and is_input:
                    raise ValueError("reset cannot be combined with is_input")
                if reset:
                    self.close()
                    self._start()
                if self._process.poll() is not None:
                    raise ValueError("Terminal is closed; use reset=true explicitly")
                if is_input and command:
                    if not self._pending:
                        raise ValueError("No running command to receive input")
                    if re.fullmatch(r"C-[a-zA-Z]", command):
                        data = bytes([ord(command[-1].lower()) - ord("a") + 1])
                    else:
                        data = KEYS.get(command)
                        if data is None:
                            data = command if command.endswith("\n") else command + "\n"
                            data = data.encode("utf-8")
                            if any(len(line) > 4095 for line in data.split(b"\n")):
                                raise ValueError(
                                    "PTY input lines are limited to 4095 bytes; "
                                    "input was not sent. Use a file for longer data."
                                )
                    self._write(data)
                elif command:
                    if self._pending:
                        raise ValueError(
                            "Previous command is still pending. Use an empty command "
                            "to collect it, is_input to interact, or reset explicitly. "
                            "The new command was NOT executed."
                        )
                    fd, filename = tempfile.mkstemp(
                        prefix="command-", suffix=".sh", dir=self.runtime_dir
                    )
                    with os.fdopen(fd, "w", encoding="utf-8", newline="") as file:
                        file.write(command)
                        file.write("\n")
                    self._command_file = filename
                    self._pending = True
                    # Sourcing a file avoids PTY canonical-line truncation and
                    # prevents read/heredocs from eating an appended sentinel.
                    self._write(
                        ("builtin source " + shlex.quote(filename) + "\n").encode()
                    )
                elif not self._pending:
                    return self._observation(command, "No command is running.")

                output, done, dropped = self._collect(timeout)
                text = output.decode("utf-8", "replace")
                if dropped:
                    text = CLIPPED + text
                closed = self._process.poll() is not None
                if closed:
                    text += (
                        "\n[Terminal closed. Use reset=true to start a new session.]"
                    )
                elif not done:
                    text += (
                        "\n[Soft timeout: command still running, not rerun. "
                        "Send an empty command to continue or C-c with is_input=true.]"
                    )
                return self._observation(command, text, done, error=closed)
            except (ValueError, OSError) as exc:
                return self._observation(
                    command, str(exc), not self._pending, error=True
                )

    def close(self):
        with self._lifecycle:
            master, process = self._master, self._process
            self._master = None
            if process is not None and process.poll() is None:
                groups = {process.pid}
                if master is not None:
                    try:
                        groups.add(os.tcgetpgrp(master))
                    except OSError:
                        pass
                for name in os.listdir("/proc"):
                    if name.isdigit():
                        try:
                            if os.getsid(int(name)) == process.pid:
                                groups.add(os.getpgid(int(name)))
                        except OSError:
                            pass
                for group in groups:
                    if group > 1 and group != os.getpgrp():
                        try:
                            os.killpg(group, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                process.wait(timeout=5)
            if master is not None:
                os.close(master)
        self._finished()


class FileEditor:
    """Atomic UTF-8 edits; bounded, conflict-checked undo in task memory only."""

    def __init__(self):
        self._lock = threading.Lock()
        self._history = collections.deque()
        self._history_bytes = 0

    @staticmethod
    def _read(path):
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as file:
            info = os.fstat(file.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Only regular UTF-8 files are supported (no symlinks)")
            if info.st_size > MAX_FILE_BYTES:
                raise ValueError("File exceeds the 1 MiB editor limit")
            data = file.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("File exceeds the 1 MiB editor limit")
        if b"\0" in data:
            raise ValueError("Binary files are not supported")
        return data.decode("utf-8"), stat.S_IMODE(info.st_mode)

    @staticmethod
    def _write(path, text, mode=0o644, create=False):
        data = text.encode("utf-8")
        if len(data) > MAX_FILE_BYTES or b"\0" in data:
            raise ValueError("Edits require UTF-8 text without NUL, at most 1 MiB")
        fd, filename = tempfile.mkstemp(prefix=".rgoh-edit-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as file:
                os.fchmod(file.fileno(), mode)
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            if create:
                os.link(filename, path)
            else:
                os.replace(filename, path)
        finally:
            Path(filename).unlink(missing_ok=True)

    def _remember(self, path, before, after, mode):
        size = len((before or "").encode()) + len(after.encode())
        self._history.append((str(path), before, after, mode, size))
        self._history_bytes += size
        while self._history_bytes > MAX_HISTORY_BYTES or len(self._history) > 32:
            self._history_bytes -= self._history.popleft()[-1]

    @staticmethod
    def _directory(path):
        result = []

        def visit(directory, depth):
            with os.scandir(directory) as entries:
                visible = []
                for entry in entries:
                    if not entry.name.startswith("."):
                        visible.append(entry)
                    if len(visible) >= 1001:
                        break
            for entry in sorted(visible, key=lambda item: item.name):
                if len(result) >= 1000:
                    return
                result.append(
                    entry.path + ("/" if entry.is_dir(follow_symlinks=False) else "")
                )
                if depth and entry.is_dir(follow_symlinks=False):
                    visit(entry.path, depth - 1)

        visit(str(path), 1)
        text = "\n".join(result)
        if len(result) >= 1000:
            text += CLIPPED
        return clipped(text)

    def execute(self, action):
        command, raw_path = action.get("command"), action.get("path")
        result = {"command": command, "path": raw_path, "is_error": False}
        with self._lock:
            try:
                if not isinstance(raw_path, str) or "\x00" in raw_path:
                    raise ValueError("path must be an absolute task path")
                path = Path(raw_path)
                if not path.is_absolute():
                    raise ValueError("path must be absolute inside the task container")
                path = Path(os.path.normpath(path))
                result["path"] = str(path)
                if path.is_symlink():
                    raise ValueError("Symlink editor targets are not supported")
                if command == "view" and path.is_dir():
                    if action.get("view_range") is not None:
                        raise ValueError("view_range is not supported for directories")
                    result["text"] = self._directory(path)
                    return result
                if command == "create":
                    text = action.get("file_text")
                    if not isinstance(text, str):
                        raise ValueError("create requires file_text")
                    self._write(path, text, create=True)
                    self._remember(path, None, text, 0o644)
                    result.update(
                        text="File created successfully.",
                        prev_exist=False,
                        new_content=text,
                    )
                    return result

                before, mode = self._read(path)
                if command == "view":
                    lines = before.splitlines()
                    bounds = action.get("view_range")
                    first, last = 1, len(lines)
                    if bounds is not None:
                        if (
                            not isinstance(bounds, list)
                            or len(bounds) != 2
                            or any(type(value) is not int for value in bounds)
                        ):
                            raise ValueError("view_range must be [start, end]")
                        first, last = bounds
                        if last == -1:
                            last = len(lines)
                        if first < 1 or last < first or last > len(lines):
                            raise ValueError("view_range is outside the file")
                    result["text"] = clipped(
                        "".join(
                            f"{index:6}\t{line}\n"
                            for index, line in enumerate(lines[first - 1 : last], first)
                        )
                    )
                    return result
                if command == "str_replace":
                    old = action.get("old_str")
                    after = action.get("new_str")
                    after = "" if after is None else after
                    if (
                        not isinstance(old, str)
                        or not old
                        or not isinstance(after, str)
                    ):
                        raise ValueError(
                            "str_replace requires nonempty old_str and text new_str"
                        )
                    index = before.find(old)
                    if index < 0 or before.find(old, index + 1) >= 0:
                        raise ValueError(
                            "old_str must match exactly one occurrence; no edit made"
                        )
                    if old == after:
                        raise ValueError("old_str and new_str must differ")
                    after = before[:index] + after + before[index + len(old) :]
                elif command == "insert":
                    line, text = action.get("insert_line"), action.get("new_str")
                    lines = before.split("\n")
                    if type(line) is not int or not 0 <= line <= len(lines):
                        raise ValueError("insert_line is outside the file")
                    if not isinstance(text, str):
                        raise ValueError("insert requires new_str")
                    after = "\n".join(lines[:line] + text.split("\n") + lines[line:])
                elif command == "undo_edit":
                    for index in range(len(self._history) - 1, -1, -1):
                        entry = self._history[index]
                        if entry[0] == str(path):
                            break
                    else:
                        raise ValueError("No retained editor history for this file")
                    _, after, expected, old_mode, size = entry
                    if before != expected:
                        raise ValueError(
                            "File changed outside the editor; undo refused"
                        )
                    if after is None:
                        path.unlink()
                    else:
                        self._write(path, after, old_mode)
                    del self._history[index]
                    self._history_bytes -= size
                    result.update(
                        text="Last edit undone.", old_content=before, new_content=after
                    )
                    return result
                else:
                    raise ValueError("Unsupported file_editor command")
                self._write(path, after, mode)
                self._remember(path, before, after, mode)
                result.update(
                    text="File edited successfully.",
                    old_content=before,
                    new_content=after,
                )
            except (OSError, ValueError, UnicodeError) as exc:
                result.update(is_error=True, text=str(exc))
        return result


def receive(sock, limit):
    def exact(size):
        result = bytearray()
        while len(result) < size:
            chunk = sock.recv(min(size - len(result), 65536))
            if not chunk:
                raise ConnectionError("RPC disconnected; request will not be retried")
            result.extend(chunk)
        return bytes(result)

    size = struct.unpack("!I", exact(4))[0]
    if size > limit:
        raise ValueError("RPC message exceeds its size limit")
    return exact(size)


def send(sock, data):
    sock.sendall(struct.pack("!I", len(data)) + data)


class RuntimeServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, socket_path, workdir):
        self.terminal = PersistentShell(workdir, str(Path(socket_path).parent))
        self.editor = FileEditor()
        try:
            super().__init__(socket_path, RuntimeHandler)
            os.chmod(socket_path, 0o600)
        except BaseException:
            self.terminal.close()
            raise

    def server_close(self):
        super().server_close()
        self.terminal.close()


class RuntimeHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(MAX_TIMEOUT + 15)
        try:
            request = json.loads(receive(self.request, MAX_REQUEST_BYTES))
            operation = request.get("operation")
            if operation == "interrupt":
                self.server.terminal.interrupt()
                response = {"interrupted": True}
            elif operation == "ping":
                response = {"ready": True}
            elif operation == "tool" and isinstance(request.get("action"), dict):
                if request.get("tool") == "terminal":
                    response = self.server.terminal.execute(request["action"])
                elif request.get("tool") == "file_editor":
                    response = self.server.editor.execute(request["action"])
                else:
                    raise ValueError("Unknown task tool")
            else:
                raise ValueError("Invalid task RPC request")
        except Exception as exc:
            response = {
                "error": f"Task RPC failed ({type(exc).__name__}); "
                "outcome may be unknown. "
                "Request was not retried."
            }
        try:
            data = json.dumps(response, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
            if len(data) > MAX_RESPONSE_BYTES:
                data = b'{"error":"RPC response too large; request was not retried"}'
            send(self.request, data)
        except (OSError, UnicodeError):
            pass


def rpc(socket_path, encoded):
    data = base64.b64decode(encoded, validate=True)
    if len(data) > MAX_REQUEST_BYTES:
        raise ValueError("RPC request exceeds 64 KiB")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(MAX_TIMEOUT + 15)
        sock.connect(socket_path)
        send(sock, data)
        return receive(sock, MAX_RESPONSE_BYTES)


def launch(socket_path, workdir):
    read_fd, write_fd = os.pipe()
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                __file__,
                "serve",
                socket_path,
                workdir,
                str(write_fd),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
    finally:
        os.close(write_fd)
    try:
        if (
            not select.select([read_fd], [], [], 10)[0]
            or os.read(read_fd, 5) != b"READY"
        ):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
            raise RuntimeError("Task service failed to start; it was not restarted")
    finally:
        os.close(read_fd)


def main():
    mode = sys.argv[1]
    if mode == "shell":
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)
        os.execvp("bash", ["bash", "--noprofile", "--norc", "--noediting", "-i"])
    elif mode == "serve":
        with RuntimeServer(sys.argv[2], sys.argv[3]) as server:
            ready_fd = int(sys.argv[4])
            os.write(ready_fd, b"READY")
            os.close(ready_fd)
            server.serve_forever()
    elif mode == "launch":
        launch(sys.argv[2], sys.argv[3])
    elif mode == "rpc":
        sys.stdout.buffer.write(rpc(sys.argv[2], sys.argv[3]))
        sys.stdout.buffer.flush()
    else:
        raise ValueError("Unknown runtime mode")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Task runtime failed ({type(error).__name__}); no automatic retry",
            file=sys.stderr,
        )
        sys.exit(1)
