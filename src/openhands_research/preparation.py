"""Image and network preparation copied from the existing runner."""
from contextlib import contextmanager
import fcntl
import os
import re
from pathlib import Path
import docker
from docker.errors import ImageNotFound
from .contracts import DatasetTask
from .runtime import TaskContainer


@contextmanager
def image_use_lock():
    """Concurrent pulls share a lock; the optional job-local GC is exclusive."""
    path = os.environ.get("REPOGEN_DOCKER_IMAGE_LOCK")
    if not path:
        yield
        return
    if not Path(path).is_absolute():
        raise ValueError("Docker image lock must be an absolute controller path")
    with Path(path).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _image(client, reference: str):
    with image_use_lock():
        try:
            return client.images.get(reference)
        except ImageNotFound:
            return client.images.pull(reference)


def _image_id(image) -> str:
    identity = image.id
    if (
        not isinstance(identity, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", identity) is None
    ):
        raise ValueError("Docker image has no immutable local identity")
    return identity


def prepare_generic_image(reference: str) -> dict[str, str]:
    """Cache the caller-selected shared image without task-specific preparation."""
    client = docker.from_env(timeout=900)
    try:
        image = _image(client, reference)
        return {"reference": reference, "image_id": _image_id(image)}
    finally:
        client.close()


def probe_development_network(image: str, run_id: str) -> dict:
    """Use the exact sandbox policy, not the network of the trusted controller.

    A disposable container downloads one unrelated public wheel, then is removed.
    It is never used by an Agent and does not preinstall task dependencies.
    """
    program = """import json, pathlib, socket, subprocess, sys
print(json.dumps({'dns': pathlib.Path('/etc/resolv.conf').read_text(),
                  'pypi_addresses': sorted({i[4][0] for i in socket.getaddrinfo('pypi.org', 443)})}), flush=True)
subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'download',
    '--index-url', 'https://pypi.org/simple', '--no-deps', '--only-binary=:all:',
    '--retries', '0', '--timeout', '20', '--dest', '/tmp/rgoh-network-probe',
    'packaging==24.2'], check=True, timeout=90)
print('REPOGEN_NETWORK_OK', flush=True)
"""
    with TaskContainer(image, run_id, "development-network-preflight",
                       network_disabled=False, mem_limit="1g") as target:
        code, out, err = target.exec(["python3", "-I", "-c", program], timeout=120)
    if code or "REPOGEN_NETWORK_OK" not in out:
        raise RuntimeError("Development sandbox cannot download public dependencies: "
                           + (out + err)[-5000:])
    return {"status": "passed", "network": "bridge", "probe": "pypi-wheel-download",
            "output": out[-5000:]}


def prepare_grading_image(task: DatasetTask) -> dict[str, str]:
    client = docker.from_env(timeout=900)
    try:
        image = _image(client, task.public.image)
        repository = (
            task.public.image.split("@", 1)[0]
            .split(":", 1)[0]
            .removeprefix("docker.io/")
        )
        immutable = next(
            (
                item
                for item in image.attrs.get("RepoDigests", [])
                if item.split("@", 1)[0].removeprefix("docker.io/") == repository
            ),
            "",
        )
        if not immutable:
            raise ValueError("matching immutable grading image digest is required")
        digest = immutable.split("@", 1)[1]
        declared = task.private.payload.get("task", {}).get("image_digest")
        if "@" in task.public.image:
            declared = task.public.image.split("@", 1)[1]
        if declared and declared != digest:
            raise ValueError("grading image differs from its declared digest")
        return {
            "image": task.public.image,
            "image_id": _image_id(image),
            "immutable_image": immutable,
            "image_digest": digest,
        }
    finally:
        client.close()
