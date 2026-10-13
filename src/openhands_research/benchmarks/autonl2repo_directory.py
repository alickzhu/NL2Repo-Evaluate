"""Original directory-submission validator, without legacy pinned scorers."""
import copy
import io
import posixpath
import tarfile
from .. import grading
from ..contracts import relative_path
from ..grading import validate_submission


def validate_directory_submission(
    archive: bytes, *, root_name: str = "workspace"
) -> bytes:
    if not isinstance(archive, bytes):
        raise ValueError("submission must be tar bytes")
    if (
        len(archive)
        > grading.MAX_SUBMISSION_BYTES + grading.MAX_SUBMISSION_FILES * 1536 + 10240
    ):
        raise ValueError("submission tar exceeds its wire-size limit")
    filtered, output, links = io.BytesIO(), io.BytesIO(), {}
    total = 0
    try:
        with (
            tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source,
            tarfile.open(fileobj=filtered, mode="w") as temporary,
        ):
            for index, member in enumerate(source, 1):
                if index > grading.MAX_SUBMISSION_FILES:
                    raise ValueError("submission exceeds the member-count limit")
                name = relative_path(member.name)
                parts = name.split("/")
                if parts[0] != root_name:
                    raise ValueError("submission has the wrong root")
                total += member.size
                if total > grading.MAX_SUBMISSION_BYTES:
                    raise ValueError("submission exceeds 128 MiB of file content")
                if (
                    any(
                        part
                        in {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
                        for part in parts[1:]
                    )
                    or name == root_name + "/start.md"
                    or name.endswith((".pyc", ".pyo"))
                ):
                    continue
                normalized = copy.copy(member)
                if member.issym():
                    link = member.linkname
                    relative = "/".join(parts[1:])
                    target = posixpath.normpath(
                        posixpath.join(posixpath.dirname(relative), link)
                    )
                    if (
                        not link
                        or link.startswith("/")
                        or "\\" in link
                        or any(ord(char) < 32 or ord(char) == 127 for char in link)
                        or target == ".."
                        or target.startswith("../")
                    ):
                        raise ValueError("submission symlink leaves the implementation")
                    links[relative] = link
                    normalized.type, normalized.linkname, normalized.size = (
                        tarfile.REGTYPE,
                        "",
                        0,
                    )
                temporary.addfile(
                    normalized, source.extractfile(member) if member.isfile() else None
                )
        checked = validate_submission(filtered.getvalue(), root_name=root_name)
        with (
            tarfile.open(fileobj=io.BytesIO(checked), mode="r:") as source,
            tarfile.open(fileobj=output, mode="w") as destination,
        ):
            for member in source:
                if member.name in links:
                    member.type = tarfile.SYMTYPE
                    member.linkname = links[member.name]
                    destination.addfile(member)
                else:
                    destination.addfile(
                        member, source.extractfile(member) if member.isfile() else None
                    )
    except (tarfile.TarError, OSError, EOFError, UnicodeError) as exc:
        raise ValueError("invalid directory submission tar") from exc
    return output.getvalue()
