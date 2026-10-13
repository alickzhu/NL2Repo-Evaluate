"""AutoNL2Repo's reference evaluation interface."""

from .evaluator import audit_reference, evaluate
from ._version import CANONICAL_RUNNER_SHA256, TASKFORGE_SOURCE_COMMIT, __version__

__all__ = [
    "CANONICAL_RUNNER_SHA256",
    "TASKFORGE_SOURCE_COMMIT",
    "__version__",
    "evaluate",
    "audit_reference",
]
