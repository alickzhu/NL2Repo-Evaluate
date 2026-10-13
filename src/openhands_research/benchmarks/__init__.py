"""Only the current AutoNL2Repo task contract is supported."""
from .work import AutoNL2RepoAdapter


def get_benchmark(kind):
    if kind != "autonl2repo":
        raise ValueError("This local package accepts current AutoNL2Repo JSON tasks only")
    return AutoNL2RepoAdapter()

