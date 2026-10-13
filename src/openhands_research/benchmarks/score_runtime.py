#!/usr/bin/env python3
"""Resource/isolation flags only. All scoring remains in the public evaluator."""

import os
import sys


def command(arguments):
    executable = os.environ.get("AUTONL2REPO_DOCKER", "docker")
    if arguments and arguments[0] == "create":
        return [
            executable,
            "create",
            "--memory",
            os.environ["REPOGEN_SCORE_MEMORY"],
            "--cpus",
            os.environ["REPOGEN_SCORE_CPUS"],
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            *arguments[1:],
        ]
    return [executable, *arguments]


def main():
    args = command(sys.argv[1:])
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
