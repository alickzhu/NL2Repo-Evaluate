"""Local four-Spec generation experiment; records no fabricated scores."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
batch = Path(sys.argv[1]).resolve()
rows = json.loads((batch / "inputs.json").read_text())
env = {**os.environ, "OPENHANDS_SUPPRESS_BANNER": "1",
       "OPENAI_API_KEY": os.environ.get("COPILOT_API_KEY") or os.environ.get("OPENAI_API_KEY", "")}
if not env["OPENAI_API_KEY"]:
    raise SystemExit("No API credential in environment")


def run(row):
    command = [str(root / ".venv/bin/python"), "-m", "openhands_research", "generate",
               "--task", row["input"], "--run-id", row["repo"],
               "--output", str(batch / "generations"), "--api-mode", "responses",
               "--base-url", "http://127.0.0.1:4145/v1", "--model", "gpt-5.6-luna",
               "--generation-timeout", "2400", "--request-timeout", "180",
               "--max-iterations", "200", "--memory", "8g", "--cpus", "4",
               "--max-output-tokens", "16384", "--max-tokens-ceiling", "32768"]
    started = time.time()
    log = batch / (row["repo"] + ".log")
    with log.open("x") as output:
        os.chmod(log, 0o600)
        child = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
        result = {**row, "pid": child.pid, "command": command, "started_at": started,
                  "log": str(log), "status": "running"}
        (batch / (row["repo"] + ".status.json")).write_text(json.dumps(result, indent=2) + "\n")
        print("STARTED", row["repo"], child.pid, flush=True)
        result.update(returncode=child.wait(), elapsed_seconds=round(time.time() - started, 2),
                      status="finished")
    reports = list((batch / "generations" / row["repo"]).glob("tasks/*/attempt-0000/result.json"))
    if reports:
        result["result"] = json.loads(reports[0].read_text())
    (batch / (row["repo"] + ".status.json")).write_text(json.dumps(result, indent=2) + "\n")
    print("FINISHED", row["repo"], result["returncode"], result["elapsed_seconds"], flush=True)
    return result


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, rows))
    (batch / "results.json").write_text(json.dumps(results, indent=2) + "\n")
