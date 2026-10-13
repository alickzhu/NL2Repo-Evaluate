"""Legacy command blocklist, adapted to fail-closed OpenHands PreToolUse hooks.

The rule builder and prompt preserve our 1.6 policy. This is NOT a firewall.
The standalone hook uses only the standard library, including under python -I.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys

_PACKAGE_MANAGERS = [
    # `show` is on denovo's forbidden list and leaks the real version and
    # dependency set, and in these logs it is repeatedly the first half of
    # `pip show X | head -5` followed by `python -c "import X; print(X.__file__)"`
    # -- reconnaissance for where the upstream source lives.
    #
    # `uninstall` is deliberately NOT here, and denovo does not list it either.
    # It is tempting: in the open-book logs `pip uninstall -y <target>` is
    # usually the second half of a cheat (`pip install jsonlines` comes first,
    # then the source is read, then it is uninstalled before testing). But
    # blocking it buys nothing -- the install is already blocked, so the cleanup
    # never happens -- and it has a real false positive, because the agent's own
    # `pip install -e .` registers a distribution under the target's name:
    #
    #     pip uninstall -y jinja2 && pip install -e .     # reinstalling ITS OWN code
    #
    # Measured: adding `uninstall` moved the block rate from 1.69% to 2.40% and
    # every new hit was of that shape.
    r"pip[\d.]*\s+(?:install|download|wheel|fetch|show|index\s+versions)",
    r"python[\d.]*\s+-m\s+pip\d*\s+(?:install|download|wheel|fetch|show)",
    r"pipx\s+(?:install|run|fetch|inject)",
    r"pipenv\s+(?:install|update)",
    r"uv\s+(?:pip\s+)?(?:install|download|add|tool\s+install)",
    r"uvx\b",
    r"poetry\s+(?:add|fetch)",
    r"(?:conda|mamba|micromamba)\s+(?:install|create|update|search)",
    r"easy_install[\d.]*",
    r"pdm\s+add",
    r"hatch\s+run\s+pip",
]

_DOWNLOADERS = r"(?:curl|wget|aria2c?|httpie|lwp-download)"

_HOSTS = (r"pypi\.org|pypi\.python\.org|test\.pypi\.org|files\.pythonhosted\.org"
          r"|pythonhosted\.org|pypistats\.org|conda\.anaconda\.org|anaconda\.org"
          r"|repo\.anaconda\.com")

LOCALHOST = r"(?:localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])"

SEGMENTS = r"(?:&&|\|\||;|\||\n)"

LEGACY_PROMPT = "\n\n<ANTI_CHEAT_CONSTRAINT>\n**ABSOLUTE RULE: DO NOT DOWNLOAD THE TARGET PACKAGE'S SOURCE CODE FROM THE INTERNET.** You MUST implement the target project from scratch based solely on the specification in start.md. It is a HARD VIOLATION to look up, download, install, clone, or copy the target package's source code from any external source -- this includes PyPI, GitHub, GitLab, Bitbucket, Codeberg, Anaconda, conda-forge, personal mirrors, company-internal mirrors, Google/Bing cache, the Wayback Machine, any local wheel/sdist cache on disk, or any other channel.\n\nSpecifically, the following are **FORBIDDEN** and will be auto-blocked:\n\n* Package managers pulling the target: `pip install <target>`, `pip3.X install <target>`, `pip download <target>`, `pip show <target>`, `pip wheel <target>`, `python -m pip install <target>`, `pipx install <target>`, `pipenv install <target>`, `uv pip install <target>`, `uv add <target>`, `uvx <target>`, `poetry add <target>`, `conda/mamba/micromamba install <target>`, `pdm add <target>`, `easy_install <target>` -- in every sudo-prefixed / env-prefixed / versioned form.\n* VCS: `git clone` or `git submodule add` pointing at the target repo (anywhere: github.com, gitlab.com, bitbucket.org, codeberg.org, any mirror).\n* Raw downloads: `curl`/`wget`/`aria2c`/`httpie`/`lwp-download` hitting PyPI-family hosts (`pypi.org`, `files.pythonhosted.org`, `pypi.python.org`, `test.pypi.org`, `pythonhosted.org`, `pypistats.org`, `conda.anaconda.org`, `anaconda.org`, `repo.anaconda.com`) or any URL containing the target name.\n* Python introspection: `python -c 'import <target>; inspect.getsource(<target>)'` or any `pkgutil`/`get_data` trick to dump the installed upstream source.\n* Extracting on-disk artefacts: `unzip <target>*.whl`, `tar xf <target>*.tar.gz`, etc.\n\nThe following are **ALLOWED**:\n\n* Installing *third-party* dependencies your own implementation needs (e.g. `pip install requests`, `pip install numpy`, `pip install pytest`).\n* Running `pip install -e .` on the code you yourself wrote in the workspace, and `pip uninstall <target>` to remove that same self-installed copy.\n* Running your own package's CLI (`python -m <target>`) to test what you wrote.\n* Checking where your own code resolves from, e.g. `python -c 'import <target>; print(<target>.__file__)'` after `pip install -e .` -- nothing named `<target>` is preinstalled here, so this can only ever show you your own workspace copy.\n* Requests to `localhost` / `127.0.0.1` against a server you are developing.\n* Reading any file already present in the workspace at the start of your session.\n\nViolations are detected automatically by the harness's security policy and will be blocked -- the blocked command returns an error instead of running. Repeated violations will not succeed; do not try to obfuscate (variable-name rewrites, base64 decode pipes, curl into a random-named file, etc.) -- the blocklist covers name variants (`Pkg` / `pkg` / `pkg-name` / `pkg_name` / `PKG_NAME`) and all common package-manager spellings. Every attempt is recorded.\n</ANTI_CHEAT_CONSTRAINT>"


def legacy_prompt():
    return LEGACY_PROMPT

def name_variants(*seeds):
    out = set()
    for s in seeds:
        s = (s or "").strip()
        if not s:
            continue
        if s.lower().startswith("git+"):
            s = s[4:]
        m = re.search(r"(?:github|gitlab|bitbucket|codeberg)\.(?:com|org)/"
                      r"([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$", s)
        names = [m.group(2), f"{m.group(1)}/{m.group(2)}"] if m else [s]
        for n in names:
            out |= {n, n.replace("-", "_"), n.replace("_", "-"),
                    n.replace(".", "_"), n.replace(".", "-")}
    out |= {v.replace("-", "_") for v in list(out)}
    out |= {v.replace("_", "-") for v in list(out)}
    # A two-character package name would turn every command into a violation.
    return sorted(v for v in out if len(v) >= 3 and not any(c.isspace() for c in v))

def _pkg_arg(word):
    """Match the name as a requirement argument, never as a path component."""
    return (r"(?<![/\w.\-])" + re.escape(word)
            + r"(?:\[[^\s]*\])?"
            + r"(?:(?:==|~=|!=|<=|>=|<|>)\S+)?"
            + r"(?=$|[\s\"'])")

def build_patterns(seeds):
    """Patterns for ``seeds``; ``seeds[0]`` (the directory name) is authoritative.

    The URL and VCS rules are built from the directory name only. A seed
    recovered from start.md is evidence about *what the agent would install or
    introspect*, not about which URLs are suspicious -- and for a handful of
    specs the dominant import is a local module rather than the distribution
    (``flasky`` -> ``app``, ``pytest-cov`` -> ``mod``, ``autorccar`` ->
    ``model``). Feeding those into ``curl [^\\n]*\\bapp\\b`` would block any
    download whose URL happens to contain "app". Feeding them into
    ``pip install`` costs nothing: nobody legitimately installs ``mod``.
    """
    variants = name_variants(*seeds)
    url_variants = name_variants(seeds[0]) if seeds else []
    patterns = []
    unsplit = []
    for v in variants:
        arg = _pkg_arg(v)
        for pm in _PACKAGE_MANAGERS:
            # Anchored at the start of a shell segment, tolerating sudo/env.
            patterns.append(
                rf"(?i)^\s*(?:sudo\s+|env\s+\S+=\S+\s+)*{pm}\b[^\n]*{arg}")
        # Reading the installed upstream source rather than downloading it.
        #
        # UNSPLIT, unlike everything else here. `;` is a shell separator but it
        # is also Python's statement separator, and the canonical form of this
        # probe puts both halves on one line either side of one:
        #
        #     python -c "import boltons, inspect; print(inspect.getsource(...))"
        #
        # Segment-splitting yields two halves and neither matches a rule that
        # requires both. Verified in the real app image: this was allowed
        # through until the rule moved here. `[^\n]*` already bounds the match
        # to a single line, so dropping the split does not widen it across a
        # heredoc.
        #
        # `__file__` is deliberately NOT in this alternation. It reads like the
        # same probe but it is the opposite: in this arm the runtime image is
        # bare, so nothing named <target> is importable except the agent's own
        # workspace copy, and `pip install -e . && python -c "import X;
        # print(X.__file__)"` is the agent checking its own editable install
        # resolved to /workspace instead of a stale site-packages copy -- the
        # hygiene we want. Measured on the 4226-command corpus: `__file__`
        # accounts for 30 of 33 hits here and the sampled ones are that shape,
        # while getsource/pkgutil are 3 hits and unambiguous.
        unsplit.append(rf"(?i)\bimport\s+{re.escape(v)}\b[^\n]*"
                       r"(?:inspect\.getsource|pkgutil|get_data)")
        # Unpacking a wheel/sdist already on disk (pip cache, earlier download).
        # Safe to build from every variant, unlike the URL rules: "<name>" next
        # to a ".tar.gz" is already specific enough that a local module name
        # can't collide with it by accident.
        patterns.append(rf"(?i)\b(?:unzip|tar|gunzip)\b[^\n]*\b{re.escape(v)}\b[^\n]*"
                        r"\.(?:whl|tar\.gz|tgz|zip)")
    for v in url_variants:
        qv = r"\b" + re.escape(v) + r"\b"
        patterns.append(rf"(?i)\bgit\s+(?:clone|submodule\s+add|fetch)\b[^\n]*{qv}")
        patterns.append(rf"(?i)\b{_DOWNLOADERS}\b[^\n]*{qv}")
    patterns.append(rf"(?i)\b{_DOWNLOADERS}\b[^\n]*(?:{_HOSTS})")
    patterns.append(rf"(?i)\bpip[\d.]*\s+(?:install|download)\b[^\n]*(?:{_HOSTS})")
    return variants, patterns, unsplit

def write_for(task_dir, seeds):
    variants, patterns, unsplit = build_patterns(seeds)
    for p in patterns + unsplit:
        re.compile(p)   # fail here, not inside the app container at run time
    spec = {
        "task": os.path.basename(task_dir.rstrip("/")),
        "seeds": seeds,
        "variants": variants,
        "patterns": patterns,
        # Matched against the raw text instead of per shell segment; see
        # build_patterns for why the introspection rule has to live here.
        "patterns_unsplit": unsplit,
        # The shim needs these to reproduce the matching semantics; hard-coding
        # them there and here is how the two drift apart.
        "segment_split": SEGMENTS,
        "skip_if_matches": LOCALHOST,
    }
    out = os.path.join(task_dir, "blocklist.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    return variants, patterns + unsplit

def prepare(directory, task, extra_seeds=()):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = [task['repo'], task.get('pypi_name'), *task.get('import_names', []),
             *(x['name'] for x in task.get('pypi_name_candidates', [])), *extra_seeds]
    seeds = list(dict.fromkeys(x for x in seeds if x))
    if not seeds:
        raise ValueError('no target identity for blocklist')
    write_for(str(directory), seeds)
    path = directory / 'blocklist.json'
    rules = json.loads(path.read_text())
    rules['task'] = task['instance_id']
    path.write_text(json.dumps(rules, indent=2))
    Guard(rules)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()

class Guard:
    def __init__(self, rules):
        self.patterns = [re.compile(p) for p in rules['patterns']]
        self.unsplit = [re.compile(p) for p in rules.get('patterns_unsplit', [])]
        if not self.patterns:
            raise ValueError('empty blocklist')
        self.split = re.compile(rules['segment_split']) if rules.get('segment_split') else None
        self.skip = re.compile(rules['skip_if_matches']) if rules.get('skip_if_matches') else None

    def check(self, event):
        if event.get('event_type') != 'PreToolUse':
            raise ValueError('unexpected hook event')
        name, args = event.get('tool_name'), event.get('tool_input')
        if name not in {'terminal', 'file_editor', 'task_tracker', 'finish', 'think'} or not isinstance(args, dict):
            raise ValueError('unsupported tool/schema; review enforcement first')
        # As in 1.6, do not scan file bodies or the user/system prompt.
        if name != 'terminal':
            return None
        if not isinstance(args.get('command'), str):
            raise ValueError('terminal command missing')
        for field in ('command', 'code', 'url', 'browser_actions'):
            text = args.get(field)
            if not isinstance(text, str):
                continue
            for pattern in self.unsplit:
                if pattern.search(text):
                    return pattern.pattern
            for part in (self.split.split(text) if self.split else [text]):
                part = part.strip()
                if not part or (self.skip and self.skip.search(part)):
                    continue
                for pattern in self.patterns:
                    if pattern.search(part):
                        return pattern.pattern
        return None

def main():
    try:
        rules = json.loads(Path(sys.argv[1]).read_text())
        event = json.loads(sys.stdin.read(2 * 1024 * 1024))
        hit = Guard(rules).check(event)
        decision = 'deny' if hit else 'allow'
        record = {'task': rules['task'], 'decision': decision, 'pattern': hit,
                  'tool': event['tool_name'], 'session_id': event.get('session_id')}
        if hit:
            record['tool_input'] = event.get('tool_input')
        with open(sys.argv[2], 'a') as log:
            log.write(json.dumps(record) + '\n')
        print(json.dumps({'decision': decision, 'reason':
              'Target source retrieval is prohibited; implement from start.md.' if hit else 'legacy blocklist passed'}))
        return 2 if hit else 0
    except Exception as exc:
        print(json.dumps({'decision': 'deny', 'reason': 'Blocklist unavailable: ' + str(exc)}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
