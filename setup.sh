#!/usr/bin/env bash
# Everything installed from this directory; no parent checkout is needed.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
"${PYTHON:-python3}" -c 'import sys; assert sys.version_info >= (3, 12), "Python >= 3.12 required"'
"${PYTHON:-python3}" -m venv .venv
.venv/bin/python -m pip install 'uv==0.12.15'
.venv/bin/uv pip install --python .venv/bin/python --require-hashes -r requirements.lock.txt
.venv/bin/uv pip install --python .venv/bin/python --no-deps \
  ./vendor/openhands-sdk ./vendor/openhands-tools ./evaluator .
.venv/bin/python -m pip check
if [[ "${1:-}" == --test ]]; then
  .venv/bin/uv pip install --python .venv/bin/python 'pytest==9.1.1' 'pytest-asyncio==1.4.0'
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -p pytest_asyncio.plugin -q -m 'not integration'
fi

