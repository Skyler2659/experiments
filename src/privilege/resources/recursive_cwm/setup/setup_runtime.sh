#!/bin/bash
# setup_runtime.sh [RCWM_ROOT] [--metrics] [--python <interpreter>] [--conda <env-name>] [--node-from <dir>]
# Builds the runtime root the solver needs (docs/environment.md) on a Linux x86_64 machine. Nothing is installed into
# the Python environment you are in: the packages go to a private venv (default) or to a dedicated conda env.
#   $RCWM_ROOT/.venv                    Python 3 (3.10-3.12) with Pillow + numpy   (+ metrics packages with --metrics)
#                                       default: a venv created from --python (or python3); with --conda <name>: a
#                                       symlink to the conda env <name> (created with python=3.12 if it does not exist)
#   $RCWM_ROOT/.render-tools/node       Node.js 22 (downloaded from nodejs.org, or symlinked with --node-from)
#   $RCWM_ROOT/.render-tools/node_modules/{three,playwright}   three.js 0.160.1 + Playwright 1.62.1
#   headless Chromium for Playwright    in $PLAYWRIGHT_BROWSERS_PATH (default ~/.cache/ms-playwright)
#   $RCWM_ROOT/runs                     one directory per run
# Re-running is safe: every step skips what already exists. Needs: python3 (with venv), curl, tar, network.
set -euo pipefail
CODE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${CODE}/runtime"; METRICS=0; NODE_FROM=""; CONDA_ENV=""
PY="${PYTHON:-python3}"
while [ $# -gt 0 ]; do
  case "$1" in
    --metrics) METRICS=1 ;;
    --python) PY="$2"; shift ;;
    --conda) CONDA_ENV="$2"; shift ;;
    --node-from) NODE_FROM="$2"; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) ROOT="$1" ;;
  esac; shift
done
ROOT="$(mkdir -p "$ROOT" && cd "$ROOT" && pwd)"
NODE_VERSION="${RCWM_NODE_VERSION:-22.14.0}"
echo "runtime root: $ROOT"

# 1. Python: a private venv, or a dedicated conda env linked in as .venv (same layout either way)
if [ -n "$CONDA_ENV" ]; then
  CONDA="${RCWM_CONDA:-$(command -v conda || command -v mamba || command -v micromamba || true)}"
  [ -n "$CONDA" ] || { echo "--conda needs conda, mamba or micromamba on PATH (or RCWM_CONDA=/path/to/it)"; exit 1; }
  if ! "$CONDA" run -n "$CONDA_ENV" python --version >/dev/null 2>&1; then
    echo "creating conda env $CONDA_ENV (python=3.12) ..."
    "$CONDA" create -y -q -n "$CONDA_ENV" python=3.12 >/dev/null
  fi
  PREFIX="$("$CONDA" run -n "$CONDA_ENV" python -c 'import sys; print(sys.prefix)' | tail -1)"
  [ -x "$PREFIX/bin/python" ] || { echo "conda env $CONDA_ENV has no bin/python at $PREFIX"; exit 1; }
  [ -e "$ROOT/.venv" ] && [ ! -L "$ROOT/.venv" ] && { echo "$ROOT/.venv exists and is not a symlink; remove it to switch to conda"; exit 1; }
  ln -sfn "$PREFIX" "$ROOT/.venv"
  echo "python:  conda env $CONDA_ENV ($PREFIX) linked as $ROOT/.venv"
elif [ ! -x "$ROOT/.venv/bin/python" ]; then
  "$PY" -m venv "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" - <<'P'
import sys; v=sys.version_info
assert (3,10) <= (v.major,v.minor) <= (3,12), f"Python 3.10-3.12 required, got {v.major}.{v.minor} (pass --python /path/to/python3.12 or --conda <env>)"
P
"$ROOT/.venv/bin/python" -m pip install --quiet --upgrade pip
"$ROOT/.venv/bin/python" -m pip install --quiet -r "$CODE/setup/requirements-runtime.txt"
[ "$METRICS" = 1 ] && "$ROOT/.venv/bin/python" -m pip install --quiet -r "$CODE/setup/requirements-metrics.txt"
echo "python:  $("$ROOT/.venv/bin/python" --version)  (metrics packages: $([ "$METRICS" = 1 ] && echo yes || echo no))"

# 2. Node.js
mkdir -p "$ROOT/.render-tools"
if [ ! -x "$ROOT/.render-tools/node/bin/node" ]; then
  if [ -n "$NODE_FROM" ]; then
    [ -x "$NODE_FROM/bin/node" ] || { echo "--node-from $NODE_FROM has no bin/node"; exit 1; }
    ln -sfn "$NODE_FROM" "$ROOT/.render-tools/node"
  else
    ARCH=$(uname -m); case "$ARCH" in x86_64) NA=x64 ;; aarch64|arm64) NA=arm64 ;; *) echo "unsupported arch $ARCH"; exit 1 ;; esac
    TB="node-v${NODE_VERSION}-linux-${NA}.tar.xz"
    echo "downloading Node.js $NODE_VERSION ..."
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/${TB}" -o "$ROOT/.render-tools/$TB"
    mkdir -p "$ROOT/.render-tools/node"
    tar -xJf "$ROOT/.render-tools/$TB" -C "$ROOT/.render-tools/node" --strip-components=1
    rm -f "$ROOT/.render-tools/$TB"
  fi
fi
export PATH="$ROOT/.render-tools/node/bin:$PATH"
echo "node:    $(node --version)"

# 3. three.js + Playwright (exact versions the paper's runs used)
cd "$ROOT/.render-tools"
[ -f package.json ] || cat > package.json <<'J'
{"name":"rcwm-render-tools","private":true,"version":"1.0.0","dependencies":{"playwright":"1.62.1","three":"0.160.1"}}
J
if [ ! -f node_modules/three/build/three.module.js ] || [ ! -f node_modules/playwright/index.mjs ]; then
  npm install --no-audit --no-fund --loglevel=error
fi
echo "three:   $(node -p "require('./node_modules/three/package.json').version")"
echo "playwright: $(node -p "require('./node_modules/playwright/package.json').version")"

# 4. headless Chromium
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
mkdir -p "$PLAYWRIGHT_BROWSERS_PATH"
if ! ls -d "$PLAYWRIGHT_BROWSERS_PATH"/chromium-* >/dev/null 2>&1; then
  npx --no-install playwright install chromium
fi
echo "chromium: $(ls -d "$PLAYWRIGHT_BROWSERS_PATH"/chromium-* | head -1)"
# Shared libraries Chromium needs (libnss3, libatk, libgbm, ...): if the smoke test below fails to launch the browser,
# run once with root:  sudo npx playwright install-deps chromium      (in $ROOT/.render-tools)

# 5. run layout + smoke test
mkdir -p "$ROOT/runs"
node "$CODE/setup/smoke_test.mjs" "$ROOT"
echo
echo "done. Use it with:   export RCWM_ROOT=$ROOT"
[ -n "${PLAYWRIGHT_BROWSERS_PATH:-}" ] && [ "$PLAYWRIGHT_BROWSERS_PATH" != "$HOME/.cache/ms-playwright" ] && echo "                     export PLAYWRIGHT_BROWSERS_PATH=$PLAYWRIGHT_BROWSERS_PATH"
exit 0
