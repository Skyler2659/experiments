#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$(uname -s)" != Linux ]; then
  echo 'RCWM official runtime requires Linux. Copy RCWMWorld to Linux and rerun this command.' >&2
  exit 1
fi
export RCWM_ROOT="${RCWM_ROOT:-$ROOT/runtime}"
# Chromium resolves its bundled Vulkan loader incorrectly on the l203 /private
# network filesystem. Keep its binaries on the server's local disk.
if [ ! -e "$RCWM_ROOT/.browsers" ] && [ ! -L "$RCWM_ROOT/.browsers" ]; then
  RCWM_BROWSER_CACHE="${RCWM_BROWSER_CACHE:-$HOME/.cache/rcwmworld-browsers}"
  mkdir -p "$RCWM_ROOT" "$RCWM_BROWSER_CACHE"
  ln -s "$RCWM_BROWSER_CACHE" "$RCWM_ROOT/.browsers"
fi
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$RCWM_ROOT/.browsers}"
export PLAYWRIGHT_BROWSERS_PATH="$(cd "$PLAYWRIGHT_BROWSERS_PATH" && pwd -P)"
if [ -d "$RCWM_ROOT/.system-libs/usr/lib/x86_64-linux-gnu" ]; then
  export LD_LIBRARY_PATH="$RCWM_ROOT/.system-libs/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
fi
# The upstream numpy 2.5.2 pin requires 3.12 despite its wider README range.
if [ ! -x "$RCWM_ROOT/.venv/bin/python" ]; then
  RCWM_PYTHON="${RCWM_PYTHON:-python3.12}"
  command -v "$RCWM_PYTHON" >/dev/null || { echo 'Set RCWM_PYTHON to a Python 3.12 interpreter.' >&2; exit 1; }
  "$RCWM_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12), "Upstream numpy pin requires Python 3.12"'
  mkdir -p "$RCWM_ROOT"
  "$RCWM_PYTHON" -m venv "$RCWM_ROOT/.venv"
fi
"$RCWM_ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Existing runtime must use Python 3.12"'
bash "$ROOT/vendor/RecursiveCWM/setup/setup_runtime.sh" "$RCWM_ROOT" "$@"
export PATH="$RCWM_ROOT/.codex-cli/node_modules/.bin:$RCWM_ROOT/.render-tools/node/bin:$PATH"
if ! command -v codex >/dev/null; then
  npm install --prefix "$RCWM_ROOT/.codex-cli" @openai/codex@0.154.0
fi
echo "Runtime ready. CLI fallback: $RCWM_ROOT/.codex-cli/node_modules/.bin/codex"
