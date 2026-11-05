#!/bin/bash
# new_workspace.sh <workspace-dir> [shared-runtime-root]
# Creates an isolated workspace for one scene: its own runs/ and (with RCWM_CLEAN_CODEX_HOME=1) its own codex home,
# while .venv and .render-tools are symlinks to a runtime built once by setup/setup_runtime.sh. Sessions of different
# scenes then never see each other's files. Use it as RCWM_ROOT.
#   tools/new_workspace.sh /work/rcwm/medieval /opt/rcwm-runtime
#   RCWM_ROOT=/work/rcwm/medieval RCWM_CLEAN_CODEX_HOME=1 ./rcwm.sh refs/medieval-village.png medieval-village
set -euo pipefail
WS="${1:?usage: new_workspace.sh <workspace-dir> [shared-runtime-root]}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
RT="${2:-${RCWM_RUNTIME:-$CODE/runtime}}"
[ -x "$RT/.venv/bin/python" ] && [ -x "$RT/.render-tools/node/bin/node" ] || { echo "no runtime at $RT (run setup/setup_runtime.sh $RT first)"; exit 1; }
mkdir -p "$WS/runs"; WS="$(cd "$WS" && pwd)"; RT="$(cd "$RT" && pwd)"
ln -sfn "$RT/.venv" "$WS/.venv"; ln -sfn "$RT/.render-tools" "$WS/.render-tools"
echo "$WS"
