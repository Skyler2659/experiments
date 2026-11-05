#!/bin/bash
# view.sh <run-dir | workspace-root> [port] [host]
# Opens a delivered world for interactive 3D viewing (drag to orbit, right-drag to pan, wheel to zoom, R to reset):
# starts tools/view_server.mjs on the run's workspace and prints the URL to open in a browser. Nothing in the run is
# modified. On a remote machine, forward the port first:  ssh -L 8000:127.0.0.1:8000 <server>   then open the URL locally
# (or pass host 0.0.0.0 to expose it on the network).
set -uo pipefail
D="$(cd "${1:?usage: view.sh <run-dir | workspace-root> [port=8000] [host=127.0.0.1]}" && pwd)"; PORT="${2:-8000}"; HOST="${3:-127.0.0.1}"
CODE="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "$D/fractal/scene" ]; then ROOT="$(cd "$D/../.." && pwd)"; else ROOT="$D"; fi
ROOT="${RCWM_ROOT_OVERRIDE:-$ROOT}"
NODE="$ROOT/.render-tools/node/bin/node"; [ -x "$NODE" ] || NODE=node
exec "$NODE" "$CODE/tools/view_server.mjs" "$ROOT" --port "$PORT" --host "$HOST"
