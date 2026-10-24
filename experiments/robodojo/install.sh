#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Simulator/reconstruction installation requires Linux; the campaign CLI and unit tests also run on macOS.' >&2
  exit 1
fi
command -v uv >/dev/null || { echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }
cd "$ROOT"
exec uv run --no-project --python 3.12 experiments/robodojo/install.py "$@"
