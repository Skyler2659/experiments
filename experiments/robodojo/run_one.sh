#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo 'Usage: bash experiments/robodojo/run_one.sh STRATEGY TASK SAMPLE [GPU=0]' >&2
  exit 2
fi
cd "$ROOT"
exec "$ROOT/.venv-benchmark/bin/python" -m robodojo_benchmark run \
  --campaign "${ROBODOJO_CAMPAIGN:-$ROOT/campaigns/main}" \
  --runtime "${ROBODOJO_RUNTIME:-$ROOT/experiments/robodojo/runtime.local.json}" \
  --strategy "$1" --task "$2" --sample "$3" --gpu "${4:-0}"
