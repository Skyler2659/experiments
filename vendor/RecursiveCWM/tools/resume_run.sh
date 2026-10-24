#!/bin/bash
# resume_run.sh <run-name> : continue an interrupted run in place (quota outage, machine restart, killed shell).
# The root runner is re-invoked; every node continues its own codex session (fractal/<node>/.sid), and a node whose
# children never delivered relaunches only those children (trace event "recover_children"). Nothing is redone.
# Env: RCWM_ROOT (workspace that holds runs/<run-name>), and the same RCWM_PROMPT / RCWM_MAXD / RCWM_MAXCYC
# the run was started with (the trace records the instruction hash; a different instruction would change the run).
# The run's private codex home ($RCWM_ROOT/.codex-home, the default of rcwm.sh) is reused with its login file refreshed
# from ~/.codex; a run started with RCWM_CLEAN_CODEX_HOME=0 has no such directory and keeps using the machine's home.
set -uo pipefail
NAME="${1:?usage: resume_run.sh <run-name>}"
export RCWM_CODE="$(cd "$(dirname "$0")/.." && pwd)"
export RCWM_ROOT="${RCWM_ROOT:-$RCWM_CODE/runtime}"
export RCWM_MAXD="${RCWM_MAXD:-4}" RCWM_MAXCYC="${RCWM_MAXCYC:-3}"
export RCWM_PROMPT="${RCWM_PROMPT:-$RCWM_CODE/solver/solver-template.md}"
R="$RCWM_ROOT/runs/$NAME"
[ -d "$R/fractal/scene" ] || { echo "no run at $R"; exit 1; }
[ -f "$R/fractal/scene/part.json" ] && { echo "$NAME already delivered: $R/fractal/scene/part.json"; exit 0; }
if pgrep -f "^bash .*solve_recursive[.]sh runs/$NAME " >/dev/null; then echo "$NAME is still running"; exit 1; fi
if [ -d "$RCWM_ROOT/.codex-home" ]; then
  SRC="${CODEX_HOME:-$HOME/.codex}"
  [ -f "$SRC/auth.json" ] && cp "$SRC/auth.json" "$RCWM_ROOT/.codex-home/auth.json" && chmod 600 "$RCWM_ROOT/.codex-home/auth.json"
  export CODEX_HOME="$RCWM_ROOT/.codex-home"
fi
echo "resuming $R  (instruction $RCWM_PROMPT, depth $RCWM_MAXD, cycles $RCWM_MAXCYC${CODEX_HOME:+, codex home $CODEX_HOME})"
bash "$RCWM_CODE/runner/solve_recursive.sh" "runs/$NAME" scene - 0 "$NAME"
python3 "$RCWM_CODE/runner/trace_report.py" "$R" 2>/dev/null | tail -3
[ -f "$R/fractal/scene/part.json" ] && echo "delivered: $R/fractal/scene/part.json" || echo "still not delivered — see $R/fractal/scene/codex-run.log"
