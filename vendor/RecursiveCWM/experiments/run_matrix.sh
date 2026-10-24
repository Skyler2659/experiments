#!/bin/bash
# Ablation matrix driver: scenes × variants × reps through the same runner; limited concurrency; one row per run in results.csv.
# Usage: run_matrix.sh "<scenes>" "<variants>" [reps=1] [concurrency=4]
#   variants: flat | localglobal | globallocal | twolevel | recursive   (instruction + depth cap per variant, tables VP/VD below)
#   references: $RCWM_REFS/<scene>.png (default experiments/pilot-scenes); runs: $RCWM_ROOT/runs/pilot/<scene>-<variant>-r<rep>
#   concurrency counts every "codex exec" process on the machine, not only this matrix's.
set -uo pipefail
CODE="${RCWM_CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${RCWM_ROOT:-$CODE/runtime}"; REFS="${RCWM_REFS:-$CODE/experiments/pilot-scenes}"
SCENES="${1:?usage: run_matrix.sh \"<scenes>\" \"<variants>\" [reps] [concurrency]}"; VARIANTS="${2:?variants}"; REPS="${3:-1}"; CONC="${4:-4}"
RES="$ROOT/runs/pilot/results.csv"
mkdir -p "$ROOT/runs/pilot"
[ -f "$RES" ] || echo "scene,variant,rep,chain,start,end,stop_reason,max_depth,sessions,usage_last,part_delivered" > "$RES"
declare -A VP=( [flat]="$CODE/baselines/variants/flat-zoom.md" \
                [localglobal]="$CODE/baselines/variants/local-global.md" \
                [globallocal]="$CODE/baselines/variants/global-local.md" \
                [twolevel]="$CODE/solver/solver-template.md" \
                [recursive]="$CODE/solver/solver-template.md" )
declare -A VD=( [flat]=0 [localglobal]=1 [globallocal]=0 [twolevel]=1 [recursive]=4 )
for SC in $SCENES; do
 for V in $VARIANTS; do
  for R in $(seq 1 "$REPS"); do
    CH="runs/pilot/${SC}-${V}-r${R}"
    [ -f "$ROOT/$CH/fractal/scene/part.json" ] && continue
    while [ "$(pgrep -fc 'codex exec' || true)" -ge "$CONC" ]; do sleep 120; done
    mkdir -p "$ROOT/$CH/fractal/scene" "$ROOT/$CH/trace"
    [ -f "$REFS/$SC.png" ] || { echo "no reference $REFS/$SC.png (set RCWM_REFS)"; continue; }
    cp "$REFS/$SC.png" "$ROOT/$CH/fractal/scene/target.png"
    echo '{"note":"root chooses its own framing"}' > "$ROOT/$CH/fractal/scene/view.json"
    cat > "$ROOT/$CH/fractal/scene/brief.md" <<B
Rebuild everything visible in target.png as a parameterized Three.js scene program
(first solve and calibrate a camera; verify the full-frame overlay by eye before locking it).
Rendering environment: $CODE/docs/environment.md (RCWM_ROOT=$ROOT).
B
    ST=$(date -Is)
    ( RCWM_PROMPT="${VP[$V]}" RCWM_MAXD="${VD[$V]}" \
        bash "$CODE/runner/solve_recursive.sh" "$CH" scene - 0 "${SC}-${V}-r${R}"
      EN=$(date -Is)
      D=$ROOT/$CH/fractal/scene
      MAXDPT=$(grep -o '"depth":[0-9]*' "$ROOT/$CH/trace/events.jsonl" 2>/dev/null | cut -d: -f2 | sort -n | tail -1)
      SESS=$(grep -c '"event":"session_end"' "$ROOT/$CH/trace/events.jsonl" 2>/dev/null || echo 0)
      USE=$(grep -o '"usage_total":[0-9]*' "$ROOT/$CH/trace/events.jsonl" 2>/dev/null | cut -d: -f2 | tail -1)
      SR=$(grep -o '"stop_reason":"[a-z_]*"' "$ROOT/$CH/trace/events.jsonl" 2>/dev/null | tail -1 | cut -d'"' -f4)
      PD=$([ -f "$D/part.json" ] && echo 1 || echo 0)
      flock "$RES" -c "echo '$SC,$V,$R,$CH,$ST,$EN,${SR:-unknown},${MAXDPT:-0},${SESS:-0},${USE:-0},$PD' >> '$RES'" 2>/dev/null \
        || echo "$SC,$V,$R,$CH,$ST,$EN,${SR:-unknown},${MAXDPT:-0},${SESS:-0},${USE:-0},$PD" >> "$RES"
    ) &
    sleep 20
  done
 done
done
wait
echo "MATRIX WAVE DONE $(date)" >> "$ROOT/runs/pilot/matrix.log"
