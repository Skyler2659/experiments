#!/bin/bash
# Unified recursive solver runner: solve(node) can spawn solve(child); the runner records events.
# Usage: solve_recursive.sh <chain> <node> <parent_id|-> <depth> <run_id>
set -uo pipefail
# RCWM_CODE = this repo; RCWM_ROOT = runtime root (see docs/environment.md), default <repo>/runtime
CODE="${RCWM_CODE:-$(cd "$(dirname "$0")/.." && pwd)}"
ROOT="${RCWM_ROOT:-$CODE/runtime}"
cd "$ROOT"
CHAIN="$1"; NODE="$2"; PARENT="${3:--}"; DEPTH="${4:-0}"; RUN="${5:-r0}"
MAXD="${RCWM_MAXD:-4}"; MAXCYC="${RCWM_MAXCYC:-3}"
PROMPT="${RCWM_PROMPT:-$CODE/solver/solver-template.md}"
MODEL="${RCWM_MODEL:-gpt-6-astra}"; EFFORT="${RCWM_REASONING:-high}"   # the paper's model and reasoning effort; passed explicitly so codex's own config cannot change them
PWCACHE="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"   # headless Chromium lives here; codex needs write access to it
ND="$CHAIN/fractal/$NODE"
TR="$CHAIN/trace"; mkdir -p "$TR" "$ND"
EV="$TR/events.jsonl"
TPLH=$(sha256sum "$PROMPT" | cut -c1-12)   # hash of the instruction actually used (RCWM_PROMPT or the default)
CAMH=$([ -f "$CHAIN/camera-contract.json" ] && sha256sum "$CHAIN/camera-contract.json" | cut -c1-12 || echo none)
SNAP=$(git rev-parse --short HEAD 2>/dev/null || echo none)   # runtime root need not be a git repo

log_ev() { # type cycle extra_json
  printf '%s
' "{\"run_id\":\"$RUN\",\"node_id\":\"$NODE\",\"parent_id\":\"$PARENT\",\"depth\":$DEPTH,\"cycle\":$2,\"event\":\"$1\",\"ts\":\"$(date -Is)\",\"solver_hash\":\"$TPLH\",\"camera_hash\":\"$CAMH\",\"parent_snapshot\":\"$SNAP\"$3}" >> "$EV"
}

usage_delta() { # Most recent tokens used (codex prints the number on the same line or the next).
  grep -a -A1 "tokens used" "$ND/codex-run.log" 2>/dev/null | grep -o "[0-9][0-9,]*" | tail -1 | tr -d ','
}

# manifest: interface version (inherited context)
[ -f "$ND/manifest.json" ] || cat > "$ND/manifest.json" <<M
{"interface_version":1,"reference_sha256":"$( [ -f "$ND/target.png" ] && sha256sum "$ND/target.png" | cut -c1-12 || echo pending)","camera_hash":"$CAMH","parent_snapshot":"$SNAP","solver_hash":"$TPLH"}
M

python3 -c 'import json,sys;json.dump({"node_id":sys.argv[2],"parent_id":sys.argv[3],"depth":int(sys.argv[4]),"max_depth":int(sys.argv[5])},open(sys.argv[1],"w"))' "$ND/node_context.json" "$NODE" "$PARENT" "$DEPTH" "$MAXD"
# A measured adaptive run can reuse a stable object before spending another model call.
if [ -n "${RCWM_FEEDBACK_HOOK:-}" ] && [ "${RCWM_ADAPTIVE_DISPATCH:-0}" = "1" ]; then
  "$RCWM_FEEDBACK_HOOK" "$CHAIN" "$NODE" 0 dispatch; HC=$?
  if [ "$HC" -eq 11 ]; then log_ev "reuse" 0 ',"stop_reason":"adaptive_scope_stopped"'; exit 0; fi
  if [ "$HC" -ne 0 ]; then exit "$HC"; fi
fi
# Keep a node's cycle budget across targeted revisits from its parent.
CYC=$(python3 -c 'import json,sys; rows=[json.loads(x) for x in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []; print(1+sum(r.get("event")=="session_start" and r.get("node_id")==sys.argv[2] for r in rows))' "$EV" "$NODE")
# Diagnosis adds a separately recorded, bounded allowance; base cycles stay cumulative.
BASECYC="$MAXCYC"
correction_cycles() {
  if [ -n "${RCWM_FEEDBACK_HOOK:-}" ]; then
    EXTRA=$(python3 -c 'import json,pathlib,sys;p=pathlib.Path(sys.argv[1]);print(json.loads(p.read_text()).get("extra_cycles",0) if p.exists() else 0)' "$ND/correction_state.json")
    MAXCYC=$((BASECYC+EXTRA))
  fi
}
correction_cycles
if [ "$CYC" -gt "$MAXCYC" ]; then log_ev "stop" "$CYC" ',"stop_reason":"node_cycle_budget_exhausted"'; exit 22; fi
RECOVERY_CHECKED=0
while [ "$CYC" -le "$MAXCYC" ]; do
  rm -f "$ND/children.json.consumed"
  [ -f "$ND/children.json" ] && mv "$ND/children.json" "$ND/children.json.prev"
  # Interruption recovery: a fresh invocation of a node that already has a session, no deliverable, and a
  # previous children request with undelivered children re-launches just those children (they resume by
  # their own .sid) instead of waking this node with an incomplete set.
  RECOVER=0
  if [ "$RECOVERY_CHECKED" = "0" ] && [ -f "$ND/.sid" ] && [ ! -f "$ND/part.json" ] && [ -f "$ND/children.json.prev" ] && [ "$DEPTH" -lt "$MAXD" ]; then
    MISSING=$(python3 -c "import json,os; ks=json.load(open('$ND/children.json.prev')); print(json.dumps([k for k in ks if not os.path.exists('$CHAIN/fractal/'+k+'/part.json') or (os.environ.get('RCWM_FEEDBACK_HOOK') and not os.path.exists('$CHAIN/fractal/'+k+'/selection.json'))]))" 2>/dev/null)
    if [ -n "$MISSING" ] && [ "$MISSING" != "[]" ]; then echo "$MISSING" > "$ND/children.json"; RECOVER=1; log_ev "recover_children" "$CYC" ",\"children\":$MISSING"; fi
  fi
  RECOVERY_CHECKED=1
  if [ "$RECOVER" = "1" ]; then :; else
  log_ev "session_start" "$CYC" ",\"model\":\"$MODEL\",\"reasoning\":\"$EFFORT\""
  sed "s#__NODE__#$NODE#g; s#__CHAIN__#$CHAIN#g; s#__DEPTH__#$DEPTH#g" \
      "$PROMPT" > "$ND/task.md"
  # At the maximum depth the runner will not launch children; the node must know, or it keeps writing children.json and never delivers.
  if [ "$DEPTH" -ge "$MAXD" ]; then
    MAXNOTE="This level is at the maximum depth: the runner will not launch children, so do not write children.json; finish this level yourself and deliver part.json."
    printf '\n%s\n' "$MAXNOTE" >> "$ND/task.md"
  else MAXNOTE=""; fi
  if [ -f "$ND/.sid" ] && grep -q "[0-9a-f]" "$ND/.sid"; then
    codex exec --skip-git-repo-check --cd "$ROOT" --sandbox "${RCWM_SANDBOX:-workspace-write}" \
      -c "model=\"$MODEL\"" -c "model_reasoning_effort=\"$EFFORT\"" \
      -c sandbox_workspace_write.network_access=true \
      -c "sandbox_workspace_write.writable_roots=[\"$PWCACHE\"]" \
      resume "$(cat "$ND/.sid")" "Continue this level: read your brief.md updates and, if present, the parent node $CHAIN/fractal/$PARENT/refinement_requests.json entry for $NODE. $( [ -s "$ND/.children_done" ] && echo " children delivered:$(cat "$ND/.children_done"). Do the whole-again step: integrate the children, settle continuity and relations;" ) $( [ -f "$ND/feedback.md" ] && echo "FIRST read $ND/feedback.md and, if present, $ND/correction_instructions.md; follow the numerical/visual feedback within your assigned scope." ) proceed per the protocol to write this level's part.json + account.md and finish; if more descent is needed, write a new children.json and end.${MAXNOTE:+ $MAXNOTE}" \
      < /dev/null >> "$ND/codex-run.log" 2>&1
    CODEX_RC=$?
  else
    codex exec --skip-git-repo-check --cd "$ROOT" --sandbox "${RCWM_SANDBOX:-workspace-write}" \
      -c "model=\"$MODEL\"" -c "model_reasoning_effort=\"$EFFORT\"" \
      -c sandbox_workspace_write.network_access=true \
      -c "sandbox_workspace_write.writable_roots=[\"$PWCACHE\"]" \
      "$(cat "$ND/task.md")" < /dev/null >> "$ND/codex-run.log" 2>&1
    CODEX_RC=$?
    grep -m1 "session id" "$ND/codex-run.log" | awk '{print $3}' > "$ND/.sid"
  fi
  if [ "$CODEX_RC" -ne 0 ]; then
    log_ev "session_error" "$CYC" ",\"returncode\":$CODEX_RC,\"stop_reason\":\"codex_execution_failed\""
    exit "$CODEX_RC"
  fi
  U=$(usage_delta); U=${U:-0}
  log_ev "session_end" "$CYC" ",\"usage_total\":$U,\"returncode\":0"
  fi   # end of the codex call skipped by recovery

  # Optional external measured-fit service. Default RCWM behavior is unchanged.
  if [ -n "${RCWM_FEEDBACK_HOOK:-}" ] && [ -f "$ND/fit_requests.json" ]; then
    "$RCWM_FEEDBACK_HOOK" "$CHAIN" "$NODE" "$CYC"; HC=$?
    log_ev "fit_service" "$CYC" ",\"returncode\":$HC"
    if [ "$HC" -eq 10 ]; then correction_cycles; CYC=$((CYC+1)); continue; fi
    if [ "$HC" -ne 0 ]; then exit "$HC"; fi
  fi

  # Children requested?
  KIDS=""
  if [ -f "$ND/children.json" ]; then
    KIDS=$(python3 "$CODE/runner/check_children.py" "$ND/children.json" "$NODE" "$PARENT" 2> "$ND/children_validation.txt"); VC=$?
    if [ "$VC" -ne 0 ]; then
      printf 'PROGRAM_ERROR: correct your children.json format before continuing.\n' > "$ND/feedback.md"
      cat "$ND/children_validation.txt" >> "$ND/feedback.md"
      mv "$ND/children.json" "$ND/children.invalid.$CYC.json"
      log_ev "children_protocol_error" "$CYC" ',"returncode":10'
      CYC=$((CYC+1)); continue
    fi
  fi
  if [ -n "$KIDS" ] && [ "$DEPTH" -lt "$MAXD" ]; then   # an empty list means "no children": finish, don't resume
    if [ -n "${RCWM_FEEDBACK_HOOK:-}" ] && [ "${RCWM_ADAPTIVE_DISPATCH:-0}" = "1" ]; then
      "$RCWM_FEEDBACK_HOOK" "$CHAIN" "$NODE" "$CYC" delegate; HC=$?
      if [ "$HC" -eq 10 ]; then log_ev "adaptive_descent_blocked" "$CYC" ',"returncode":10'; CYC=$((CYC+1)); continue; fi
      if [ "$HC" -ne 0 ]; then exit "$HC"; fi
    fi
    PIDS=""
    for K in $KIDS; do
      log_ev "child_call" "$CYC" ",\"child\":\"$K\""
      bash "$CODE"/runner/solve_recursive.sh "$CHAIN" "$K" "$NODE" "$((DEPTH+1))" "$RUN" &
      PIDS="$PIDS $!"
    done
    CHILD_ERROR=0
    for P in $PIDS; do wait "$P" || CHILD_ERROR=$?; done
    DONE=""
    for K in $KIDS; do
      [ -f "$CHAIN/fractal/$K/part.json" ] && DONE="$DONE $K"
      log_ev "child_return" "$CYC" ",\"child\":\"$K\",\"delivered\":$([ -f "$CHAIN/fractal/$K/part.json" ] && echo true || echo false)"
    done
    echo "$DONE" > "$ND/.children_done"
    if [ "$CHILD_ERROR" -ne 0 ]; then
      log_ev "child_service_error" "$CYC" ",\"returncode\":$CHILD_ERROR"
      exit "$CHILD_ERROR"
    fi
    CYC=$((CYC+1))
    continue
  fi
  if [ -n "${RCWM_FEEDBACK_HOOK:-}" ] && [ -f "$ND/part.json" ]; then
    "$RCWM_FEEDBACK_HOOK" "$CHAIN" "$NODE" "$CYC"; HC=$?
    log_ev "geometry_feedback" "$CYC" ",\"returncode\":$HC"
    if [ "$HC" -eq 10 ]; then correction_cycles; CYC=$((CYC+1)); continue; fi
    if [ "$HC" -ne 0 ]; then exit "$HC"; fi
  fi
  if [ ! -f "$ND/part.json" ]; then
    printf 'PROGRAM_ERROR: no part.json was delivered in your actual flat node directory %s. Read node_context.json; write fit_requests.json, children.json, or the final component.js + part.json in THIS directory. Files written under a nested parent directory are not a delivery for this runner node.
' "$ND" > "$ND/feedback.md"
    log_ev "delivery_protocol_error" "$CYC" ',"returncode":10'
    CYC=$((CYC+1)); continue
  fi
  break
done

# Runner-level delivery validation (only machine-checkable facts, no procedural formalities).
REASON="visual_stop"
if [ -f "$ND/part.json" ]; then
  python3 "$CODE"/runner/check_part.py "$ND/part.json" || REASON="invalid_artifact"
else
  REASON="no_artifact"
fi
log_ev "stop" "$CYC" ",\"stop_reason\":\"$REASON\""
if [ -f "$ND/refinement_pending.json" ]; then exit 21; fi
