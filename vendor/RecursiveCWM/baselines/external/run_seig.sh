#!/bin/bash
# SEIG (our reproduction of "Thinking in Blender", no official code): the reproduction's harness is
# untouched; codex only plays the VLM role the paper describes.
# Env: RCWM_SEIG = reproduction checkout (has seig/ and CODEX_BRIEF.md); RCWM_ROOT = runtime root (its .venv runs seig.cli)
set -uo pipefail
SC="$1"; REF="${2:?reference png}"
SEIG="${RCWM_SEIG:?set RCWM_SEIG (path of the SEIG reproduction)}"
ROOT="${RCWM_ROOT:?set RCWM_ROOT (runtime root, see docs/environment.md)}"
cd "$SEIG"
RUN="runs/${RCWM_RUN_PREFIX:-pilot}-$SC"
[ -d "$RUN" ] || "${SEIG_PY:-$ROOT/.venv/bin/python}" -m seig.cli create --run "$RUN" --reference "$REF"
sed "s#runs/school-codex#$RUN#g" CODEX_BRIEF.md > "$RUN/brief.md"
nohup codex exec --cd "$SEIG" --skip-git-repo-check --sandbox workspace-write \
  -c sandbox_workspace_write.network_access=true \
  "$(cat "$RUN/brief.md")" < /dev/null > "$RUN/codex-run.log" 2>&1 &
echo "seig $SC pid $!"
