#!/bin/bash
# VIGA native baseline: the official runner + the official codex shim, same base model and reasoning effort as ours.
# Start the shim first (from the VIGA checkout):  .venv-agent/bin/python codex_openai_shim.py --port 8102 --reasoning-effort high --max-images 10
# Env: RCWM_VIGA = VIGA checkout (data/pilot/<scene>/target.png prepared); BLENDER = blender binary; VIGA_SHIM_PORT (default 8102)
# Usage: run_viga.sh <scene>   -> output/static_scene/<RCWM_RUN_PREFIX or pilot2>-<scene>/<scene>/renders/<round>/*.png
set -uo pipefail
SC="$1"; PORT="${VIGA_SHIM_PORT:-8102}"; TID="${RCWM_RUN_PREFIX:-pilot2}-$SC"
VIGA="${RCWM_VIGA:?set RCWM_VIGA (path of the VIGA checkout)}"
BLENDER="${BLENDER:?set BLENDER (path of the blender binary VIGA should use)}"
cd "$VIGA"
export OPENAI_BASE_URL=http://127.0.0.1:$PORT/v1 OPENAI_API_KEY=not_used
nohup ./.venv/bin/python runners/static_scene.py \
  --dataset-path data/pilot --task "$SC" --test-id "$TID" \
  --output-dir runs/"$TID" \
  --model gpt-5.6-sol \
  --blender-command "$BLENDER" \
  --max-rounds 20 \
  > runs/"$TID".log 2>&1 &
echo "viga $SC pid $!"
