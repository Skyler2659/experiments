#!/bin/bash
# Native img2threejs baseline with isolated workspaces (user decision, 2026-09-09):
#   Each scene has its own $RCWM_I2T_ISO/<scene>/ directory containing only an img2threejs copy, reference image, and rendering tool locations.
#   codex runs in that directory without access to our runs/; the prompt only says to follow the project's own procedure, with no scene/single-object framing or extra assistance.
#   The only allowed intervention: after a quality-gate rejection or hard stop, resume the same session and require completion (stage two), still without hints.
# Usage: run_img2threejs.sh <scene> <reference.png>
set -uo pipefail
SC="$1"; REF="${2:?reference png}"
ROOT="${RCWM_ROOT:?set RCWM_ROOT (runtime root, see docs/environment.md)}"
ISO="${RCWM_I2T_ISO:?set RCWM_I2T_ISO (directory for the isolated img2threejs runs)}"
D="$ISO/$SC"
if [ -d "$D" ]; then echo "exists: $D (rename it first)"; exit 1; fi
mkdir -p "$D"
cp -r "$ROOT/vendor/img2threejs" "$D/img2threejs"
cp "$REF" "$D/reference.png"
cat > "$D/ENVIRONMENT.md" <<E
# Tools available on this machine
- Node: $ROOT/.render-tools/node/bin
- three: $ROOT/.render-tools/node_modules/three/build/three.module.js
- playwright (headless Chromium): $ROOT/.render-tools/node_modules/playwright
- Python with Pillow/numpy: $ROOT/.venv/bin/python
Work only inside this directory.
E
(
  cd "$D"
  # Stage one: native flow. Preserve the paper's baseline prompt verbatim.
  # Translation: Follow the ./img2threejs project's own procedure (first read img2threejs/CLAUDE.md, SKILL.md, and grimoire/). Reconstruct ./reference.png as a code-only Three.js program, write all outputs in the current directory, and see ./ENVIRONMENT.md for tool locations. Follow the project's own workflow and quality gates; stop when complete.
  codex exec --skip-git-repo-check --cd "$D" --sandbox workspace-write -c sandbox_workspace_write.network_access=true "按 ./img2threejs 项目自己的规程工作(先读 img2threejs/CLAUDE.md、SKILL.md 与 grimoire/),把 ./reference.png 重建为 code-only 的 Three.js 程序,全部产物写在当前目录,工具位置见 ./ENVIRONMENT.md。按该项目自身的流程与质量门走,完成即结束。" \
    < /dev/null > codex-run.log 2>&1
  echo "STAGE1_DONE $(date -Is)" >> stages.log
  if ! ls final-render.png render-hires.png >/dev/null 2>&1; then
    # Stage two: quality-gate rejection/stop -> require completion (without hints).
    SID=$(grep -a -m1 'session id' codex-run.log | awk '{print $3}')
    # Translation: Stopping is not acceptable. Continue regardless, reconstruct ./reference.png as a code-only Three.js program, deliver final-render.png rendered from the reference camera, and stop when complete.
    codex exec --skip-git-repo-check --cd "$D" --sandbox workspace-write -c sandbox_workspace_write.network_access=true resume "$SID" "不接受停止。强行继续,把 ./reference.png 重建为 code-only 的 Three.js 程序,交付从参考相机渲染的 final-render.png,完成即结束。" \
      < /dev/null >> codex-run.log 2>&1
    echo "STAGE2_DONE $(date -Is)" >> stages.log
  fi
  echo "ALL_DONE $(date -Is)" >> stages.log
) > "$D/runner.log" 2>&1 &
echo "img2threejs(iso) $SC pid $! dir $D"
