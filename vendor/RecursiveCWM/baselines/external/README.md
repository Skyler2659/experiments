# External baselines

Three image-to-scene-program baselines, each run on the same base model as ours and given the same raw reference image.

- **SEIG** — "Thinking in Blender" has no released implementation; `run_seig.sh` drives our from-paper reproduction (a separate checkout, `RCWM_SEIG`), with codex playing only the VLM role the paper describes.
- **VIGA** — the official runner (`runners/static_scene.py`) with only the model endpoint swapped to a codex shim (`run_viga.sh`, one scene per launch via `--task`). Run without VIGA's commercial asset service, so its assets are built procedurally in Blender by its own agents.
- **img2threejs** — its released pipeline, in an isolated working directory per scene that contains only the img2threejs release, the reference image, and the tool locations (`run_img2threejs.sh`). Stage 1 runs it natively; if its own gate or correction limit stops it, stage 2 resumes the same session with a single instruction to continue and deliver a render, with no other help. Both stages are logged in the run directory.

Every method receives the reference image unmodified (all figure furniture kept). Metrics are computed by `experiments/eval_metrics.py` and are record-only.
