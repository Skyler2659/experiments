# Measured assistance inside RCWM

The recursive model still writes and composes the executable Three.js world. This module adds measured evidence, model-requested generic fitting, and Linux feedback between node turns. Current CodeWorld compilation and robot interfaces are unchanged.

## Flow

1. `evidence.py` reprocesses FIT RGB-D with cached SAM image masks, erodes mask boundaries, and requires two-view depth/identity agreement. It publishes measured samples, visible-envelope initializations, original image crops, and a table plane/footprint. No historical object mesh is loaded.
2. A node can write `fit_requests.json` and yield. `fit_tools.py` fits an upright box, rotational profile (bottle/bowl/plate hypothesis), or tilted slab. The node chooses the representation and consumes the returned numeric recipe. `recipe_geometry.mjs` builds those exact parameters in Three.js.
3. The optional `RCWM_FEEDBACK_HOOK` runs after numerical requests or node delivery. It transfers source to the isolated Linux runtime, builds `component.js` directly, and returns FIT point-to-surface distances plus occlusion-aware depth overlays. It does not depend on the model's viewer camera.
4. The original recursive runner resumes the same model session with `feedback.md`. Each node has at most three geometry reviews; its lowest-score delivered source is retained. The score is object-mean (p95 distance / 10 mm + missing fraction + false-occupied fraction). Budget exhaustion remains review-required. Fit requests use separate runner cycles.
5. Root export and current `codeworld_workspace.build` compile the final program. Only after freezing it does `benchmark.py` consume the four held-out views.

The optional assistance is inactive without `assistance` in the launch config. The example config is `configs/libero_bowl_plate_rcwm_measured_gpt56sol_20261018_v1.json`; transport is in `configs/rcwm_assist_transport.json`. The model and reasoning are explicit: GPT-5.6 Sol / high. The trial increases the node cycle limit to six to allow fit and feedback turns, so it is not an equal-compute ablation of the earlier run.

## Geometry conventions

All delivered node builds are in robot-base W, metres, Z-up. Parents compose them without another pose transform. A semantic object's local geometry may sit under its own pose, but that pose must be applied exactly once. Row-major recipe matrices use column vectors. Sensor unprojection uses integer pixel centers and K directly. WebGL frustum edges use principal point + 0.5 to map those pixel centers to raster cells.

The visible envelope is not a complete shape or measured confidence interval. Rotational symmetry and unseen underside thickness remain hypotheses. Cached masks may contain association errors. Missing/false occupancy scores use observed depth, ignoring a foreground surface that occludes the candidate. RGB textures do not affect those scores. FIT checks do not establish independent reconstruction acceptance.

## Verification

On l203, run `python -m unittest discover -s source/reconstruction_assist/tests`. `tests/check_recipe_roundtrip.py RUNTIME EVIDENCE` compares real Three.js output for a fitted bowl, tilted slab and box against the numerical recipes. Upstream `tests/test_runner_offline.sh` checks the default recursion remains functional.

The final benchmark uses the same object IDs, cached FIT masks, table plane and evaluation cameras for all compared methods. Its held-out ROI is observed depth above the table in common measured workspace bounds, excluding public robot occlusion. It is not independently annotated per-object holdout segmentation. Existing reference views have already been inspected during development; broader claims require new scenes/views and repeated runs.

## Entry point

```bash
python3 reconstruction_assist/run_assisted.py \
  --config configs/libero_bowl_plate_rcwm_measured_gpt56sol_20261018_v1.json \
  --runtime /Users/wangsikai/research/RCWMWorld/runtime-assisted-gpt56sol
```

Use a new scene name, evidence output and remote work directory for a fresh reconstruction. `--reuse-evidence` deliberately reuses the same observation-derived evidence; `--publish-only` publishes an already delivered program without calling the model. The entry point deploys assistance/current CodeWorld code to the configured geometry host, stages FIT inputs, runs the selected model, exports executed geometry, checks a hypothetical CodeWorld edit, and renders the final evaluation cameras. The established l203 Python/Node environment and shared jump-host connection must be available. No robot process is started.

A parent checks assembly errors separately from unchanged residuals inherited from exhausted child nodes; the latter remain in the final review status and do not silently reset child budgets. New fit requests retain earlier named fitting results. Source selection includes the matching program, parameters, viewer and narrative when available.

### Agent diagnosis after stalled repair (new adaptive runs)

New adaptive runs launched by `scripts/rcwm_world.py` enable `agent_correction` in
**their copied** `assist_transport.json`. Existing running batches and frozen run
sources/configurations are unchanged. Explicit `{"agent_correction":{"enabled":false}}`
in the transport preserves the previous flow.

The feedback hook invokes `correction_review.py` for an object or scene only after
stagnation, review exhaustion, or an inherited stopped-child error. A numerical
pass stops immediately and never invokes visual diagnosis, including after repair.
An independent Codex CLI inspects original FIT RGB/render pairs, all available FIT
views, mask paths, instance bounds, code, and previous feedback. It uses the same
model/provider and reasoning settings as reconstruction. It returns a structured
visual diagnosis for unresolved failures; its judgment never overrides a failing numerical check.

A diagnosed repair reopens only named semantic object nodes. The scene can identify
a wrongly accepted sibling; an object reviewer cannot edit a sibling. One recovery
round per object is enabled by default, adding two measured reviews and four model
cycles without resetting the consumed base budget. Final repaired geometry is
numerically checked; a pass ends repair without another visual review. Previously selected, visually
rejected geometry cannot win merely by retaining a lower old depth score.

For identity/segmentation failures, the reviewer may propose simultaneous per-view
mask relabeling or original-pixel cropping of existing FIT masks. A server-local
geometry worker creates a new annotations/evidence version inside this run, reruns
RGB-D collection and consensus, refreshes the model-readable measurements, and
routes **both fitting and evaluation** to the corrected evidence. Stale local fit
recipes are archived and removed for affected objects. The original batch's masks,
observations, and assets are preserved. This is not arbitrary mask painting or a
new SAM segmentation pass; cases needing those return an observation request.

If evidence is insufficient, `observation_request.json` records the precise needed
views/details with `PENDING_ACQUISITION` and `capture_performed:false`. This module
**does not yet drive cameras or robots to acquire them**. Visual rejection, exhausted
repair, or diagnostic API failure remains explicitly unresolved. No simulator
geometry or held-out views enter these corrections.

Artifacts live under `fractal/<node>/correction/<timestamp>/` (`packet.json`,
`diagnosis.json`, `agent.log`, optional corrected annotations/evidence or acquisition
request), plus `correction_state.json` and `correction_instructions.md`. Run-wide
`correction/active_evidence.json` selects the currently active evidence version.
Mask correction currently requires `transport:local` on the server; other transports
can use geometry-only diagnosed repair but report mask-regeneration requests as
unsupported rather than silently applying partial corrections.

Correction diagnosis is restricted to instance association, segmentation, geometry,
pose and support. Texture, color, wrapper graphics, text, lighting and material
appearance are not reviewed and cannot justify repair or extra observations.
RGB appearance may only provide cues for tracking instance identity. Structured
`issue_types` restricts repair/observation escalation to structural categories.

The updated hook holds a per-node ownership lock, snapshots diagnostic inputs
under the shared feedback lock, releases that shared lock while Codex reasons,
and reacquires it before committing. Geometry/evidence versions are checked on
return; stale diagnoses cannot commit and instead request remeasurement without
refunding consumed reviews. Evidence rebuilds remain serialized. Mask assignment
source IDs refer to immutable original annotations, while untouched destination
masks retain already committed corrections, so sequential identity repairs do not
silently reinterpret an already relabeled source ID.

Identity swaps may update several known object masks in one transaction when the
diagnosis contains reciprocal per-view assignments linking them to the reviewed
object. Unrelated scope expansion remains invalid. After remeasurement, unchanged
accepted partners retain their code and recovery budget; only failing or changed
objects reopen. Export/source snapshots omit diagnostic history and adaptive best
archives; those remain available in the original run as audit records.
