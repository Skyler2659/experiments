# Reproducing the paper

[Operating guide](usage.md) · [Paper conditions and recorded runs](paper-conditions.md) · [Code structure](code-structure.md)

Run commands from the repository root. Install the metrics packages as described in the
[installation guide](usage.md#3-install), prepare all ten [reference images](../experiments/pilot-scenes/REFERENCES.md),
and generate the scene and baseline runs before building the tables. The output directories must exist:

```bash
mkdir -p "$RCWM_ROOT/runs/pilot/metrics" OUT
```

## Tables and figures

The scripts read runs through `experiments/pickers.py` (one rule per method for "which image is the method's final
render") and are configured by environment variables so they contain no machine paths:

| variable | meaning |
|---|---|
| `RCWM_ROOT` | runtime/workspace root that holds `runs/` |
| `RCWM_REFS` | directory with the ten reference images `<scene>.png` (see `experiments/pilot-scenes/REFERENCES.md` to rebuild the five WorldClaw crops; the exact files are in the paper repository) |
| `RCWM_OURS_CHAIN` | where our run of each scene is, with `{scene}` substituted: e.g. `'/work/rcwm/{scene}/runs/{scene}'` (absolute pattern = one workspace per scene). Default: `runs/pilot/{scene}-recursive-r1` |
| `RCWM_OURS_OVERRIDE` | `scene=/path/render.png,…` forces the render used for a scene's figure (we used the 4× renders) |
| `RCWM_SEIG`, `RCWM_SEIG_RUN`, `RCWM_VIGA`, `RCWM_VIGA_TESTID`, `RCWM_I2T_ISO`, `RCWM_I2T_ISO_ALL=1` | where the baselines' outputs are (see `baselines/external/README.md`) |
| `RCWM_METRICS_CSV` | the CSV the metrics are accumulated in (default `$RCWM_ROOT/runs/pilot/metrics/all-methods-full.csv`) |
| `RCWM_PAPER` / `RCWM_NO_PAPER=1` | paper checkout whose `sections/04_experiments.tex` receives the table, or write the table next to the CSV only |

```bash
PY=$RCWM_ROOT/.venv/bin/python
$PY experiments/refresh_metrics.py                      # metrics of every (scene, method) → $RCWM_METRICS_CSV
RCWM_NO_PAPER=1 python3 experiments/main_table.py       # Table 1 (LaTeX + markdown next to the CSV)
$PY experiments/make_case_figures.py OUT --full city-full,medieval-village \
   --ours snow-village,island-harbor,japan-island,school-block,police-corner,park-lake,shop-row,valley-village
                                                        # Figs. 3-4 (reference | ours | baselines, progressive detail windows)
                                                        # and Figs. 5-8 (reference vs ours); RCWM_CASE_CANW=2000 for 4× renders,
                                                        # RCWM_CASE_FIX='scene=idx:fx,fy,ws;…' pins detail windows by hand
$PY experiments/make_novel_views.py OUT/novel-views-grid.png   # Fig. 9: one row per scene, reference camera + five novel views
                                                        # (reads fractal/scene/novel-views/ written by tools/render_views.sh)
python3 experiments/variant_table.py                    # ablation table (Table 2/3) from runs/pilot/<scene>-<variant>-r1
```

`experiments/eval_metrics.py reference.png render.png` computes one pair's metrics (JSON); metrics missing a library
come out `null`. Scores are recorded only; nothing in the pipeline reads them.

## Baselines and ablations

**Ablation (structural controls).** Same runner, different instruction and depth cap:

```bash
experiments/run_matrix.sh "school-block medieval-village" "flat localglobal globallocal twolevel recursive" 1 4
```

`flat` = `baselines/variants/flat-zoom.md` at depth 0; `localglobal` at depth 1; `globallocal` at depth 0;
`twolevel` = the main instruction with `RCWM_MAXD=1`; `recursive` = the main instruction at depth 4. Runs land in
`$RCWM_ROOT/runs/pilot/<scene>-<variant>-r1`, rows in `runs/pilot/results.csv`; the last argument limits concurrent
codex processes (it counts every `codex exec` on the machine). References come from `$RCWM_REFS/<scene>.png` (default `experiments/pilot-scenes/`).

**External baselines.** `baselines/external/README.md` describes the three (SEIG reproduction, VIGA with the official
runner and a codex shim, img2threejs in an isolated directory per scene) and their launch scripts; each needs its own
checkout (`RCWM_SEIG`, `RCWM_VIGA` + `BLENDER`, `$RCWM_ROOT/vendor/img2threejs` + `RCWM_I2T_ISO`). Every method receives
the reference image unmodified.


## Configuration details

The [external baseline guide](../baselines/external/README.md) describes the protocols;
the [script map](code-structure.md#external-baselines) lists each launcher's arguments and environment variables.
The launchers require separate installed checkouts. SEIG and img2threejs inherit their Codex configuration;
VIGA passes `gpt-5.6-sol` as its endpoint model name, and the separately started shim controls the executor.
Check those configurations when matching the paper's base model and reasoning effort.

The matrix driver calls the recursive runner directly. To give a matrix the private Codex home used by the
paper, prepare it before launching the matrix (after logging in):

```bash
tools/make_codex_home.sh "$RCWM_ROOT"
export CODEX_HOME="$RCWM_ROOT/.codex-home"
```

The matrix's concurrency argument is a launch-time check of all `codex exec` processes on the machine;
children can increase the count after a run starts. For one workspace per scene, follow the
[isolation guide](usage.md#7-several-scenes-in-parallel-isolated).

The main table includes only rows marked `final=True` with PSNR. Figure panels for incomplete methods are
left blank. `RCWM_OURS_OVERRIDE` affects the case-figure script only; it does not change the metric pickers.
The novel-view grid requires all six views for each of its ten scenes.
The ablation table reads repetition `r1` for school-block and medieval-village; its CSV retains cost and
time even though those columns are hidden in the LaTeX table. See the
[selection rules](code-structure.md#render-selection-rules) for exact completion checks.
