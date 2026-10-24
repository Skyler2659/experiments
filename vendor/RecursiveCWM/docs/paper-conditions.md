# Paper conditions and recorded runs

[Operating guide](usage.md) · [Runtime environment](environment.md)

## The paper's conditions

Following [§3](usage.md#3-install) and [§4](usage.md#4-run-one-scene) as written reproduces the paper's condition. What the code enforces, and what only you can check:

| condition | paper | enforced by |
|---|---|---|
| instruction | `solver/solver-template.md`, sha256 `8f194d22f285…` | default `RCWM_PROMPT`; hash printed at launch, recorded in `conditions.json` and in every trace event |
| recursion limits | depth ≤ 4, ≤ 3 cycles per node | defaults of `rcwm.sh` |
| executor | `gpt-6-astra`, reasoning effort `high` | passed to every `codex exec` (`-c model`, `-c model_reasoning_effort`) whatever `~/.codex/config.toml` says; recorded in `conditions.json` and each `session_start` event |
| what codex can see | only this repo's `worldgen-techniques` skill (English); no other skills, memories, `AGENTS.md` or config options | private codex home, built at launch (default). `rcwm.sh` prints `codex home: private (…); skills: worldgen-techniques` |
| sandbox | workspace-write, network on, Chromium cache writable | the runner's `codex exec` flags |
| root brief and environment note | identical text | `rcwm.sh` |
| runtime | Node 22.14, three 0.160.1, Playwright 1.62.1 with Chromium; Python 3.12 with Pillow 12.3, numpy 2.5 | `setup/setup_runtime.sh` pins them; versions recorded in `conditions.json` |
| codex CLI | 0.153.0 (0.154 verified compatible) | **you**: `rcwm.sh` prints a note for any other version; the runner parses codex's `session id` and `tokens used` lines |
| model access | an account that can run `gpt-6-astra` | **you**: `codex login`; a different model must be requested explicitly with `RCWM_MODEL` |
| reference image | unmodified, at the size given (figure furniture kept) | **you**: pass the file as is |
| isolation | one workspace per scene, nothing else in it | **you**: `tools/new_workspace.sh` per scene ([§7](usage.md#7-several-scenes-in-parallel-isolated)); do not put other material under `$RCWM_ROOT` |

Anything you change on purpose (`RCWM_MODEL`, `RCWM_REASONING`, `RCWM_PROMPT`, `RCWM_CLEAN_CODEX_HOME=0`,
`RCWM_CODEX_CONFIG`) is written to `runs/<name>/conditions.json`, so a run always carries the record of how it differs.
`tests/test_runner_offline.sh` checks the enforced rows with a fake codex ([§14](usage.md#14-tests)).


## The paper's runs in numbers

The ten runs behind the paper's main table (Table 1), scored with `tools/score_run.py`. Use them to know what to expect
from a scene of a given size:

| scene | nodes | max depth | nodes per level | PSNR↑ | SSIM↑ | Edge F1↑ | LPIPS↓ | CLIP↑ | tokens (M) | wall (min) |
|---|---|---|---|---|---|---|---|---|---|---|
| city-full | 24 | 4 | 1/4/12/6/1 | 18.2 | 0.66 | 0.94 | 0.158 | 0.95 | 2.4 | 69 |
| snow-village | 83 | 4 | 1/3/10/27/42 | 17.9 | 0.73 | 0.84 | 0.237 | 0.83 | 9.2 | 79 |
| island-harbor | 57 | 4 | 1/3/11/22/20 | 17.4 | 0.74 | 0.90 | 0.186 | 0.93 | 10.9 | 121 |
| medieval-village | 24 | 4 | 1/2/6/11/4 | 18.7 | 0.71 | 0.89 | 0.200 | 0.93 | 3.2 | 77 |
| japan-island | 29 | 3 | 1/4/10/14 | 19.3 | 0.73 | 0.87 | 0.201 | 0.96 | 3.9 | 79 |
| school-block | 12 | 3 | 1/3/6/2 | 16.4 | 0.56 | 0.97 | 0.171 | 0.94 | 1.1 | 65 |
| police-corner | 7 | 2 | 1/3/3 | 16.6 | 0.44 | 0.97 | 0.159 | 0.91 | 0.5 | 31 |
| park-lake | 12 | 3 | 1/3/6/2 | 23.3 | 0.83 | 0.99 | 0.075 | 0.95 | 1.1 | 55 |
| shop-row | 9 | 2 | 1/3/5 | 17.4 | 0.65 | 0.96 | 0.120 | 0.96 | 0.8 | 54 |
| valley-village | 21 | 3 | 1/3/9/8 | 14.1 | 0.38 | 0.48 | 0.583 | 0.87 | 3.2 | 86 |

Depth 0 is the root, so "max depth 4" is five levels. Tokens are summed over every session of the run; wall time is
first to last trace event. The delivered programs and full traces of medieval-village and city-full can be explored node
by node on the project page.
