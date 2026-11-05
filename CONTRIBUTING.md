# Development

Install `.[dev,collision]`, run `python -m pytest -q`, then build the wheel with
`python -m build`. Keep compatibility with the embedded PyRUA APIs. Changes to agent
prompts, observation access, action limits, collision handling or evaluator behavior
must be recorded separately from packaging changes.

Use `*.local.json` for machine configuration and `runs/` for outputs. Neither should
be committed. Do not add login directories, model weights, simulator assets or archived
experiment trees. Keep third-party copyright notices and license files.

Host-side qualification uses fake backends. Full RoboDojo validation needs the external
simulator, an authorized model account and a GPU; do not claim unit tests are evaluated
robot task success.
