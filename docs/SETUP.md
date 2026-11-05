# Server setup

## Environments

WorldRUA has three runtime layers:

1. **Agent/host Python**: install this repository with `pip install -e '.[dev,collision]'`.
2. **Rendering runtime**: `worldrua setup-runtime /path/to/runtime --python python3.12`.
   It installs a separate Python environment, Node, Three.js, Playwright and Chromium.
3. **RoboDojo/Isaac Sim**: install RoboDojo with its XPolicyLab submodule and robot/task
   assets using that project's instructions. Ensure its evaluation shell scripts can
   start the simulator on this machine. WorldRUA does not redistribute these assets.

The tested host dependencies are recorded in `requirements/constraints-tested-linux.txt`.
For a controlled deployment, use:

```bash
python -m pip install -c requirements/constraints-tested-linux.txt -e '.[dev,collision]'
```

The simulator may need its own Python version and `msgpack-numpy`; do not replace the
simulator's environment with the agent environment. XPolicyLab's host uses websockets
13 or later. See its adapter's `ROBODOJO_XPL_SITE` option if host/client dependency
versions need separate locations.

The rendering setup uses the embedded runner's version pins. Chromium on Linux also
needs system libraries; use Playwright's `install-deps chromium` in that runtime if
browser startup reports missing libraries. On macOS, pass `--node-from /path/to/node-prefix`
to `setup-runtime`; simulator batch scheduling remains Linux-only (`/proc`, NVIDIA).

## Server login

Choose a directory outside the source tree and outside batch output:

```bash
export WORLD_RUA_AUTH_HOME=/path/to/server-experiment-login
CODEX_HOME="$WORLD_RUA_AUTH_HOME" codex login
```

Each episode gets a private login copy and separate sessions. Reconstruction gets its
own nested sessions. Only batch-owned copies are removed on completion. The source
login is untouched; no logout/revoke operation is performed. No local login is copied
by this repository's scripts. A provider needing a custom config can be added explicitly
to the deployment; the tested configuration uses the standard CLI login.

## Configuration

Copy `configs/robodojo.example.json` to an ignored `*.local.json`. All relative paths
resolve against the config file. `${VARIABLE}` is expanded from the environment and
missing variables are errors. Sample numbers are layout indices; `eval_seed` selects
the RoboDojo layout set. Each run records the resolved configuration and its exact
command. Use a new output directory for new experiments.

The `environment` object supplies simulator-specific settings, for example `PATH`,
`LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, `__EGL_VENDOR_LIBRARY_FILENAMES`,
`XDG_RUNTIME_DIR`, and proxy variables. Set these to the target server's installation;
none of the original server paths or proxy ports are hard-coded here. Accept simulator
licenses according to your deployment. GPU indices refer to physical `nvidia-smi`
indices; do not set `CUDA_VISIBLE_DEVICES` to a conflicting remapping.

`truth_audit: true` enables the original offline simulator recorder in a private copy
of RoboDojo. Its data is never inserted into policy observations or reconstruction.
It is off by default and is not needed to run WorldRUA, export task videos or save
CodeWorld/agent logs. Report its use when analyzing geometric error.

## Qualification

`worldrua doctor CONFIG` checks paths, packages and task config files; it does not
consume model calls or start a robot. `stage` only writes the queue. Before scaling,
qualify the simulator and provider with a small batch on the target machine. Two
episodes per L20 is the previously exercised configuration, not a guarantee for every
scene and driver. Three episodes per card has not been load-tested.

The preserved button drive-reset patch is in `patches/robodojo-pr48.patch`. It was
used for the corrected Swap Blocks experiments. It is an evaluator/environment fix,
not a WorldRUA strategy change; apply it to a private benchmark checkout and document
it, or use an upstream version already containing the fix. It is not applied silently.
