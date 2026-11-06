# Server setup

## Environments

WorldRUA has three runtime layers:

1. **Agent/host Python**: install this repository with `pip install -e '.[dev,collision]'`.
2. **Rendering runtime**: `worldrua setup-runtime /path/to/runtime --python python3.12`.
   It installs a separate Python environment, Node, Three.js, Playwright and Chromium.
3. **RoboDojo/Isaac Sim/Isaac Lab**: install RoboDojo with its XPolicyLab submodule and robot/task
   assets using [RoboDojo installation instructions](https://robodojo-benchmark.com/doc/usage/install-and-download/). Ensure its evaluation shell scripts can
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

## CUDA, SAM3 and simulator Python

The Agent loads SAM3 even when VLA is disabled. Its environment needs CUDA PyTorch,
SAM3 and its checkpoint; these are separate from the rendering runtime. The L20 runs
used torch 2.7.0 + torchvision 0.22.0 with CUDA 12.8 (a compatible NVIDIA driver is
required). [PyTorch's versioned installation commands](https://pytorch.org/get-started/previous-versions/)
list this pair. Inside the Agent venv:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python scripts/install_sam3.py
```

The installer pins the tested SAM3 source and adds einops, pycocotools and psutil;
OpenCV and h5py are included in WorldRUA's dependencies because XPolicyLab imports them. This does
not install the simulator. Keep the simulator environment's own torch/numpy versions.

Download the SAM3 weights after obtaining access to `facebook/sam3` on Hugging Face:

```bash
hf auth login
hf download facebook/sam3 sam3.pt --local-dir "$HOME/models/sam3"
export SAM3_CHECKPOINT_PATH="$HOME/models/sam3/sam3.pt"
```

`hf` is installed with SAM3's huggingface_hub dependency. If the checkpoint is already
available, simply set `SAM3_CHECKPOINT_PATH` to that file. Weights are not in Git.

RoboDojo's `scripts/eval_policy.sh` launches **`python3` from PATH**. The example batch
configuration prepends the simulator environment's `bin` directory to child-process
PATH while the Agent continues to use its own absolute Python executable:

```bash
export ROBODOJO_ROOT=/path/to/RoboDojo
export ROBODOJO_PYTHON_BIN=/path/to/robodojo-env/bin
"$ROBODOJO_PYTHON_BIN/python3" -c 'import isaaclab, isaacsim, yaml, cv2, msgpack_numpy, transforms3d'
```

Use the Python environment created by RoboDojo's installation. If msgpack-numpy is
missing, install it with `"$ROBODOJO_PYTHON_BIN/python3" -m pip install msgpack-numpy`.
Download both object/robot assets and `Assets/Eval_Layout/RoboDojo/arx_x5/<eval_seed>/`
layouts for the requested tasks. Initialize the XPolicyLab submodule as instructed by
RoboDojo. The expected source directories are `env`, `src`, `task`, `utils`, `scripts`
and `env_cfg`. A code-only RoboDojo clone is insufficient.

For a headless server, inherit the EGL/Vulkan/driver variables from its working
RoboDojo installation, or put them in the batch `environment` object. If the installation
uses EULA environment switches, set them after accepting NVIDIA's license; the framework
does not accept it on the user's behalf. No old server driver paths are required.

## Chromium system libraries

`worldrua setup-runtime` installs the matching Chromium build and renders a test box.
If Linux reports missing shared libraries, install them on that host and rerun setup:

```bash
sudo "$WORLD_RUA_RUNTIME/.render-tools/node/bin/node" \
  "$WORLD_RUA_RUNTIME/.render-tools/node_modules/playwright/cli.js" install-deps chromium
```

Set `WORLD_RUA_RUNTIME` to the absolute runtime path first. Root access is only needed
for OS packages; use the normal experiment user for runtime setup and execution.

## Server login

Choose a directory outside the source tree and outside batch output:

```bash
export WORLD_RUA_AUTH_HOME=/path/to/server-experiment-login
CODEX_HOME="$WORLD_RUA_AUTH_HOME" codex -c 'cli_auth_credentials_store="file"' login
```

Install the Codex CLI using its [official setup instructions](https://developers.openai.com/codex/cli/),
and ensure `codex` is on PATH when loading the configuration. `doctor` checks the
required CLI options without sending a model request. The configured model
(default `gpt-6-astra`) must be available to the supplied account/API project.

Each episode gets a private login copy and separate sessions. Reconstruction gets its
own nested sessions. Only batch-owned copies are removed on completion. The source
login is untouched; no logout/revoke operation is performed. No local login is copied
by this repository's scripts. A provider needing a custom config can be added explicitly
to the deployment; the tested configuration uses the standard CLI login.

## Configuration

Copy `configs/robodojo11.example.json` to an ignored `*.local.json`. All relative paths
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

`worldrua doctor CONFIG` checks the configured Agent Python (including SAM3 imports
and CUDA), the simulator Python selected by child PATH (Isaac Sim/Isaac Lab and client
packages), checkpoint presence, requested layout files, NVIDIA card indices, CLI
options and real Chromium/WebGL rendering. It does not load SAM3 weights, consume
model calls or start a simulator. Thus a pass does not verify checkpoint integrity,
model permissions, all simulator assets, or an entire physics episode. `stage` only writes the queue. Before scaling,
qualify the simulator and provider with a small batch on the target machine. Two
episodes per L20 is the previously exercised configuration, not a guarantee for every
scene and driver. Three episodes per card has not been load-tested.

The preserved button drive-reset patch is in `patches/robodojo-pr48.patch`. It was
used for the corrected Swap Blocks experiments. It is an evaluator/environment fix,
not a WorldRUA strategy change; apply it to a private benchmark checkout and document
it, or use an upstream version already containing the fix. It is not applied silently.
