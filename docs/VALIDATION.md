# Packaging validation — 2026-11-06

This records checks of the WorldRUA source handoff. It is not a new robot evaluation.

| Check | Result |
|---|---|
| Fresh Python 3.12 environment, editable install including collision dependencies | Passed |
| Complete test suite with the recorded server dependency constraints | **472 passed, 6 skipped** |
| Focused packaging/recovery tests after formatting | Passed |
| Linux server: guard, recovery, checkpoint and packaging tests, no simulator | **20 passed** |
| Calibrated single-view and multi-view export/compile with the real browser runtime | **2 passed** |
| Source distribution and wheel build | Passed |
| Wheel installed outside the repository; imports and packaged XPolicyLab adapter | Passed |
| Two-frame synthetic RGB/CodeWorld export using a real headless browser | Passed |
| Textured current-state GLB, camera hiding, artifact archive | Passed |
| Local dashboard HTML/status HTTP routes and private-file refusal | Passed |
| Source scan for machine-specific paths, credentials and unwanted artifacts | Passed |

The synthetic render contained one material and one texture. Both observation-only
and historical-state comparison MP4s were encoded. The synthetic RGB is a placeholder;
this test checks the exporter, not reconstruction accuracy.

Three inherited test expectations were updated to match already-deployed behavior:
early `finish(stuck)` requires a recovery review, and public task completion instructions
may contain the word “success.” The robot strategy was not weakened to satisfy those
old assertions. Added tests cover staging without launching, duplicate samples,
coordinator adoption, PID reuse, GPU slot/memory accounting, login-copy cleanup,
visual material export, archive exclusions and dashboard file access.

No robot actions or model calls were issued by packaging validation. No existing
experiment was restarted. End-to-end qualification of this newly packaged CLI on a
GPU server remains necessary before a large campaign. The standard macOS suite skips
four Linux process-guard cases and two browser fixtures. These were subsequently
covered by the 20-test Linux check and the two real-browser export/compile tests.

Known limits are documented in RECOVERY.md and ARTIFACTS.md. In particular, simulator
death has no lossless physics resume, and task-plus-Agent-log video composition is
not implemented. The tested Linux constraints reproduce the host package versions;
they do not specify every simulator, system driver or external benchmark revision.

## Fresh deployment audit — 2026-11-06

The first GitHub Actions run exposed a Python 3.10 installation failure:
websockets 17.4 requires Python >=3.11. The constraint now selects 15.0.1 on 3.10;
3.11/3.12 keep the tested 17.4. CI runs both versions without fail-fast cancellation.

The audit also found missing OpenCV/h5py dependencies in the XPolicyLab host import
chain, undocumented SAM3/CUDA/weight requirements, and an inherited PATH that could
launch the simulator with the Agent Python. The requirements, examples and setup
instructions now cover those dependencies and select the simulator Python explicitly.
`doctor` probes both environments, requested layouts, CUDA and actual WebGL rendering.
It does not send a model request or boot a simulator.

A separate Python 3.12 environment installed the project and imported the real
XPolicyLab server. The updated source suite passed **475 tests, 6 skipped** on macOS;
`pip check` reported no broken requirements. A newly created rendering runtime
installed its own Chromium and successfully rendered the setup test box. These checks
still do not replace a full target-server robot episode or validate model permissions.

Linux GitHub Actions then passed **479 tests, 2 browser fixtures skipped**, on both
Python 3.10 and 3.12, including installation, wheel/sdist builds and release checks
([run](https://github.com/Skyler2659/experiments/actions/runs/42135404924)).
An independent clone also installed and built successfully. On the existing L20 server,
Agent/SAM3 imports, CUDA availability, simulator package discovery and all 33 requested
layout files passed the new probes. The rendering check exposed that its library search
path differed from the reconstruction runner; the smoke test and video renderer now
honor runtime-relative `.system-libs` too. The server's real Chromium test then passed.
A fresh macOS runtime also exported a two-frame RGB/CodeWorld video successfully.
No account credentials were copied for these checks and no robot episode was launched.
