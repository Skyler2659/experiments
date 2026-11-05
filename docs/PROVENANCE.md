# Source provenance

Prepared on 2026-11-06 from the final server-frozen **codeworld_resilience_20261106**
source, after the tool-contact calibration experiment was abandoned. The contact-end
calibration patch is not included. The existing reconstruction, tracking, collision,
finish-recovery and resilient CLI changes are retained.

PyRUA-Lean upstream base: `20dbb21e823030077ff4ddd4463cb91e36277515`,
https://github.com/DAGroup-PKU/PyRUA-Lean. This release includes local modifications;
the upstream commit alone does not describe WorldRUA.

RecursiveCWM base: `3ecb133bc51d663ca296f2c653b70fa51cb02564`,
https://github.com/ZhiqiLi-CG/RecursiveCWM_code. Its license, original source revision
record and embedded runner are under `src/privilege/resources/recursive_cwm/`.
Local reconstruction checkpoint and input-recovery modifications are included.

Packaging changes:

- Renamed the distribution/experiment entry point to WorldRUA while retaining
  `pyrualean` and `privilege` imports for compatibility.
- Moved the extracted CodeWorld extension into the same Python distribution.
- Added a portable config-driven batch runner based on the deployed owner/adoption
  and same-session recovery design; removed hard-coded hosts, logins and campaign paths.
- Packaged the XPolicyLab adapter so non-editable wheel installations can locate it.
- Retained call/state recorders and optional offline truth diagnostics.
- Parameterized rendering/media tools; added offline visual GLB and artifact export.
- Added localhost monitoring, documentation, release checks and packaging tests.

No historical trajectories, results, videos, login files, model checkpoints or simulator
assets are distributed. Local unrelated LIBERO development is not merged into this
RoboDojo-tested release. Upstream adapters remain available, but this release does not
claim our online CodeWorld strategy was validated on every upstream backend.

The server's RoboDojo/XPolicyLab source directories do not expose usable Git HEADs.
Their exact upstream revisions therefore cannot be claimed here. Preserve the target
deployment's own revisions and benchmark patches when running large experiments.
The historical PR48 drive-reset diff is preserved separately as an explicit benchmark
patch, not silently applied by WorldRUA.
