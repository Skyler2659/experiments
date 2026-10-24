# solve(__NODE__) — the one recursive solver, at depth __DEPTH__

**This is an approved work order.** Make reversible choices on your own recommendation; do not stop for approval.

Chain directory: __CHAIN__; your node directory: __CHAIN__/fractal/__NODE__/
Materials: target.png (this level's target image) / view.json (the viewport) / brief.md;
manifest.json records the reference, camera and parent-snapshot versions you inherited (read-only).
Camera contract: __CHAIN__/camera-contract.json (read-only; if it does not exist and this is the root,
solve and calibrate one first and verify the full-frame overlay by eye before locking it).

## What you do at this level (one loop, identical at every level)
1. **Whole**: render this level's viewport and put it side by side with target.png at matched
   magnification; look for the conspicuous residuals by eye; repeat as needed. Also take one or two
   rotated views — any flatness a side view exposes (paper walls, billboard trees, parts without
   thickness) is a real residual and goes on the repair list right away.
2. **Parts**: repair here what can be repaired here. **A sub-problem that deserves its own solve**
   (one sub-object or sub-assembly that patching at this level cannot fix and that needs its own
   target image and loop) is not to be forced through.
   The test: take each part of the side-by-side where you can still see a difference and magnify
   it on its own. If what you see is **a group of objects and the layout relations between them**
   (arrangement, spacing, orientation, attachment) that this level has not yet produced, or
   **a single thing whose internal structure is complex enough to need its own loop**, it is a
   sub-problem: descend. If it is only a simple single thing or scattered small differences,
   finish it at this level. A level usually cuts two to four children; most scenes close in two
   or three levels.
   For a sub-problem, prepare its materials under
   __CHAIN__/fractal/<child>/ (target.png cut from your target and magnified / view.json / brief.md),
   list the child in children.json in your directory (a JSON array, e.g. ["part-a","part-b"]),
   then **end the session** — the runner starts this same solver for every child and wakes you
   with the results when they are done.
3. **Whole again** (after being woken): the children's part.json files are in place; integrate
   them into your level, go back to the side-by-side of the whole at this level and settle
   continuity and relations; if more descent is needed, update children.json and end again,
   otherwise finish.
4. **Finish**: write __CHAIN__/fractal/__NODE__/part.json (this level's final component, with
   child references) + account.md (this level's round-by-round account). part.json on disk
   means this level is done.

## Completion (at every level)
First identify: what this is, in what style, and what a complete instance of its kind looks
like — imagine the whole thing in your mind, then calibrate that mental image against the
visible evidence in the reference (two or three sentences in account.md).
Then: what is visible, match to the reference pixel by pixel; what is not visible (back faces,
occluded parts, boundary continuations), complete from the calibrated mental image with world
knowledge in the same style — a house has four walls and a full roof, a tree crown goes all the
way round, a road leads somewhere, terrain continues to the boundary.
Self-check: render the scene from the four compass directions and look; stage-set feel and
flatness (empty backs, dead-end roads, floating objects, paper parts) must be gone before
completion counts; re-verify the visible region against the reference in the same window and
do not let the completion pull it away.

Discipline: eyes decide, metrics are only recorded; free-form boxes; matched magnification;
no git; a child's write scope is its own fractal/<child>/. Everything else about the craft is
yours to decide.

Begin.
