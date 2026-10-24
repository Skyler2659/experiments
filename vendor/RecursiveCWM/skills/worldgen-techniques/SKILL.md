---
name: worldgen-techniques
description: Use when building 3D scenes/terrain from a reference image — proven techniques distilled from open-source worldgen projects
---

# Worldgen techniques (a digest of open-source practice)

A toolbox to draw on, not a rulebook. The core process is still the recursion: whole → parts → whole again.

## Terrain (after Infinigen and terrain-erosion-3-ways)

- **Layered height fields**: a low-frequency base + ridge detail (ridged noise, `1-|noise|`) +
  domain warping (perturb the sampling coordinates with another noise field to kill the grid look).
  Check each layer against the reference silhouette side by side.
- **Geomorphic operators**: a river valley = cut down along the water line + grade both banks;
  a lake basin = flatten locally + gentle shore slopes; a ridge line = ridge noise stretched along
  its direction. Place them one by one from the reference, not by global randomness.
- **Erosion**: a few dozen to a few hundred iterations of simplified hydraulic erosion (droplets
  carrying sediment downhill) immediately removes the "plastic" look; thermal erosion (collapse
  where the slope exceeds a threshold) makes cliff faces break naturally. Both fit in ~100 lines
  of JS or Python.
- **Cliffs without stretched textures**: use **triplanar** mapping (blend three projections by
  the normal) or a separate rock material on steep faces; projecting a height-field texture straight
  down always produces vertical streaks on steep faces.
- **Boundaries**: snow lines and rock/grass transitions use a height threshold plus a slope
  threshold, with noise on the edge; never a hard cut.

## Scene specification (plan before building)

Produce a structured specification first: regions / terrain features / assets (list and counts) /
materials / spatial relations (adjacency, orientation, attachment) — then build.

## Render-based refinement (the same loop as our eye-based review)

Render → side by side with the reference → change parameters → render again, at every level;
judge at magnification (the narrowed sub-frustum) first.

## Open-source code worth mining

- Infinigen (Princeton): fully procedural terrain, vegetation and materials.
- dandrino/terrain-erosion-3-ways: three erosion methods (simulation / ML / noise) with Python source.
- three.js built-ins: PlaneGeometry vertex displacement is a height field; ShaderMaterial does triplanar.
