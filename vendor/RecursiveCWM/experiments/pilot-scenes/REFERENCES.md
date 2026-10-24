# Reference images

## City (in this directory)
`city-full.png` is the example composition shipped with the CC0 "Isometric city" sprite pack by JanaChumi
(https://opengameart.org/content/isometric-city-0, OpenGameArt, 2017). `school-block.png`, `police-corner.png`,
`park-lake.png`, `shop-row.png` are crops of it.

## WorldClaw scenes
Five references are crops of demonstration figures in the WorldClaw paper (Guo et al., arXiv:2608.05248).
They are used only as input images; WorldClaw is a text-to-3D-world generator and is not a baseline.
`medieval-village.png` (Fig. 9, the paper's worked example and the scene on the project page) is included in this
directory, byte-identical to the file the paper used (hash below). The other four are not redistributed here. To rebuild
them, render the page that carries each figure to an image 8516 px wide (we used the arXiv PDF at about 1000 dpi; page
renders were 8516×10144 px), then run `make_references.py <dir-with-page-renders> <out-dir>` with the crop table below.
Because of JPEG and resampling differences the rebuilt files match ours to about 16–30 dB PSNR, not byte for byte;
the exact files used in the paper are in the paper repository under `evidence/references/` and their hashes are listed here.

| scene | WorldClaw figure | page-render file | crop box (x, y, w, h) in page pixels | output size | sha256 (first 16) |
|---|---|---|---|---|---|
| medieval-village (in this directory) | Fig. 9  | fig09.png | 2208, 804, 4200, 2888 | 1400×963 | bfcdc380cc2a3f34 |
| snow-village     | Fig. 10 | fig10.png | 2200, 796, 4200, 2888 | 1400×963 | 7717a496b1c87b15 |
| island-harbor    | Fig. 4  | fig04.png | 2196, 796, 4200, 2888 | 1400×963 | e36bb3f59e316edb |
| japan-island     | Fig. 12 | fig12.png | 2192, 800, 4200, 2888 | 1400×963 | 62b17239b1aae166 |
| valley-village   | Fig. 15 | fig15.png | 76, 2196, 1987, 1447 | 1987×1447 (no resize) | 9dc0cf6462402db8 |

All figure furniture in the crops (the a–d view tabs, biome badge, location pins) is kept on purpose; every method
receives the same unmodified image and the metrics are computed against it.
