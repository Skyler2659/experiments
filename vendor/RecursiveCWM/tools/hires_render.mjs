// hires_render.mjs <workspace-root> <page-path-under-root> <out.png> [scale=2] [ready-expr] [WxH=1400x963]
// True high-resolution re-render of a delivered viewer at its own (reference) camera: the renderer's pixel ratio is
// raised to <scale>, so the WebGL drawing buffer becomes W*scale x H*scale and geometry is rasterised at that size
// (a device scale factor alone would only stretch a 1x buffer). The whole page (canvas + overlay layers) is captured.
// Example: node tools/hires_render.mjs $RCWM_ROOT runs/my-scene/fractal/scene/index.html final-hires.png 4
import { openViewer, DEFAULT_READY } from './lib/viewer_harness.mjs';
const [root, pageRel, out, scaleArg, readyArg, vpArg] = process.argv.slice(2);
if (!out) { console.error('usage: hires_render.mjs <workspace-root> <page-path-under-root> <out.png> [scale] [ready-expr] [WxH]'); process.exit(2); }
const scale = Number(scaleArg || 2); const [W, H] = (vpArg || '1400x963').split('x').map(Number);
const v = await openViewer(root, pageRel, { width: W, height: H, scale, readyExpr: readyArg || DEFAULT_READY });
if (!v.hooked) { console.log(JSON.stringify({ hookMissing: true, pageErrors: v.errors })); await v.close(); process.exit(2); }
const info = await v.page.evaluate((scale) => {
  const r = window.__lastRenderer, s = window.__lastScene, c = window.__lastCamera;
  const cw = r.domElement.clientWidth || r.domElement.width, ch = r.domElement.clientHeight || r.domElement.height;
  r.setPixelRatio(scale); r.setSize(cw, ch, false); r.render(s, c);
  return { buffer: [r.domElement.width, r.domElement.height], css: [cw, ch] };
}, scale);
await v.page.screenshot({ path: out });
console.log(JSON.stringify({ out, scale, buffer: info.buffer, css: info.css, pageErrors: v.errors }));
await v.close();
