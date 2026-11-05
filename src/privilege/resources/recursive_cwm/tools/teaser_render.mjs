// teaser_render.mjs <workspace-root> <page-path-under-root> <out.png> [scale=4] [ready-expr] [WxH=1400x963] [hide-name-regex]
// Reference-camera render at high scale with the screen-space furniture hidden (overlay canvases/divs, gradient
// backdrop, painted scene background): the scene program alone on white. Optionally hides scene objects whose
// name matches the regex (for example location pins). Only the WebGL canvas is written, at buffer resolution.
import fs from 'node:fs';
import { openViewer, DEFAULT_READY, HIDE_FURNITURE } from './lib/viewer_harness.mjs';
const [root, pageRel, out, scaleArg, readyArg, vpArg, hideRe] = process.argv.slice(2);
if (!out) { console.error('usage: teaser_render.mjs <workspace-root> <page-path-under-root> <out.png> [scale] [ready-expr] [WxH] [hide-name-regex]'); process.exit(2); }
const scale = Number(scaleArg || 4); const [W, H] = (vpArg || '1400x963').split('x').map(Number);
const v = await openViewer(root, pageRel, { width: W, height: H, scale, readyExpr: readyArg || DEFAULT_READY });
if (!v.hooked) { console.log(JSON.stringify({ hookMissing: true, pageErrors: v.errors })); await v.close(); process.exit(2); }
await v.page.evaluate(HIDE_FURNITURE);
const info = await v.page.evaluate(([hideRe, scale]) => {
  const r = window.__lastRenderer, s = window.__lastScene, c = window.__lastCamera; const hidden = [];
  if (hideRe) { const re = new RegExp(hideRe, 'i'); s.traverse(o => { if (o.name && re.test(o.name) && o.visible) { o.visible = false; hidden.push(o.name); } }); }
  const cw = r.domElement.clientWidth || r.domElement.width, ch = r.domElement.clientHeight || r.domElement.height;
  r.setPixelRatio(scale); r.setSize(cw, ch, false); r.render(s, c);
  return { hidden: hidden.slice(0, 20), buffer: [r.domElement.width, r.domElement.height], data: r.domElement.toDataURL('image/png') };
}, [hideRe || '', scale]);
fs.writeFileSync(out, Buffer.from(info.data.split(',')[1], 'base64'));
console.log(JSON.stringify({ out, scale, buffer: info.buffer, hidden: info.hidden, pageErrors: v.errors }));
await v.close();
