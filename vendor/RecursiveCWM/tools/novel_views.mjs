// novel_views.mjs <workspace-root> <page-path-under-root> <out-dir> [ready-expr] [WxH=1400x963]
// Renders a delivered scene program from five cameras it was never shown from, with one generic harness:
//   view-L35 / view-R35   azimuth -35 / +35 degrees around the point the reference camera looks at
//   view-orbit            azimuth +25, elevation +30, slightly closer
//   view-close1 / close2  2.4x zoom on two off-centre look points
// The anchor is the first surface the reference camera sees through the centre of its frame (a ray cast), falling
// back to the content's floor level. Screen-space furniture is hidden; the WebGL canvas is written at 2x.
// Example: node tools/novel_views.mjs $RCWM_ROOT runs/my-scene/fractal/scene/index.html runs/my-scene/fractal/scene/novel-views
import fs from 'node:fs'; import path from 'node:path';
import { openViewer, DEFAULT_READY, HIDE_FURNITURE } from './lib/viewer_harness.mjs';
const [root, pageRel, outDir, readyArg, vpArg] = process.argv.slice(2);
if (!outDir) { console.error('usage: novel_views.mjs <workspace-root> <page-path-under-root> <out-dir> [ready-expr] [WxH]'); process.exit(2); }
const [W, H] = (vpArg || '1400x963').split('x').map(Number);
const v = await openViewer(root, pageRel, { width: W, height: H, scale: 2, readyExpr: readyArg || DEFAULT_READY });
if (!v.hooked) { console.log(JSON.stringify({ hookMissing: true, pageErrors: v.errors })); await v.close(); process.exit(2); }
fs.mkdirSync(outDir, { recursive: true });
await v.page.evaluate(HIDE_FURNITURE);
const VIEWS = [['view-L35', { az: -35 }], ['view-R35', { az: 35 }], ['view-orbit', { az: 25, el: 30 }], ['view-close1', { zoom: 2.4, look: -0.2 }], ['view-close2', { zoom: 2.4, look: 0.2 }]];
const report = {};
for (const [name, spec] of VIEWS) {
  const r = await v.page.evaluate((spec) => {
    const T = window.__THREE, r = window.__lastRenderer, s = window.__lastScene, c = window.__lastCamera;
    if (!window.__pose0) {
      const box = new T.Box3(); s.traverse(o => { if (o.isMesh && o.visible) box.expandByObject(o); });
      const centre = box.getCenter(new T.Vector3()); const size = box.getSize(new T.Vector3());
      window.__pose0 = { pos: c.position.clone(), up: c.up.clone(), q: c.quaternion.clone(), zoom: c.zoom, centre, floorY: box.min.y + 0.02 * size.y };
    }
    const p = window.__pose0;
    c.position.copy(p.pos); c.up.copy(p.up); c.quaternion.copy(p.q); c.zoom = p.zoom; if (c.clearViewOffset) c.clearViewOffset();
    const dir = new T.Vector3(0, 0, -1).applyQuaternion(p.q);
    let t = null;
    try { const rc = new T.Raycaster(); rc.setFromCamera(new T.Vector2(0, 0), c); const hits = rc.intersectObjects(s.children, true).filter(h => h.object.isMesh && h.object.visible); if (hits.length) t = hits[0].point.clone(); } catch (e) {}
    if (!t) { const sgn = (p.floorY - p.pos.y) / (dir.y || -1e-9); t = p.pos.clone().addScaledVector(dir, (sgn > 0 && sgn < 1e6) ? sgn : p.pos.distanceTo(p.centre)); }
    const extent = c.isOrthographicCamera ? (c.right - c.left) / p.zoom : 2 * p.pos.distanceTo(t) * Math.tan((c.fov || 50) * Math.PI / 360) * (c.aspect || 1.45);
    let target = t.clone();
    if (spec.az !== undefined || spec.el !== undefined) {
      const rel = p.pos.clone().sub(t);
      if (spec.az) rel.applyAxisAngle(new T.Vector3(0, 1, 0), spec.az * Math.PI / 180);
      if (spec.el) { const axis = new T.Vector3().crossVectors(rel, new T.Vector3(0, 1, 0)).normalize(); rel.applyAxisAngle(axis, spec.el * Math.PI / 180); rel.multiplyScalar(0.9); }
      c.position.copy(t).add(rel);
    }
    if (spec.zoom) {
      const right = new T.Vector3(1, 0, 0).applyQuaternion(p.q); right.y = 0; right.normalize();
      target = t.clone().addScaledVector(right, spec.look * extent); c.zoom = p.zoom * spec.zoom;
      c.position.copy(target).add(p.pos.clone().sub(t));
    }
    c.lookAt(target); c.updateMatrixWorld(); c.updateProjectionMatrix(); r.render(s, c);
    let data = null; try { data = r.domElement.toDataURL('image/png'); } catch (e) {}
    return { pos: c.position.toArray().map(x => +x.toFixed(2)), zoom: c.zoom, data };
  }, spec);
  const file = path.join(outDir, name + '.png');
  if (r.data && r.data.length > 2000) fs.writeFileSync(file, Buffer.from(r.data.split(',')[1], 'base64')); else await v.page.screenshot({ path: file });
  delete r.data; report[name] = r;
}
console.log(JSON.stringify({ out: outDir, views: report, pageErrors: v.errors }));
await v.close();
