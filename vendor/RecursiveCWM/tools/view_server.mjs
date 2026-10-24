// view_server.mjs <workspace-root> [--port N] [--host 127.0.0.1]
// Interactive 3D viewing of any delivered scene program, without changing its files: serves the workspace, hooks the
// served three.module.js so the viewer's renderer/scene/camera are reachable, and injects orbit controls into every
// page under runs/ (drag to orbit, right-drag to pan, wheel to zoom, R = reset to the delivered camera, H = hide the
// hint, ?clean=1 hides the viewer's screen-space overlays). Open the URL it prints. Works for every node that ships
// its own index.html (the root always does; children usually deliver modules rendered through the root's viewer).
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
import { HOOK } from './lib/viewer_harness.mjs';
const argv = process.argv.slice(2); let root = null, port = Number(process.env.RCWM_VIEW_PORT || 0), host = '127.0.0.1';
for (let i = 0; i < argv.length; i++) { if (argv[i] === '--port') port = Number(argv[++i]); else if (argv[i] === '--host') host = argv[++i]; else root = argv[i]; }
if (!root) { console.error('usage: view_server.mjs <workspace-root> [--port N] [--host 127.0.0.1]'); process.exit(2); }
root = path.resolve(root);
const THREE_URL = '/.render-tools/node_modules/three/build/three.module.js';
const ORBIT_SRC = path.join(root, '.render-tools/node_modules/three/examples/jsm/controls/OrbitControls.js');
if (!fs.existsSync(ORBIT_SRC)) { console.error('no three.js under ' + root + '/.render-tools (is this a workspace root built by setup_runtime.sh / new_workspace.sh?)'); process.exit(1); }
const MIME = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.css': 'text/css', '.svg': 'image/svg+xml', '.otf': 'font/otf', '.ttf': 'font/ttf', '.wasm': 'application/wasm' };

// the client script injected into every html page under runs/
const CLIENT = `
(async () => {
  const until = (f) => new Promise(r => { const t = setInterval(() => { if (f()) { clearInterval(t); r(); } }, 150); });
  await until(() => window.__lastRenderer && window.__lastScene && window.__lastCamera && window.__THREE);
  await new Promise(r => setTimeout(r, 600));   // let the viewer finish its own first frames
  const { OrbitControls } = await import('/__rcwm/OrbitControls.js');
  const T = window.__THREE, r = window.__lastRenderer, s = window.__lastScene, c = window.__lastCamera, el = r.domElement;
  if (new URLSearchParams(location.search).get('clean') === '1') {
    document.querySelectorAll('body *').forEach(e => { if (e !== el && !e.contains(el)) e.style.visibility = 'hidden'; });
    document.body.style.background = 'white'; if (s.background) s.background = null; try { r.setClearColor(0xffffff, 1); } catch (e) {}
  }
  // overlays must not swallow the mouse
  document.querySelectorAll('body *').forEach(e => { if (e !== el && !e.contains(el)) e.style.pointerEvents = 'none'; });
  el.style.pointerEvents = 'auto'; el.style.touchAction = 'none';
  // orbit target: the first surface the delivered camera sees through the centre of its frame, else the content centre
  let target = null;
  try { const rc = new T.Raycaster(); rc.setFromCamera(new T.Vector2(0, 0), c); const h = rc.intersectObjects(s.children, true).filter(x => x.object.isMesh && x.object.visible); if (h.length) target = h[0].point.clone(); } catch (e) {}
  if (!target) { const b = new T.Box3(); s.traverse(o => { if (o.isMesh && o.visible) b.expandByObject(o); }); target = b.getCenter(new T.Vector3()); }
  const ctl = new OrbitControls(c, el); ctl.target.copy(target); ctl.enableDamping = true; ctl.dampingFactor = 0.12; ctl.zoomToCursor = true;
  ctl.enableDamping = false; ctl.update(); ctl.saveState(); ctl.enableDamping = true;   // the delivered camera is the saved state
  const reset = () => { ctl.enableDamping = false; ctl.reset(); ctl.enableDamping = true; };
  const hud = document.createElement('div'); hud.id = '__rcwm_hud';
  hud.style.cssText = 'position:fixed;left:12px;bottom:12px;z-index:2147483647;font:13px/1.4 system-ui,sans-serif;color:#fff;background:rgba(0,0,0,.65);padding:8px 10px;border-radius:6px;pointer-events:auto;user-select:none';
  hud.innerHTML = '<b>3D view</b> &nbsp; drag: orbit &nbsp;·&nbsp; right-drag / shift-drag: pan &nbsp;·&nbsp; wheel: zoom &nbsp;·&nbsp; <kbd>R</kbd> reset &nbsp;·&nbsp; <kbd>H</kbd> hide';
  document.body.appendChild(hud);
  window.addEventListener('keydown', e => { if (e.key === 'r' || e.key === 'R') reset(); if (e.key === 'h' || e.key === 'H') hud.hidden = !hud.hidden; });
  window.__rcwmView = { controls: ctl, reset, camera: c, scene: s, renderer: r };
  const loop = () => { ctl.update(); r.render(s, c); requestAnimationFrame(loop); }; loop();
})();
`;
const server = http.createServer((req, res) => {
  let u = decodeURIComponent(req.url.split('?')[0]);
  if (u === '/__rcwm/OrbitControls.js') { res.setHeader('Content-Type', 'text/javascript'); res.end(fs.readFileSync(ORBIT_SRC, 'utf8').replace(/from\s+['"]three['"]/g, `from '${THREE_URL}'`)); return; }
  if (u === '/__rcwm/orbit.js') { res.setHeader('Content-Type', 'text/javascript'); res.end(CLIENT); return; }
  if (u.startsWith('/vendor/') && !fs.existsSync(path.join(root, u))) u = '/.render-tools/node_modules/' + u.slice(8);
  let p = path.join(root, u); if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, 'index.html');
  fs.readFile(p, (e, b) => {
    if (e) { res.writeHead(404); res.end('not found: ' + u); return; }
    res.setHeader('Content-Type', MIME[path.extname(p)] || 'application/octet-stream');
    if (p.endsWith('three.module.js')) return res.end(Buffer.concat([b, Buffer.from(HOOK)]));
    if (p.endsWith('.html') && u.startsWith('/runs/')) { const tag = '\n<script type="module" src="/__rcwm/orbit.js"></script>\n'; const h = b.toString('utf8'); return res.end(h.includes('</body>') ? h.replace('</body>', tag + '</body>') : h + tag); }
    res.end(b);
  });
});
server.listen(port, host, () => {
  const base = `http://${host}:${server.address().port}`;
  console.log(`serving ${root} at ${base}  (Ctrl-C to stop)`);
  const runs = fs.existsSync(path.join(root, 'runs')) ? fs.readdirSync(path.join(root, 'runs')).filter(n => fs.existsSync(path.join(root, 'runs', n, 'fractal/scene/index.html'))) : [];
  for (const n of runs) console.log(`  ${base}/runs/${n}/fractal/scene/index.html${fs.existsSync(path.join(root, 'runs', n, 'fractal/scene/part.json')) ? '' : '   (not delivered yet)'}`);
  if (!runs.length) console.log('  (no runs/<name>/fractal/scene/index.html under this root yet)');
  console.log('  add ?clean=1 to hide the viewer\'s overlays; a child node that has its own index.html (runs/<name>/fractal/<node>/) opens the same way');
});
