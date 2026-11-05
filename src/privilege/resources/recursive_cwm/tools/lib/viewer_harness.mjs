// Shared pieces of the render harnesses (hires_render / novel_views / teaser_render):
//  - a static file server rooted at the workspace (so a delivered viewer's absolute imports such as
//    /.render-tools/node_modules/three/build/three.module.js and /runs/<run>/fractal/... resolve);
//  - a hook appended to the served three.module.js that records the renderer, scene and camera of the last
//    render call, so any delivered viewer can be re-rendered without per-viewer wiring;
//  - Playwright resolved from the workspace's runtime (.render-tools) or from $RCWM_ROOT.
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';

export const HOOK = `
;(function(){try{
  // three.js assigns this.render inside the WebGLRenderer constructor, so the assignment is intercepted with a
  // prototype accessor; the scene with the most meshes is remembered as "the" scene (viewers may render overlays too)
  Object.defineProperty(WebGLRenderer.prototype,'render',{configurable:true,get(){return undefined},set(fn){const self=this;Object.defineProperty(self,'render',{configurable:true,writable:true,value:function(s,c){window.__lastRenderer=self;window.__renders=window.__renders||[];if(!window.__renders.some(x=>x.s===s))window.__renders.push({s,c});var best=window.__renders.map(x=>{var n=0;x.s.traverse(function(o){if(o.isMesh)n++});return {x:x,n:n}}).sort(function(a,b){return b.n-a.n})[0];window.__lastScene=best.x.s;window.__lastCamera=best.x.c;return fn.call(self,s,c)}})}});
  window.__THREE={Vector3,Vector2,Box3,Quaternion,Matrix4,Raycaster};window.__hookInstalled=true;
}catch(e){window.__hookError=String(e)}})();`;

const MIME = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.html': 'text/html', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.otf': 'font/otf', '.ttf': 'font/ttf', '.css': 'text/css', '.svg': 'image/svg+xml', '.wasm': 'application/wasm' };

export const DEFAULT_READY = '(window.ready || window.__SCENE_READY__ || window.sceneReady || window.__READY__ || window.renderReady)';
export const READY_TIMEOUT = Number(process.env.RCWM_READY_TIMEOUT_MS || 180000);   // viewers without a ready flag: waits this long, then proceeds

export async function serveRoot(root) {
  const server = http.createServer((req, res) => {
    let u = decodeURIComponent(req.url.split('?')[0]);
    if (u.startsWith('/vendor/') && !fs.existsSync(path.join(root, u))) u = '/.render-tools/node_modules/' + u.slice(8);   // some viewers alias vendor/ to node_modules
    const p = path.join(root, u);
    fs.readFile(p, (e, b) => {
      if (e) { res.writeHead(404); res.end('not found'); return; }
      res.setHeader('Content-Type', MIME[path.extname(p)] || 'application/octet-stream');
      res.end(p.endsWith('three.module.js') ? Buffer.concat([b, Buffer.from(HOOK)]) : b);
    });
  });
  await new Promise(r => server.listen(0, '127.0.0.1', r));
  return { server, port: server.address().port, close: () => server.close() };
}

export function resolvePlaywright(root) {
  for (const base of [root, process.env.RCWM_ROOT, path.join(path.dirname(new URL(import.meta.url).pathname), '../../runtime')]) {
    if (!base) continue;
    const p = path.join(base, '.render-tools/node_modules/playwright/index.mjs');
    if (fs.existsSync(p)) return p;
  }
  throw new Error('playwright not found under <root>/.render-tools or $RCWM_ROOT/.render-tools (run setup/setup_runtime.sh)');
}

// Opens the viewer page, waits for its ready flag (or the timeout), then for the hook to have seen a render.
export async function openViewer(root, pageRel, { width = 1400, height = 963, scale = 1, readyExpr = DEFAULT_READY } = {}) {
  const srv = await serveRoot(root);
  const { chromium } = await import(resolvePlaywright(root));
  const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--disable-dev-shm-usage'] });
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: scale });
  const errors = []; page.on('pageerror', e => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${srv.port}/${pageRel.replace(/^\//, '')}`);
  try { await page.waitForFunction(readyExpr, null, { timeout: READY_TIMEOUT }); } catch (e) { errors.push('ready flag not seen; proceeded after timeout'); await page.waitForTimeout(20000); }
  let hooked = true;
  try { await page.waitForFunction(() => window.__lastRenderer && window.__lastCamera, null, { timeout: 30000 }); } catch (e) { hooked = false; }
  return { page, browser, errors, hooked, close: async () => { await browser.close(); srv.close(); } };
}

// Hides everything on the page except the WebGL canvas (overlay canvases, tab/badge divs, gradient backdrops) and
// clears a painted scene background, so the capture shows the scene program alone on white.
export const HIDE_FURNITURE = `(() => { const r = window.__lastRenderer, s = window.__lastScene; const el = r.domElement;
  document.querySelectorAll('body *').forEach(e => { if (e !== el && !e.contains(el)) e.style.visibility = 'hidden'; });
  document.body.style.background = 'white'; if (s.background) s.background = null; try { r.setClearColor(0xffffff, 1); } catch (e) {} })()`;
