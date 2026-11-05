// Render one delivered scene through every assigned camera, preserving its materials.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
const [rootArg, name, outArg] = process.argv.slice(2);
if (!outArg) throw Error('Usage: node render_calibrated.mjs RUNTIME RUN_NAME OUTPUT_DIR');
const root = path.resolve(rootArg), out = path.resolve(outArg);
const run = path.join(root, 'runs', name);
const packet = JSON.parse(fs.readFileSync(path.join(run, 'input/views.json')));
const viewer = fs.existsSync(path.join(run, 'fractal/scene/index.html'))
  ? 'fractal/scene/index.html' : 'scene/index.html';
const {chromium} = await import(path.join(root, '.render-tools/node_modules/playwright/index.mjs'));
const server = http.createServer((req,res) => {
  const file = path.resolve(root, '.' + decodeURIComponent(req.url.split('?')[0]));
  if (!file.startsWith(root + path.sep)) {res.writeHead(403);res.end();return;}
  fs.readFile(file, (err, data) => {
    if (err) {res.writeHead(404);res.end();return;}
    res.setHeader('Content-Type', ({'.js':'text/javascript','.mjs':'text/javascript',
      '.html':'text/html','.json':'application/json','.png':'image/png'})[path.extname(file)] || 'application/octet-stream');
    if (file.endsWith('.html')) {
      const html=data.toString();
      const map='<script type="importmap">{"imports":{"three":"/.render-tools/node_modules/three/build/three.module.js"}}</script>';
      res.end(html.includes('type="importmap"') ? html : html.replace('<head>','<head>'+map));
    } else res.end(data);
  });
});
await new Promise(resolve => server.listen(0,'127.0.0.1',resolve));
let browser;
try {
  const browserEnv={...process.env};
  const libraries=path.join(root,'.system-libs/usr/lib/x86_64-linux-gnu');
  if(fs.existsSync(libraries)) browserEnv.LD_LIBRARY_PATH=libraries+':'+(browserEnv.LD_LIBRARY_PATH||'');
  browser = await chromium.launch({headless:true,args:['--no-sandbox'],env:browserEnv});
  const page = await browser.newPage();
  page.on('pageerror',e=>console.error('Viewer error:',String(e)));
  await page.goto(`http://127.0.0.1:${server.address().port}/runs/${name}/${viewer}`);
  await page.waitForFunction('window.ready',null,{timeout:60000});
  await page.evaluate(async()=>{await window.ready;});
  fs.mkdirSync(out,{recursive:true});
  for (const view of packet.views) {
    if (!/^[a-zA-Z0-9_-]+$/.test(view.view_id)) throw Error('Unsafe view_id');
    await page.setViewportSize({width:view.image_size[0],height:view.image_size[1]});
    const png = await page.evaluate(async v => {
      const THREE = await import('/.render-tools/node_modules/three/build/three.module.js');
      const [w,h] = v.image_size, k=v.K, n=.01;
      const c = new THREE.PerspectiveCamera();
      const t = new THREE.Matrix4().set(...v.T_W_C.flat());
      t.multiply(new THREE.Matrix4().makeRotationX(Math.PI));
      t.decompose(c.position,c.quaternion,c.scale);
      c.projectionMatrix.makePerspective(-k[0][2]*n/k[0][0],(w-k[0][2])*n/k[0][0],
        k[1][2]*n/k[1][1],-(h-k[1][2])*n/k[1][1],n,100);
      c.projectionMatrixInverse.copy(c.projectionMatrix).invert();c.updateMatrixWorld(true);
      const r=window.renderer;r.setPixelRatio(1);r.setSize(w,h);
      r.render(window.scene,c);
      return r.domElement.toDataURL('image/png').split(',')[1];
    },view);
    fs.writeFileSync(path.join(out,view.view_id+'.png'),Buffer.from(png,'base64'));
  }
  fs.writeFileSync(path.join(out,'cameras.json'),JSON.stringify(packet,null,2));
} finally {
  if (browser) await browser.close();
  server.close();
}
