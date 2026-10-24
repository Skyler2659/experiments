// Native WebGL preview and GLB export of a completed RCWM program.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import {sourceFiles} from './codeworld_source_files.mjs';
const [runtimeArg, name, outArg] = process.argv.slice(2);
if (!runtimeArg || !name || !outArg || !/^[\w-]+$/.test(name)) {
  throw new Error('usage: node export_world.mjs RUNTIME RUN_NAME OUTPUT_DIRECTORY');
}
const root = path.resolve(runtimeArg), out = path.resolve(outArg);
const part = path.join(root, 'runs', name, 'fractal/scene/part.json');
if (!fs.existsSync(part)) throw new Error('Root program has not been delivered');
const runRoot = path.join(root, 'runs', name);
const sourceBefore=sourceFiles(runRoot);
const png = fs.readFileSync(path.join(root, 'runs', name, 'fractal/scene/target.png'));
const viewport = {width:png.readUInt32BE(16), height:png.readUInt32BE(20)};
// Export the delivered build() independently of a model-authored viewer when requested.
// This changes only the viewing harness; source modules and geometry remain untouched.
const canonicalView = process.env.RCWM_EXPORT_CANONICAL_VIEW === '1';
const canonicalViews = canonicalView ? JSON.parse(fs.readFileSync(path.join(runRoot,'input/capture.json'))).views : [];
const canonicalCamera = canonicalViews.find(v=>v.view_id===(process.env.RCWM_EXPORT_VIEW || 'scanner_000'))
  || (!process.env.RCWM_EXPORT_VIEW && canonicalViews.find(v=>v.role==='fit'));
if(canonicalView && !canonicalCamera)throw Error('No calibrated FIT camera available for canonical export');
const canonicalHtml = canonicalView ? `<html><head><script type="importmap">{"imports":{"three":"/.render-tools/node_modules/three/build/three.module.js"}}</script></head><body style="margin:0"><script type="module">
import * as THREE from 'three';
import {build} from '/runs/${name}/fractal/scene/component.js';
window.ready=(async()=>{
 const scene=window.scene=new THREE.Scene();scene.background=new THREE.Color(0x8794aa);
 const component=build();if(!component?.isObject3D)throw Error('build() must return Object3D synchronously');scene.add(component);
 scene.add(new THREE.AmbientLight(0xffffff,1.25));const light=new THREE.DirectionalLight(0xffffff,1.4);light.position.set(1,-1,2);scene.add(light);
 const renderer=window.renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(1);document.body.appendChild(renderer.domElement);
 const v=${JSON.stringify(canonicalCamera)},[w,h]=v.image_size,K=v.K,n=.01;
 const camera=window.camera=new THREE.PerspectiveCamera();
 const m=new THREE.Matrix4().compose(new THREE.Vector3(...v.T_W_C.translation_m),new THREE.Quaternion(...v.T_W_C.quaternion_xyzw),new THREE.Vector3(1,1,1));
 m.multiply(new THREE.Matrix4().makeRotationX(Math.PI));m.decompose(camera.position,camera.quaternion,camera.scale);
 camera.projectionMatrix.makePerspective(-(K[0][2]+.5)*n/K[0][0],(w-K[0][2]-.5)*n/K[0][0],(K[1][2]+.5)*n/K[1][1],-(h-K[1][2]-.5)*n/K[1][1],n,10);
 camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();camera.updateMatrixWorld();renderer.setSize(w,h);renderer.render(scene,camera);
 return true;
})();</script></body></html>` : null;
fs.mkdirSync(out, {recursive: true});
const {chromium} = await import(path.join(root, '.render-tools/node_modules/playwright/index.mjs'));
const mime = {'.html':'text/html', '.js':'text/javascript', '.mjs':'text/javascript', '.json':'application/json', '.png':'image/png'};
const server = http.createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  if(canonicalView && u === '/runs/'+name+'/fractal/scene/index.html') {
    res.setHeader('Content-Type','text/html');res.end(canonicalHtml);return;
  }
  const file = path.resolve(root, '.' + u);
  const allowed = [path.join(root, 'runs', name) + path.sep, path.join(root, '.render-tools/node_modules/three') + path.sep];
  if (!allowed.some(prefix => file.startsWith(prefix))) {res.writeHead(403); res.end(); return;}
  fs.readFile(file, (error, bytes) => {
    if (error) {res.writeHead(404); res.end(); return;}
    res.setHeader('Content-Type', mime[path.extname(file)] || 'application/octet-stream');
    if (file.endsWith('.html')) {
      const map = '<script type="importmap">{"imports":{"three":"/.render-tools/node_modules/three/build/three.module.js"}}</script>';
      const html=bytes.toString(); res.end(html.includes('type="importmap"') ? html : html.replace('<head>', '<head>' + map)); return;
    }
    res.end(bytes);
  });
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({headless: true, args: ['--no-sandbox']});
  const page = await browser.newPage({viewport});
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${server.address().port}/runs/${name}/fractal/scene/index.html`);
  await page.waitForFunction('window.ready', null, {timeout:30000});
  await page.evaluate(async () => { await window.ready; });
  const binding = await page.evaluate(() => {
    window.scene.updateMatrixWorld(true);
    const objects={}, nodes=[];
    function walk(node, nodePath, owner=null) {
      const id=node.userData?.object_id;
      if(id) {
        if(objects[id]) throw new Error('Duplicate object ID: '+id);
        if(owner) throw new Error('Nested object ID requires explicit articulation support: '+id);
        owner=id; objects[id]={node_path:nodePath, matrix_world_column_major:node.matrixWorld.toArray(), user_data:node.userData};
      }
      nodes.push({path:nodePath, name:node.name, type:node.type, object_id:owner,
        matrix_local_column_major:node.matrix.toArray(), matrix_world_column_major:node.matrixWorld.toArray(),
        user_data:node.userData, vertices:node.geometry?.attributes?.position?.count ?? 0});
      node.children.forEach((child,i)=>walk(child,nodePath+'/'+i,owner));
    }
    walk(window.scene,'scene');
    return {schema:'rcwm-executed-binding-v1',objects,nodes};
  });
  await page.screenshot({path:path.join(out,'native_webgl.png')});
  const originalCamera = await page.evaluate(() => ({position:window.camera.position.toArray(), quaternion:window.camera.quaternion.toArray(), up:window.camera.up.toArray()}));
  for (const [label, position] of [['oblique',[0.65,-0.55,0.78]], ['reverse',[0,0.9,0.8]]]) {
    await page.evaluate(position => {
      const c = window.camera; c.position.fromArray(position); c.up.set(0,0,1);
      c.lookAt(0,0.28,0.03); c.updateMatrixWorld();
      window.renderer.render(window.scene,c);
    }, position);
    await page.screenshot({path:path.join(out,`native_${label}.png`)});
  }
  await page.evaluate(saved => {
    const c = window.camera; c.position.fromArray(saved.position); c.quaternion.fromArray(saved.quaternion); c.up.fromArray(saved.up); c.updateMatrixWorld();
    window.renderer.render(window.scene,c);
  }, originalCamera);
  const result = await page.evaluate(async () => {
    const scene = window.scene || window.__lastScene;
    if (!scene) throw new Error('Viewer does not expose its Three.js scene');
    const {GLTFExporter} = await import('/.render-tools/node_modules/three/examples/jsm/exporters/GLTFExporter.js');
    const data = await new GLTFExporter().parseAsync(scene, {binary:true});
    const base64 = await new Promise(resolve => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]);
      reader.readAsDataURL(new Blob([data]));
    });
    let meshes = 0; scene.traverse(o => {if (o.isMesh) meshes++;});
    return {base64, meshes};
  });
  if (process.env.RCWM_EXPORT_OBJECTS === '1') {
    const ids = await page.evaluate(() => {
      const ids=[]; window.scene.traverse(o=>{if(o.userData.object_id)ids.push(o.userData.object_id);});
      return ids;
    });
    if(!ids.length || new Set(ids).size!==ids.length) throw new Error('Missing or duplicate object identifiers');
    const objects=[];
    for(const id of ids){
      if(!/^[a-zA-Z0-9_-]+$/.test(id))throw new Error('Invalid object identifier');
      const item=await page.evaluate(async id=>{
        const {GLTFExporter}=await import('/.render-tools/node_modules/three/examples/jsm/exporters/GLTFExporter.js');
        let object; window.scene.updateMatrixWorld(true);
        window.scene.traverse(o=>{if(o.userData.object_id===id)object=o;});
        const clone=object.clone(true);
        object.matrixWorld.decompose(clone.position,clone.quaternion,clone.scale);
        clone.updateMatrixWorld(true);
        const data=await new GLTFExporter().parseAsync(clone,{binary:true});
        const base64=await new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.readAsDataURL(new Blob([data]));});
        return {id,origin:object.userData.geometry_origin,base64};
      },id);
      const dir=path.join(out,'objects',id);fs.mkdirSync(dir,{recursive:true});
      fs.writeFileSync(path.join(dir,'mesh.glb'),Buffer.from(item.base64,'base64'));
      objects.push({object_id:id,geometry_origin:item.origin,mesh_ref:`objects/${id}/mesh.glb`,coordinates:'world-baked Z-up meters'});
    }
    fs.writeFileSync(path.join(out,'object_inventory.json'),JSON.stringify(objects,null,2)+'\n');
  }
  if (errors.length) throw new Error(errors.join('\n'));
  const sourceAfter=sourceFiles(runRoot);
  if(JSON.stringify(sourceBefore)!==JSON.stringify(sourceAfter)) throw new Error('Source changed during export');
  binding.source_files=sourceBefore;
  binding.run=name;
  fs.writeFileSync(path.join(out,'executed_binding.json'),JSON.stringify(binding,null,2));
  fs.writeFileSync(path.join(out, 'scene.glb'), Buffer.from(result.base64, 'base64'));
  fs.writeFileSync(path.join(out, 'export.json'), JSON.stringify({run:name, renderer:'native Three.js WebGL', meshes:result.meshes, page_errors:errors, metric_geometry_verified:false, robot_execution_ready:false}, null, 2)+'\n');
  console.log(JSON.stringify({output:out, meshes:result.meshes}));
} finally {
  if (browser) await browser.close();
  server.close();
}
