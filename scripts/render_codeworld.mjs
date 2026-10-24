// Render the current CodeWorld from calibrated cameras, for any reconstruction backend.
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';

const [runtimeArg, workspaceArg, captureArg, outputArg] = process.argv.slice(2);
if (!outputArg) throw new Error('usage: node render_codeworld.mjs RUNTIME WORKSPACE CAPTURE OUTPUT');
const runtime = path.resolve(runtimeArg), workspace = path.resolve(workspaceArg);
const output = path.resolve(outputArg);
const capture = JSON.parse(fs.readFileSync(captureArg, 'utf8'));
const {chromium} = await import(path.join(runtime, '.render-tools/node_modules/playwright/index.mjs'));
fs.mkdirSync(output, {recursive: true});
const html = `<!doctype html><html><head><style>body{margin:0}</style>
<script type="importmap">{"imports":{"three":"/three/build/three.module.js"}}</script></head>
<body><script type="module">
import * as THREE from 'three';
import {GLTFLoader} from '/three/examples/jsm/loaders/GLTFLoader.js';
window.ready=(async()=>{
 const world=await fetch('/world/world.json').then(r=>r.json());
 const state=await fetch('/world/state/current.json').then(r=>r.json());
 const scene=window.scene=new THREE.Scene();scene.background=new THREE.Color(0x8794aa);
 const root=window.worldRoot=new THREE.Group();root.name='CodeWorld_W_Z_up';scene.add(root);
 const loader=new GLTFLoader();
 for(const [id,ref] of Object.entries(world.objects)){
  const metadata=await fetch('/world/'+ref).then(r=>r.json());
  const object=new THREE.Group();object.name=id;object.userData.object_id=id;
  const assembly=state.objects[id].assembly;
  if(assembly){
   for(const [partId,part] of Object.entries(assembly.parts)){
    const asset=await loader.loadAsync('/world/'+part.local_mesh);
    const node=new THREE.Group();node.name=partId;node.add(asset.scene);node.matrixAutoUpdate=false;
    node.matrix.fromArray(part.T_object_part.flat()).transpose();object.add(node);
   }
  }else{const asset=await loader.loadAsync('/world/'+metadata.geometry.local_mesh);object.add(asset.scene);}
  object.matrixAutoUpdate=false;
  object.matrix.fromArray(state.objects[id].T_W_object.flat()).transpose();root.add(object);
 }
 scene.add(new THREE.AmbientLight(0xffffff,1.25));
 const light=new THREE.DirectionalLight(0xffffff,1.4);light.position.set(1,-1,2);scene.add(light);
 window.camera=new THREE.PerspectiveCamera();window.camera.up.set(0,0,1);
 const renderer=window.renderer=new THREE.WebGLRenderer({antialias:true});
 renderer.setPixelRatio(1);renderer.setSize(512,512);document.body.appendChild(renderer.domElement);
 window.setCamera=v=>{
  const [w,h]=v.image_size||[512,512],K=v.K,n=.01;
  const m=Array.isArray(v.T_W_C)?new THREE.Matrix4().fromArray(v.T_W_C.flat()).transpose():
   new THREE.Matrix4().compose(new THREE.Vector3(...v.T_W_C.translation_m),
   new THREE.Quaternion(...v.T_W_C.quaternion_xyzw),new THREE.Vector3(1,1,1));
  m.multiply(new THREE.Matrix4().makeRotationX(Math.PI));
  m.decompose(window.camera.position,window.camera.quaternion,window.camera.scale);
  const c=window.camera;c.near=n;c.far=10;
  const cx=K[0][2]+.5,cy=K[1][2]+.5;
  c.projectionMatrix.makePerspective(-cx*n/K[0][0],(w-cx)*n/K[0][0],cy*n/K[1][1],-(h-cy)*n/K[1][1],n,10);
  c.projectionMatrixInverse.copy(c.projectionMatrix).invert();c.updateMatrixWorld(true);
  renderer.setSize(w,h);renderer.render(scene,c);
 };
 return {objects:Object.keys(world.objects),state_revision:state.revision};
})();
</script></body></html>`;
const mime={'.html':'text/html','.js':'text/javascript','.json':'application/json','.glb':'model/gltf-binary'};
const server=http.createServer((req,res)=>{
 const url=decodeURIComponent(req.url.split('?')[0]);
 if(url==='/'){res.setHeader('Content-Type','text/html');res.end(html);return;}
 const base=url.startsWith('/world/')?workspace:url.startsWith('/three/')?path.join(runtime,'.render-tools/node_modules/three'):null;
 if(!base){res.writeHead(404);res.end();return;}
 const prefix=url.startsWith('/world/')?'/world/':'/three/';
 const file=path.resolve(base,url.slice(prefix.length));
 if(!file.startsWith(base+path.sep)){res.writeHead(403);res.end();return;}
 fs.readFile(file,(error,data)=>{
  if(error){res.writeHead(404);res.end();return;}
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(data);
 });
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try{
 browser=await chromium.launch({headless:true,args:['--no-sandbox']});
 const page=await browser.newPage({viewport:{width:512,height:512}}),errors=[];
 page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:'+server.address().port+'/');
 await page.waitForFunction('window.setCamera',null,{timeout:60000});
 const info=await page.evaluate(async()=>await window.ready);
 for(const view of capture.views){
  await page.setViewportSize({width:view.image_size[0],height:view.image_size[1]});
  await page.evaluate(v=>window.setCamera(v),view);
  await page.screenshot({path:path.join(output,view.view_id+'_render.png')});
 }
 const glb=await page.evaluate(async()=>{
  const {GLTFExporter}=await import('/three/examples/jsm/exporters/GLTFExporter.js');
  const {Group}=await import('three');const standard=new Group();
  standard.rotation.x=-Math.PI/2;standard.add(window.worldRoot.clone(true));
  const buffer=await new GLTFExporter().parseAsync(standard,{binary:true});
  return await new Promise(resolve=>{const r=new FileReader();r.onload=()=>resolve(r.result.split(',')[1]);r.readAsDataURL(new Blob([buffer]));});
 });
 if(errors.length)throw new Error(errors.join('\n'));
 fs.writeFileSync(path.join(output,'codeworld_viewer_y_up.glb'),Buffer.from(glb,'base64'));
 const report={renderer:'native Three.js WebGL',...info,rendered_views:capture.views.map(v=>({view_id:v.view_id,role:v.role})),
  page_errors:errors,robot_actions_executed_by_renderer:0,scope:'calibrated visual comparison; geometry accuracy not certified'};
 fs.writeFileSync(path.join(output,'render_report.json'),JSON.stringify(report,null,2)+'\n');
 console.log(JSON.stringify(report));
}finally{if(browser)await browser.close();server.close();}
