// Build a delivered node directly; do not trust its hand-authored camera or viewer.
import fs from 'node:fs';import path from 'node:path';import http from 'node:http';
const [runtimeArg,name,node,captureArg,outArg]=process.argv.slice(2);
const runtime=path.resolve(runtimeArg),out=path.resolve(outArg),capture=JSON.parse(fs.readFileSync(captureArg));
const {chromium}=await import(path.join(runtime,'.render-tools/node_modules/playwright/index.mjs'));
fs.mkdirSync(out,{recursive:true});
const html=`<html><head><script type="importmap">{"imports":{"three":"/.render-tools/node_modules/three/build/three.module.js"}}</script></head><body style="margin:0"><script type="module">
import * as THREE from 'three';
import {build} from '/runs/${name}/fractal/${node}/component.js';
window.ready=(async()=>{
 const scene=window.scene=new THREE.Scene();scene.background=new THREE.Color(0x8794aa);const component=build();if(!component?.isObject3D)throw Error('build() must return Object3D synchronously');scene.add(component);
 scene.add(new THREE.AmbientLight(0xffffff,1.25));const light=new THREE.DirectionalLight(0xffffff,1.4);light.position.set(1,-1,2);scene.add(light);
 const renderer=new THREE.WebGLRenderer({antialias:true});renderer.setPixelRatio(1);document.body.appendChild(renderer.domElement);const c=new THREE.PerspectiveCamera();
 window.render=v=>{const [w,h]=v.image_size,K=v.K,n=.01;const m=new THREE.Matrix4().compose(new THREE.Vector3(...v.T_W_C.translation_m),new THREE.Quaternion(...v.T_W_C.quaternion_xyzw),new THREE.Vector3(1,1,1));m.multiply(new THREE.Matrix4().makeRotationX(Math.PI));m.decompose(c.position,c.quaternion,c.scale);c.projectionMatrix.makePerspective(-(K[0][2]+.5)*n/K[0][0],(w-K[0][2]-.5)*n/K[0][0],(K[1][2]+.5)*n/K[1][1],-(h-K[1][2]-.5)*n/K[1][1],n,10);c.projectionMatrixInverse.copy(c.projectionMatrix).invert();c.updateMatrixWorld();renderer.setSize(w,h);renderer.render(scene,c);};
 scene.updateMatrixWorld(true);return true;
})();</script></body></html>`;
const server=http.createServer((req,res)=>{const u=decodeURIComponent(req.url.split('?')[0]);if(u==='/'){res.setHeader('Content-Type','text/html');res.end(html);return;}const f=path.resolve(runtime,'.'+u);if(![path.join(runtime,'runs',name)+path.sep,path.join(runtime,'.render-tools/node_modules/three')+path.sep].some(p=>f.startsWith(p))){res.writeHead(403);res.end();return;}fs.readFile(f,(err,b)=>{if(err){res.writeHead(404);res.end();return;}res.setHeader('Content-Type',f.endsWith('.js')||f.endsWith('.mjs')?'text/javascript':f.endsWith('.json')?'application/json':'application/octet-stream');res.end(b);});});
await new Promise(r=>server.listen(0,'127.0.0.1',r));let browser;
try{
 browser=await chromium.launch({headless:true,args:['--no-sandbox']});const page=await browser.newPage(),errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:'+server.address().port+'/');await page.waitForFunction('window.ready',null,{timeout:30000});await page.evaluate(()=>window.ready);
 const audit=await page.evaluate(()=>{const objects={},parts={},export_issues=[];function triangles(o){let count=0;o.traverse(n=>{if(n.isMesh)count+=(n.geometry?.index?.count??n.geometry?.attributes?.position?.count??0)/3;});return count;}window.scene.traverse(o=>{if(o.isMesh){for(const m of (Array.isArray(o.material)?o.material:[o.material])){if(m && (m.isShaderMaterial||Object.hasOwn(m,'onBeforeCompile'))){let owner=o,part=o;while(owner&&!owner.userData.object_id)owner=owner.parent;while(part&&!part.userData.part_id)part=part.parent;export_issues.push({object_id:owner?.userData.object_id??null,part_id:part?.userData.part_id??null,material:m.name||m.type,reason:'custom shader is not represented by GLB standard material'});}}}const id=o.userData.object_id,pid=o.userData.part_id;if(id){if(objects[id])throw Error('Duplicate object ID');if(!triangles(o))throw Error('Empty object geometry: '+id);objects[id]=o.matrixWorld.toArray();}if(pid){if(parts[pid])throw Error('Duplicate part ID: '+pid);const n=triangles(o);if(!n)throw Error('Empty subpart geometry: '+pid);parts[pid]={triangles:n};}});return {objects,parts,export_issues};});
 const objects=audit.objects;
 if(!Object.keys(objects).length)throw Error('No tagged object roots');
 for(const [id,matrix] of Object.entries(objects)){
  if(!/^[\w-]+$/.test(id))throw Error('Invalid object ID');
  const data=await page.evaluate(async id=>{const {GLTFExporter}=await import('/.render-tools/node_modules/three/examples/jsm/exporters/GLTFExporter.js');let o;window.scene.traverse(x=>{if(x.userData.object_id===id)o=x;});const c=o.clone(true);o.matrixWorld.decompose(c.position,c.quaternion,c.scale);c.updateMatrixWorld(true);const b=await new GLTFExporter().parseAsync(c,{binary:true});return await new Promise(r=>{const f=new FileReader();f.onload=()=>r(f.result.split(',')[1]);f.readAsDataURL(new Blob([b]));});},id);
  fs.writeFileSync(path.join(out,id+'.glb'),Buffer.from(data,'base64'));
 }
 for(const v of capture.views.filter(v=>v.role==='fit'&&v.surface_enabled!==false).filter((v,i)=>i%7===0)){
  await page.setViewportSize({width:v.image_size[0],height:v.image_size[1]});await page.evaluate(v=>window.render(v),v);await page.screenshot({path:path.join(out,v.view_id+'_render.png')});
 }
 if(errors.length)throw Error(errors.join('\n'));fs.writeFileSync(path.join(out,'render.json'),JSON.stringify({objects,part_inventory:audit.parts,material_export_issues:audit.export_issues,page_errors:errors},null,2));
}finally{if(browser)await browser.close();server.close();}
