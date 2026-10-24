// Numerical recipe parameters are the geometry source. Matrices are row-major W transforms.
import * as THREE from 'three';
export function recipeGeometry(recipe, material=new THREE.MeshStandardMaterial({color:0x999999,roughness:1,side:THREE.DoubleSide})) {
 const group=new THREE.Group();
 for(const p of recipe.parts){
  let geometry;
  if(p.kind==='box') geometry=new THREE.BoxGeometry(...p.size_m);
  else if(p.kind==='lathe'){
   geometry=new THREE.LatheGeometry(p.profile_radius_z_m.map(([r,z])=>new THREE.Vector2(r,z)),96);
   geometry.rotateX(Math.PI/2); // Three lathe Y axis -> world/local Z.
  }else if(p.kind==='field_slab'){
   const x=p.x_m,y=p.y_m,z=p.top_z_m,nx=x.length,ny=y.length,n=nx*ny,v=[],f=[];
   for(let layer=0;layer<2;layer++)for(let j=0;j<ny;j++)for(let i=0;i<nx;i++)v.push(x[i],y[j],z[j][i]-layer*p.thickness_m);
   for(let j=0;j<ny-1;j++)for(let i=0;i<nx-1;i++){const a=j*nx+i,b=a+1,c=a+nx,d=c+1;f.push(a,b,d,a,d,c,a+n,d+n,b+n,a+n,c+n,d+n);}
   const ring=[];for(let i=0;i<nx;i++)ring.push(i);for(let j=1;j<ny;j++)ring.push(j*nx+nx-1);for(let i=n-2;i>=n-nx;i--)ring.push(i);for(let j=ny-2;j>0;j--)ring.push(j*nx);
   for(let i=0;i<ring.length;i++){const a=ring[i],b=ring[(i+1)%ring.length];f.push(a,b+n,b,a,a+n,b+n);}
   geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(v,3));geometry.setIndex(f);geometry.computeVertexNormals();
  }else throw new Error('Unsupported recipe kind '+p.kind);
  const mesh=new THREE.Mesh(geometry,material);mesh.applyMatrix4(new THREE.Matrix4().fromArray(p.T_W_part.flat()).transpose());group.add(mesh);
 }
 return group;
}
