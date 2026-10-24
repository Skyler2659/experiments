"""Measured support texture and known robot context for the SAME MeshCodeWorld.

Extracts G01 table projection/foreground exclusion into a scene-independent
function. No previous tabletop asset, task mesh or original object pose is read.
"""
from pathlib import Path
import json
import numpy as np
import cv2
import trimesh
from PIL import Image
from .mesh_world import pose_matrix


def bake_support(views, root, lower, upper, z, output, size=1024):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x,y=np.meshgrid(np.linspace(lower[0],upper[0],size),np.linspace(upper[1],lower[1],size))
    points=np.c_[x.ravel(),y.ravel(),np.full(size*size,z)]
    total=np.zeros((size,size,3));count=np.zeros((size,size));used=[]
    for view in views:
        if not view['view_id'].startswith('wrist_'):continue
        rgb=cv2.cvtColor(cv2.imread(str(root/view['rgb_uri'])),cv2.COLOR_BGR2RGB)
        h,w=rgb.shape[:2];r,g,b=[rgb[:,:,i].astype(float) for i in range(3)]
        valid=((r-g>5)&(g-b>5)&(r>110)&(r<245)&(r-b<90)).astype(np.uint8)
        valid[int(h*.75):]=0
        for uri in view.get('masks',{}).values():
            mask=cv2.imread(str(root/uri),cv2.IMREAD_GRAYSCALE)
            valid[cv2.dilate(mask,np.ones((21,21),np.uint8))>0]=0
        t=np.linalg.inv(pose_matrix(view['T_W_C']));camera=(t[:3,:3]@points.T).T+t[:3,3]
        k=np.array(view['K']);uv=camera@k.T;uv=uv[:,:2]/uv[:,2:]
        u=uv[:,0].reshape(size,size).astype(np.float32);v=uv[:,1].reshape(size,size).astype(np.float32)
        take=cv2.remap(valid,u,v,cv2.INTER_NEAREST)>0;take &= camera[:,2].reshape(size,size)>.01
        colors=cv2.remap(rgb,u,v,cv2.INTER_LINEAR)
        total[take]+=colors[take];count[take]+=1;used.append(view['view_id'])
    known=count>0
    if known.mean()<.03: raise ValueError('Insufficient observed table pixels for texture')
    texture=total/np.maximum(count[:,:,None],1)
    # Blend missing interior texels from their observed neighbours. Directly
    # stamping a repeated patch left object-mask-shaped seams on the support.
    filled=cv2.inpaint(np.uint8(np.clip(texture,0,255)),np.uint8(~known)*255,7,cv2.INPAINT_TELEA)
    texture[~known]=filled[~known]
    texture=texture-cv2.GaussianBlur(texture,(0,0),24)+np.median(texture[known],axis=0)
    image=Image.fromarray(np.uint8(np.clip(texture,0,255)));image.save(output/'table_rgb.png')
    Image.fromarray(np.uint8(known)*255).save(output/'observed_mask.png')
    (output/'texture_audit.json').write_text(json.dumps({'source':'current observed wrist RGB projected on measured support','views':used,'observed_fraction':float(known.mean()),'unobserved_fill':'neighbour-based image inpainting, not measured','support_z_m':z},indent=2))
    mesh=trimesh.creation.box([upper[0]-lower[0],upper[1]-lower[1],.02]);mesh.apply_translation([(lower[0]+upper[0])/2,(lower[1]+upper[1])/2,z-.01])
    uv=np.c_[(mesh.vertices[:,0]-lower[0])/(upper[0]-lower[0]),(mesh.vertices[:,1]-lower[1])/(upper[1]-lower[1])]
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=uv,material=trimesh.visual.material.PBRMaterial(baseColorTexture=image,metallicFactor=0,roughnessFactor=1))
    return mesh


def public_robot_mesh(npz):
    data=np.load(npz);inventory=json.loads(str(data['inventory_json']));pieces=[]
    face_cursor=0
    for item in inventory['geometries']:
        gid=item['id'];kind=item['kind'];s=data[f'size_{gid}']
        if kind=='mjGEOM_MESH':mesh=trimesh.Trimesh(data[f'vertices_{gid}'],data[f'faces_{gid}'],process=False)
        elif kind=='mjGEOM_BOX':mesh=trimesh.creation.box(s*2)
        elif kind=='mjGEOM_SPHERE':mesh=trimesh.creation.icosphere(subdivisions=2,radius=s[0])
        elif kind in ('mjGEOM_CYLINDER','mjGEOM_CAPSULE'):mesh=trimesh.creation.cylinder(radius=s[0],height=s[1]*2)
        else:continue
        mesh.apply_transform(data[f'pose_{gid}']);mesh.visual.vertex_colors=np.uint8(np.clip(data[f'color_{gid}']*255,0,255));pieces.append(mesh)
        item['source_face_range']=[face_cursor,face_cursor+len(mesh.faces)];face_cursor+=len(mesh.faces)
    return trimesh.util.concatenate(pieces),inventory
