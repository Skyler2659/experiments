"""Linux EGL preview of current and hypothetical state, using the source camera."""
import os
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
import argparse,json
from pathlib import Path
import numpy as np
import pyrender
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from .world import World,read
p=argparse.ArgumentParser();p.add_argument('workspace');a=p.parse_args();root=Path(a.workspace)
w=World(root);report=read(root/'integration_report.json');other=World(root,report['branch'])
row=w.object(w.objects()[0]);camera=read(root/row['source_parameters'])['reference_camera']
if 'K' not in camera:
    capture=read((root/row['source_parameters']).parent/camera['source'])
    camera=next(v for v in capture['views'] if v['view_id']==camera['view_id'])
k=np.array(camera['K'])
t=np.eye(4);t[:3,:3]=Rotation.from_quat(camera['T_W_C']['quaternion_xyzw']).as_matrix();t[:3,3]=camera['T_W_C']['translation_m'];t=t@np.diag([1,-1,-1,1])
images=[]
for world,label in [(w,'CURRENT RECONSTRUCTED WORLD'),(other,'HYPOTHESIS ONLY: BOWL +10 CM X')]:
    scene=pyrender.Scene(bg_color=[.5,.55,.65,1],ambient_light=[.65,.65,.65])
    for oid in world.objects():scene.add(pyrender.Mesh.from_trimesh(world.mesh(oid),smooth=False))
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2]+.5,k[1,2]+.5,znear=.01,zfar=10),pose=t)
    scene.add(pyrender.DirectionalLight(color=np.ones(3),intensity=1.2),pose=t)
    renderer=pyrender.OffscreenRenderer(512,512);color,_=renderer.render(scene);renderer.delete()
    im=Image.new('RGB',(512,548),(24,28,32));im.paste(Image.fromarray(color[:,:,:3]),(0,36));ImageDraw.Draw(im).text((10,12),label,fill='white');images.append(im)
board=Image.new('RGB',(1024,548));board.paste(images[0],(0,0));board.paste(images[1],(512,0));board.save(root/'work/state_update_comparison.png')
print(root/'work/state_update_comparison.png')
