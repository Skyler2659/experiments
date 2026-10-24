"""Image-only box annotation, SAM2 masks, measured evidence and public robot export."""
import sys,os,json,hashlib,shutil,base64,urllib.request,re
from pathlib import Path
import numpy as np
from PIL import Image
case=Path(sys.argv[1]);source=Path(__file__).resolve().parents[2];spec=json.loads((case/'spec.json').read_text());cap=json.loads((case/'capture/capture.json').read_text());inp=case/'input';inp.mkdir(exist_ok=True)
ids=[0,2,4,8,10,14];selected=[v for v in cap['views'] if v['view_id']=='agentview' or v['view_id'] in [f'scanner_{i:06d}' for i in ids]]
from scipy.spatial.transform import Rotation
for v in selected:
 if isinstance(v['T_W_C'],list):
  T=np.asarray(v['T_W_C'],float);v['T_W_C']={'translation_m':T[:3,3].tolist(),'quaternion_xyzw':Rotation.from_matrix(T[:3,:3]).as_quat().tolist()}
for v in selected:
 for key in ['rgb_uri','depth_uri']:shutil.copy2(case/'capture'/v[key],inp/v[key])
fit={**cap,'views':selected};(inp/'capture.json').write_text(json.dumps(fit,indent=2));shutil.copy2(inp/'ood_agentview.png' if (inp/'ood_agentview.png').exists() else inp/'agentview.png',inp/'reference.png')
objects=json.loads((case/'inventory.json').read_text())['objects'];(inp/'objects.json').write_text(json.dumps({'objects':objects},indent=2));(inp/'input.json').write_text(json.dumps({'sha256':hashlib.sha256((inp/'reference.png').read_bytes()).hexdigest(),'input_kind':'multi_view_rgbd','previous_reconstruction_used':False}))
boxesdir=case/'boxes';boxesdir.mkdir(exist_ok=True)
if not all((boxesdir/(v['view_id']+'.json')).exists() for v in selected):
 content=[{'type':'input_text','text':'Jointly annotate tight bounding boxes for all listed visible target objects in these calibrated RGB views. Each view gives its own width and height in pixels. Keep instance IDs consistent across viewpoints by matching appearance and relative arrangement. Instance IDs of indistinguishable objects may be assigned arbitrarily in the first view but MUST track the same physical item in every subsequent view. Derive boxes from RGB only, no tabletop/robot/room background. Omit fully occluded objects. Return ONLY JSON {"views":[{"view_id":"...","boxes":[{"object_id":"...","box_xyxy_px":[x0,y0,x1,y1]}],"absent":["..."]}]}. Inventory: '+json.dumps(objects)}]
 for v in selected:
  content.extend([{'type':'input_text','text':'view_id='+v['view_id']+' image_size_wh='+json.dumps(v['image_size'])},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode((inp/v['rgb_uri']).read_bytes()).decode()}])
 payload={'model':'gpt-5.6-sol','input':[{'role':'user','content':content}],'reasoning':{'effort':'max'},'stream':False}
 from robodojo_benchmark.vision_transport import request
 response=request(payload,case/'boxes_request')
 text='\n'.join(p.get('text','') for o in response.get('output',[]) for p in o.get('content',[]) if p.get('type')=='output_text');(boxesdir/'joint.response.json').write_text(json.dumps(response))
 clean=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip());packet=json.loads(clean);by={r['view_id']:r for r in packet['views']};allowed={o['object_id'] for o in objects}
 for v in selected:
  row=by[v['view_id']]
  for box in row['boxes']:
   if box['object_id'] not in allowed:raise ValueError('Unknown object ID')
   a=np.asarray(box['box_xyxy_px'],dtype=float)
   w,h=v['image_size']
   if a.shape!=(4,) or not np.isfinite(a).all() or min(a)<0 or a[0]>w or a[2]>w or a[1]>h or a[3]>h or a[2]<=a[0] or a[3]<=a[1]:raise ValueError('Invalid RGB box')
  row.update(source_rgb_sha256=v['rgb_sha256'],image_size_px=v['image_size']);(boxesdir/(v['view_id']+'.json')).write_text(json.dumps(row,indent=2))
# Reuse image-bound masks after an infrastructure-only preparation repair.
if (case/'annotations.json').exists():
 rows=json.loads((case/'annotations.json').read_text())['views']
else:
 import fcntl,torch,cv2
 from sam2.build_sam import build_sam2_hf
 from sam2.sam2_image_predictor import SAM2ImagePredictor
 with Path(os.environ.get('RCWM_GPU_LOCK','/tmp/rcwm-robodojo42-gpu.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);predictor=SAM2ImagePredictor(build_sam2_hf('facebook/sam2.1-hiera-tiny',device='cuda'));masks=case/'masks';masks.mkdir(exist_ok=True);rows=[]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for v in selected:
    predictor.set_image(np.asarray(Image.open(inp/v['rgb_uri']).convert('RGB')));row={'view_id':v['view_id'],'rgb_sha256':v['rgb_sha256'],'masks':{}}
    for b in json.loads((boxesdir/(v['view_id']+'.json')).read_text())['boxes']:
     ms,scores,_=predictor.predict(box=np.array(b['box_xyxy_px']),multimask_output=True);mask=ms[int(np.argmax(scores))];path=masks/(v['view_id']+'_'+b['object_id']+'.png')
     if mask.sum()<12:continue
     Image.fromarray(mask.astype('uint8')*255).save(path);row['masks'][b['object_id']]={'uri':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    rows.append(row)
  del predictor;torch.cuda.empty_cache()
visible={x for r in rows for x in r['masks']};missing={o['object_id'] for o in objects}-visible
if missing:raise ValueError('No RGB-derived mask for inventory objects: '+str(sorted(missing)))
ann={'views':rows,'capture_sha256':hashlib.sha256((inp/'capture.json').read_bytes()).hexdigest(),'role':'fit','holdout_used':False,'source':'Configured vision model RGB-only boxes + SAM2 image masks; no simulator segmentation'};(case/'annotations.json').write_text(json.dumps(ann,indent=2))
cfg={'capture':str(inp/'capture.json'),'annotations':str(case/'annotations.json'),'objects':objects,'regions':[],'output':str(case/'evidence')};(case/'evidence_config.json').write_text(json.dumps(cfg,indent=2))
from reconstruction_assist.evidence import run
if (case/'evidence').exists() and not (case/'evidence/model/evidence.json').exists():
 import time
 (case/'evidence').rename(case/('evidence_infrastructure_attempt_'+str(time.time_ns())))
if not (case/'evidence/model/evidence.json').exists():run(case/'evidence_config.json')
if not (inp/'measurement').exists():shutil.copytree(case/'evidence/model',inp/'measurement')
shutil.copy2(source/'reconstruction_assist/recipe_geometry.mjs',inp/'measurement/recipe_geometry.mjs')

shutil.copy2(case/'capture/known_robot.glb',inp/'known_robot.glb')
binding={'reference_sha256':hashlib.sha256((inp/'reference.png').read_bytes()).hexdigest(),'robot_node':{'asset':{'sha256':hashlib.sha256((inp/'known_robot.glb').read_bytes()).hexdigest()}},'frame':'robot_base_W','source':'public X5 URDF at measured joint states; no target geometry'}
(inp/'robot_binding.json').write_text(json.dumps(binding,indent=2))
