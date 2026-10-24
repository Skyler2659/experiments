"""Identify scene objects from FIT RGB through the configured model transport."""
import base64,json,re,sys,urllib.request
from pathlib import Path
case=Path(sys.argv[1]);cap=json.loads((case/'capture/capture.json').read_text());spec=json.loads((case/'spec.json').read_text())
ids={'agentview',*[f'scanner_{i:06d}' for i in [0,2,4,8,10,14]]}
content=[{'type':'input_text','text':'''Identify EVERY independently visible task object and fixture in these calibrated views of ONE frozen tabletop scene. Return only JSON {"objects":[{"object_id":"unique_snake_case","description":"concrete visual description distinguishing this instance across views","family":"visual category","texture_edge_m":0.004}],"notes":"occlusion or uncertain identities"}. Keep identical-looking instances separate when spatially distinguishable. Include receptacles, tools, supports, apparatus, racks, monitors, containers and visible cloth/liquid if present. Exclude the robot, its mounting frame, cameras, tabletop, floor and background room. Do not infer invisible items from the task name. Use RGB alone; no simulator assets or poses. Several views show the same objects: do not duplicate them. The reconstruction must cover the whole visible task scene, not only a target object. Task label for semantic context: '''+spec['task']}]
for v in cap['views']:
 if v['view_id'] not in ids:continue
 content.extend([{'type':'input_text','text':'view_id='+v['view_id']},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode((case/'capture'/v['rgb_uri']).read_bytes()).decode()}])
payload={'model':'gpt-5.6-sol','input':[{'role':'user','content':content}],'reasoning':{'effort':'max'},'stream':False}
from robodojo_benchmark.vision_transport import request
response=request(payload,case/'inventory_request')
(case/'inventory.response.json').write_text(json.dumps(response))
text='\n'.join(p.get('text','') for o in response.get('output',[]) for p in o.get('content',[]) if p.get('type')=='output_text')
packet=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip()))
assert packet['objects'] and len({o['object_id'] for o in packet['objects']})==len(packet['objects'])
for o in packet['objects']:
 assert re.fullmatch('[a-z][a-z0-9_]*',o['object_id']),o
 assert o['object_id'] not in ['known_robot','tabletop']
 o.setdefault('texture_edge_m',.004);o.setdefault('family','observed_object')
packet.update(source='FIT RGB only; configured model transport',model=response.get('model',payload['model']),holdout_used=False)
(case/'inventory.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False))
print('INVENTORY',spec['id'],len(packet['objects']),flush=True)
