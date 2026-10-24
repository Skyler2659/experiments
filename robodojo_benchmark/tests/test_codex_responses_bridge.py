import base64,json
import pytest
from robodojo_benchmark.codex_responses_bridge import prepare,response

def test_preserves_history_and_extracts_images(tmp_path):
 p={'model':'gpt-6-astra','reasoning':{'effort':'high'},'input':[{'role':'user','content':[{'type':'input_text','text':'native prompt'},{'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(b'png').decode()}]},{'type':'function_call_output','call_id':'c1','output':'original receipt'}],'tools':[{'type':'function','name':'native_tool','parameters':{'type':'object'}}]}
 original=json.dumps(p);prompt,images=prepare(p,tmp_path)
 assert json.dumps(p)==original
 assert 'original receipt' in prompt and 'native prompt' in prompt and 'native_tool' in prompt
 assert len(images)==1 and (tmp_path/'image_000.png').read_bytes()==b'png'

def test_rejects_extra_tool_and_non_object_arguments():
 p={'model':'gpt-6-astra','tools':[{'type':'function','name':'native_tool'}]}
 for name,args in [('shell','{}'),('native_tool','[]')]:
  with pytest.raises(ValueError):response({'kind':'function_call','name':name,'arguments':args,'text':''},p)
 r=response({'kind':'function_call','name':'native_tool','arguments':'{"x":1}','text':''},p)
 assert r['output'][0]['arguments']=='{"x":1}' and r['output'][0]['name']=='native_tool'
