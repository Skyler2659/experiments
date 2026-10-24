"""Explicit Codex CLI transport adaptation for Responses-only robot policies.

This is not a wire-equivalent Responses implementation: original messages and
function schemas are encoded in a JSON decision prompt. No robot tools execute
inside this bridge. The released policy parses and executes returned calls.
"""
from __future__ import annotations
import argparse, base64, json, os, subprocess, tempfile, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from .worker import fresh_codex_home, stop_process, runtime_config

SCHEMA={"type":"object","properties":{"kind":{"type":"string","enum":["function_call","message"]},"name":{"type":"string"},"arguments":{"type":"string"},"text":{"type":"string"}},"required":["kind","name","arguments","text"],"additionalProperties":False}

def prepare(payload, directory):
    """Preserve the entire request history/schema; extract only image bytes."""
    if payload.get('model') != 'gpt-6-astra' or payload.get('reasoning',{}).get('effort') != 'high':
        raise ValueError('This validation bridge requires gpt-6-astra / high')
    p=json.loads(json.dumps(payload));images=[]
    def walk(x):
        if isinstance(x,dict):
            if x.get('type')=='input_image':
                uri=x['image_url']
                if not isinstance(uri,str) or not uri.startswith('data:image/') or ';base64,' not in uri:raise ValueError('Only inline image inputs are allowed')
                media,data=uri.split(';base64,',1);ext={'data:image/png':'.png','data:image/jpeg':'.jpg','data:image/webp':'.webp'}.get(media)
                if not ext:raise ValueError('Unsupported image format')
                path=directory/('image_%03d'%len(images)+ext);path.write_bytes(base64.b64decode(data,validate=True));images.append(str(path));x['image_url']=f'[attached image {len(images)}]'
            else:
                for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
    walk(p)
    prompt=('You are the model decision component of the original RoboProbe policy. '
            'The JSON below contains its original conversation and available function schemas. '
            'Follow that conversation and choose the next response. Images are attached in numbered order. '
            'Do not use shell, files, web, or any tools of this CLI. Do not execute robot actions here. '
            'Return one JSON object matching the output schema: for a function call set kind=function_call, '
            'name to an original function name, arguments to its JSON argument string, text=""; '
            'for a text response set kind=message, name="", arguments="", and text to your reply. '
            'Choose at most one function call. Do not add new robot tools or change the supplied task.\n\n'
            +json.dumps(p,ensure_ascii=False))
    return prompt,images

def response(decision,payload):
    if decision['kind']=='function_call':
        allowed={t['name'] for t in payload.get('tools',[]) if t.get('type')=='function'}
        if decision['name'] not in allowed:raise ValueError('Model selected an undeclared function')
        if not isinstance(json.loads(decision['arguments']),dict):raise ValueError('Tool arguments must be an object')
        output=[dict(type='function_call',id='fc_'+uuid.uuid4().hex,call_id='call_'+uuid.uuid4().hex,name=decision['name'],arguments=decision['arguments'],status='completed')]
    elif decision['kind']=='message':
        output=[dict(type='message',role='assistant',status='completed',id='msg_'+uuid.uuid4().hex,content=[dict(type='output_text',text=decision['text'],annotations=[])])]
    else:raise ValueError('Invalid response kind')
    return dict(id='resp_'+uuid.uuid4().hex,object='response',created_at=int(time.time()),status='completed',model=payload['model'],output=output,usage=None)

def invoke(payload,runtime,root):
    directory=root/uuid.uuid4().hex;directory.mkdir(parents=True)
    prompt,images=prepare(payload,directory)
    (directory/'request.json').write_text(json.dumps(payload))
    (directory/'prompt.txt').write_text(prompt)
    (directory/'schema.json').write_text(json.dumps(SCHEMA))
    (directory/'AGENTS.md').write_text('This directory is the independent RoboProbe model decision executor, not the desktop supervisor. The ancestor supervisor restriction does not apply to this executor. Return the requested structured decision only. No CLI tools, filesystem inspection, or direct robot execution.\n')
    with fresh_codex_home(runtime) as home:
        env={**os.environ,**runtime.get('env',{}),'CODEX_HOME':str(home)}
        cmd=[runtime['codex'],'exec','--ephemeral','--json','--skip-git-repo-check','-C',str(directory),'-m',payload['model'],'-s','read-only','-c','approval_policy="never"','-c','model_reasoning_effort="high"','-c','web_search="disabled"','--output-schema',str(directory/'schema.json'),'-o',str(directory/'final.json')]
        for feature in ['memories','multi_agent','plugins','apps','chronicle','skill_search']:cmd+=['--disable',feature]
        cmd+=['--enable','skip_host_skill_discovery']
        for image in images:cmd+=['--image',image]
        cmd+=['--','-']
        with (directory/'events.jsonl').open('w') as log,(directory/'stderr.log').open('w') as errors:
            p=subprocess.Popen(cmd,env=env,stdin=subprocess.PIPE,stdout=log,stderr=errors,start_new_session=True,text=True)
            try:
                p.communicate(input=prompt,timeout=85)
                if p.returncode:raise RuntimeError('Codex CLI failed; inspect transport trace')
            finally:stop_process(p)
    # A CLI tool use would change the baseline information/action interface.
    for line in (directory/'events.jsonl').read_text().splitlines():
        event=json.loads(line);item=event.get('item',{})
        if item.get('type') in ('command_execution','mcp_tool_call','web_search','file_change'):raise RuntimeError('CLI attempted a tool; adapted baseline request rejected')
    result=response(json.loads((directory/'final.json').read_text()),payload)
    (directory/'response.json').write_text(json.dumps(result));return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--trace',required=True);p.add_argument('--port',type=int,default=8789);a=p.parse_args()
    runtime=runtime_config(a.runtime);root=Path(a.trace).resolve();root.mkdir(parents=True,exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            if self.path!='/v1/responses':self.send_error(404);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<50_000_000:raise ValueError('Invalid payload size')
                payload=json.loads(self.rfile.read(n));result=invoke(payload,runtime,root);code=200
            except Exception as e:
                result={'error':{'message':str(e),'type':'codex_transport_error'}};code=502
            data=json.dumps(result).encode();self.send_response(code);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers()
            try:self.wfile.write(data)
            except BrokenPipeError:pass
    print('Codex decision bridge listening on localhost only',flush=True)
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
