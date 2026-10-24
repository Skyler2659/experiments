#!/usr/bin/env python3
"""Local Responses-to-Chat bridge for RCWM's Codex executor.

The upstream credential stays in this process; model/tool execution stays in Codex.
"""
import argparse
import concurrent.futures
import getpass
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


def chat_request(request):
    messages = []
    pending_images = []
    if request.get('instructions'):
        messages.append({'role': 'developer', 'content': request['instructions']})
    items = request.get('input', [])
    if isinstance(items, str):
        items = [{'role': 'user', 'content': items}]
    for item in items:
        kind = item.get('type', 'message')
        if pending_images and kind not in ('function_call_output', 'custom_tool_call_output'):
            messages.append({'role': 'user', 'content': pending_images})
            pending_images = []
        if kind in ('reasoning',):
            continue
        if kind in ('function_call', 'custom_tool_call'):
            arguments = item.get('arguments') if kind == 'function_call' else json.dumps({'input': item['input']})
            tool = {'id': item['call_id'], 'type': 'function', 'function': {
                'name': item['name'], 'arguments': arguments}}
            if messages and messages[-1]['role'] == 'assistant' and messages[-1].get('tool_calls'):
                messages[-1]['tool_calls'].append(tool)
            else:
                messages.append({'role': 'assistant', 'content': None, 'tool_calls': [tool]})
        elif kind in ('function_call_output', 'custom_tool_call_output'):
            output = item.get('output', '')
            images = []
            if isinstance(output, list):
                texts = []
                for part in output:
                    if part.get('type') in ('input_image', 'image_url'):
                        url = part.get('image_url')
                        images.append({'type': 'image_url', 'image_url': {'url': url} if isinstance(url, str) else url})
                    elif part.get('type') in ('input_text', 'text'):
                        texts.append(part.get('text', ''))
                    else:
                        raise ValueError('Unsupported tool-output part: ' + str(part.get('type')))
                output = '\n'.join(texts) or 'Image returned; see the following image message.'
            messages.append({'role': 'tool', 'tool_call_id': item['call_id'], 'content': output})
            pending_images.extend(images)
        elif kind == 'message':
            content = item.get('content', '')
            if isinstance(content, list):
                converted = []
                for part in content:
                    ptype = part['type']
                    if ptype in ('input_text', 'output_text', 'text'):
                        converted.append({'type': 'text', 'text': part['text']})
                    elif ptype == 'input_image':
                        image = {'url': part['image_url']}
                        if part.get('detail') in ('auto', 'low', 'high'):
                            image['detail'] = part['detail']
                        converted.append({'type': 'image_url', 'image_url': image})
                    else:
                        raise ValueError('Unsupported message part: ' + ptype)
                content = converted
            messages.append({'role': item.get('role', 'user'), 'content': content})
        else:
            raise ValueError('Unsupported input item: ' + kind)
    if pending_images:
        messages.append({'role': 'user', 'content': pending_images})
    tools, names = [], {}
    for tool in request.get('tools', []):
        kind = tool['type']
        if kind == 'tool_search':
            continue
        if kind == 'function':
            fn = {k: tool[k] for k in ('name', 'description', 'parameters') if k in tool}
        elif kind == 'custom':
            fn = {'name': tool['name'], 'description': tool.get('description', '') + '\nPass the complete tool text in input.',
                  'parameters': {'type': 'object', 'properties': {'input': {'type': 'string'}}, 'required': ['input']}}
        else:
            raise ValueError('Unsupported tool: ' + json.dumps(tool)[:6000])
        alias = re.sub(r'[^a-zA-Z0-9_-]', '_', fn['name'])
        names[alias] = (fn['name'], kind)
        fn['name'] = alias
        tools.append({'type': 'function', 'function': fn})
    # History must use the same alias as the current tool definitions.
    aliases = {original: alias for alias, (original, _) in names.items()}
    for msg in messages:
        for call in msg.get('tool_calls', []):
            call['function']['name'] = aliases.get(call['function']['name'], call['function']['name'])
    body = {'model': request['model'], 'messages': messages, 'stream': False}
    if tools:
        body['tools'] = tools
    reasoning = request.get('reasoning') or {}
    if reasoning.get('effort') in ('minimal', 'low', 'medium', 'high'):
        body['reasoning_effort'] = reasoning['effort']
    if request.get('max_output_tokens'):
        body['max_completion_tokens'] = request['max_output_tokens']
    return body, names


def anthropic_request(chat):
    """Translate the complete tool/image history, preserving tool-use identities."""
    messages, system = [], []
    for message in chat['messages']:
        role = message['role']
        content = message.get('content') or []
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        if role in ('system', 'developer'):
            system.extend(p['text'] for p in content)
            continue
        blocks = []
        if role == 'tool':
            blocks = [{'type': 'tool_result', 'tool_use_id': message['tool_call_id'], 'content': content}]
            role = 'user'
        else:
            for part in content:
                if part['type'] == 'text':
                    blocks.append(part)
                elif part['type'] == 'image_url':
                    url = part['image_url']['url']
                    if url.startswith('data:'):
                        header, data = url.split(',', 1)
                        blocks.append({'type': 'image', 'source': {'type': 'base64',
                            'media_type': header[5:].split(';')[0], 'data': data}})
                    else:
                        blocks.append({'type': 'image', 'source': {'type': 'url', 'url': url}})
                else:
                    raise ValueError('Unsupported Anthropic content: ' + str(part['type']))
            for call in message.get('tool_calls', []):
                blocks.append({'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'],
                               'input': json.loads(call['function']['arguments'])})
        if not blocks:
            continue
        if messages and messages[-1]['role'] == role:
            messages[-1]['content'].extend(blocks)
        else:
            messages.append({'role': role, 'content': blocks})
    result = {'model': chat['model'], 'messages': messages, 'max_tokens': chat.get('max_completion_tokens', 32768),
              'stream': False}
    if system:
        result['system'] = '\n\n'.join(system)
    if chat.get('tools'):
        result['tools'] = [{'name': t['function']['name'], 'description': t['function'].get('description', ''),
                            'input_schema': t['function']['parameters']} for t in chat['tools']]
    # Adaptive thinking does not need opaque thinking blocks replayed through the Codex wire format.
    if chat.get('reasoning_effort'):
        result['thinking'] = {'type': 'adaptive'}
        result['output_config'] = {'effort': chat['reasoning_effort']}
    return result


def anthropic_completion(result):
    text, calls = [], []
    for block in result['content']:
        if block['type'] == 'text':
            text.append(block['text'])
        elif block['type'] == 'tool_use':
            calls.append({'id': block['id'], 'type': 'function', 'function': {
                'name': block['name'], 'arguments': json.dumps(block['input'])}})
    usage = result.get('usage', {})
    cached = usage.get('cache_read_input_tokens', 0)
    input_tokens = usage.get('input_tokens', 0) + cached + usage.get('cache_creation_input_tokens', 0)
    output_tokens = usage.get('output_tokens', 0)
    return {'model': result.get('model'), 'choices': [{'message': {'content': '\n'.join(text), 'tool_calls': calls}}],
            'usage': {'prompt_tokens': input_tokens, 'completion_tokens': output_tokens,
                      'total_tokens': input_tokens + output_tokens, 'prompt_tokens_details': {'cached_tokens': cached}}}


def response_output(completion, names):
    message = completion['choices'][0]['message']
    output = []
    if message.get('content'):
        output.append({'id': 'msg_' + uuid.uuid4().hex, 'type': 'message', 'role': 'assistant', 'status': 'completed',
                       'content': [{'type': 'output_text', 'text': message['content'], 'annotations': []}]})
    for call in message.get('tool_calls') or []:
        fn = call['function']
        name, kind = names[fn['name']]
        item = {'id': 'fc_' + uuid.uuid4().hex, 'type': 'function_call' if kind == 'function' else 'custom_tool_call',
                'call_id': call['id'], 'name': name, 'status': 'completed'}
        if kind == 'function':
            item['arguments'] = fn['arguments']
        else:
            item['input'] = json.loads(fn['arguments'])['input']
        output.append(item)
    return output


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path == '/health':
            data = b'{"status":"ready"}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def event(self, kind, **values):
        self.sequence += 1
        data = {'type': kind, 'sequence_number': self.sequence, **values}
        self.wfile.write(('event: ' + kind + '\ndata: ' + json.dumps(data) + '\n\n').encode())
        self.wfile.flush()

    def do_POST(self):
        if self.path.rstrip('/') != '/v1/responses':
            self.send_error(404)
            return
        if self.headers.get('Authorization') != 'Bearer rcwm-local':
            self.send_error(401)
            return
        streaming = False
        self.sequence = -1
        started = time.monotonic()
        try:
            req = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if req['model'] == 'gpt-5.6-sol':
                # JDCloud requires native Responses for GPT-5.6 reasoning plus tools.
                req['tools'] = [t for t in req.get('tools', []) if t['type'] != 'tool_search']
                # Gateway requests can land on different Azure resources. Replay explicit
                # messages and tool results; resource-local item IDs/encrypted reasoning
                # cannot be referenced on the next backend.
                if isinstance(req.get('input'), list):
                    req['input'] = [{k: v for k, v in item.items() if k != 'id'}
                                    for item in req['input'] if item.get('type') != 'reasoning']
                req.pop('previous_response_id', None)
                req['store'] = False
                req['include'] = []
                request = urllib.request.Request(self.server.upstream + '/responses', data=json.dumps(req).encode(),
                    headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.server.key})
                with self.server.inference_slots, urllib.request.urlopen(request, timeout=600) as result:
                    deadline = time.monotonic() + 600
                    streaming = req.get('stream', False)
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream' if streaming else 'application/json')
                    self.end_headers()
                    if streaming:
                        final = None
                        for line in result:
                            if time.monotonic() > deadline:
                                raise TimeoutError('Native Responses request exceeded 600 seconds')
                            self.wfile.write(line)
                            self.wfile.flush()
                            if line.startswith(b'data: {'):
                                event = json.loads(line[6:])
                                self.sequence = max(self.sequence, event.get('sequence_number', -1))
                                if event.get('type') == 'response.completed':
                                    final = event['response']
                        if final is None:
                            raise ValueError('Native upstream stream ended without response.completed')
                    else:
                        final = json.load(result)
                        self.wfile.write(json.dumps(final).encode())
                    self.server.record({'status': 'completed', 'model': final.get('model'), 'native_responses': True,
                        'elapsed_s': round(time.monotonic()-started, 2), 'usage': final.get('usage'),
                        'tool_calls': [x.get('name') for x in final.get('output', []) if 'name' in x]})
                return
            payload, names = chat_request(req)
            response = {'id': 'resp_' + uuid.uuid4().hex, 'object': 'response', 'created_at': int(time.time()),
                        'status': 'in_progress', 'model': req['model'], 'output': [], 'error': None}
            streaming = req.get('stream', False)
            if streaming:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()
                self.sequence = -1
                self.event('response.created', response=response)
                self.event('response.in_progress', response=response)
            def upstream():
                claude = req['model'].startswith('Claude-')
                url = self.server.upstream.removesuffix('/v1') + '/anthropic/v1/messages' if claude else self.server.upstream + '/chat/completions'
                body = anthropic_request(payload) if claude else payload
                headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.server.key}
                if claude:
                    headers['anthropic-version'] = '2023-06-01'
                with self.server.inference_slots:
                    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
                    for attempt in range(3):
                        try:
                            with urllib.request.urlopen(request, timeout=600) as result:
                                value = json.load(result)
                            break
                        except urllib.error.HTTPError as error:
                            detail = error.read().decode(errors='replace')
                            if attempt == 2 or not (error.code in (429, 500, 502, 503, 504) or 'request failed (request id:' in detail):
                                raise ValueError('Upstream HTTP ' + str(error.code) + ': ' + detail)
                            time.sleep(2 ** attempt)
                    return anthropic_completion(value) if claude else value
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                task = pool.submit(upstream)
                while True:
                    try:
                        result = task.result(timeout=15)
                        break
                    except concurrent.futures.TimeoutError:
                        if streaming:
                            self.wfile.write(b': keepalive\n\n')
                            self.wfile.flush()
            output = response_output(result, names)
            usage = result.get('usage', {})
            response.update(status='completed', model=result.get('model', req['model']), output=output,
                usage={'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0),
                       'total_tokens': usage.get('total_tokens', 0),
                       'input_tokens_details': {'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0)},
                       'output_tokens_details': {'reasoning_tokens': (usage.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)}})
            if streaming:
                for index, item in enumerate(output):
                    initial = {**item, 'status': 'in_progress'}
                    for key in ('arguments', 'input'):
                        if key in initial:
                            initial[key] = ''
                    if item['type'] == 'message':
                        initial['content'] = []
                    self.event('response.output_item.added', output_index=index, item=initial)
                    if item['type'] == 'message':
                        part = item['content'][0]
                        self.event('response.content_part.added', item_id=item['id'], output_index=index, content_index=0,
                                   part={**part, 'text': ''})
                        self.event('response.output_text.delta', item_id=item['id'], output_index=index, content_index=0, delta=part['text'])
                        self.event('response.output_text.done', item_id=item['id'], output_index=index, content_index=0, text=part['text'])
                        self.event('response.content_part.done', item_id=item['id'], output_index=index, content_index=0, part=part)
                    else:
                        field = 'arguments' if item['type'] == 'function_call' else 'input'
                        prefix = 'response.function_call_arguments' if field == 'arguments' else 'response.custom_tool_call_input'
                        self.event(prefix + '.delta', item_id=item['id'], output_index=index, delta=item[field])
                        self.event(prefix + '.done', item_id=item['id'], output_index=index, **{field: item[field]})
                    self.event('response.output_item.done', output_index=index, item=item)
                self.event('response.completed', response=response)
            else:
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            self.server.record({'status': 'completed', 'model': response['model'], 'elapsed_s': round(time.monotonic()-started, 2),
                                'tool_calls': [x.get('name') for x in output if 'name' in x], 'usage': response['usage']})
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError):
                detail = error.read().decode(errors='replace')
            else:
                detail = str(error)
            detail = detail.replace(self.server.key, '[REDACTED]')
            self.server.record({'status': 'error', 'requested_model': locals().get('req', {}).get('model'), 'error': detail})
            try:
                if streaming:
                    self.event('error', message=detail, code='upstream_error')
                else:
                    self.send_error(502, detail[:300])
            except (BrokenPipeError, ConnectionResetError):
                pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8769)
    parser.add_argument('--upstream', default='https://modelservice.jdcloud.com/v1')
    parser.add_argument('--log', required=True)
    parser.add_argument('--key-stdin', action='store_true', help='Read the upstream key from one stdin line')
    args = parser.parse_args()
    key = sys.stdin.readline().strip() if args.key_stdin else getpass.getpass('JDCloud API key: ').strip()
    if not key:
        raise ValueError('Provide upstream key on stdin')
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.key, server.upstream = key, args.upstream.rstrip('/')
    server.inference_slots = threading.BoundedSemaphore(8)
    def record(row):
        with log.open('a') as stream:
            stream.write(json.dumps({'time': time.time(), **row}) + '\n')
        print(json.dumps(row), flush=True)
    server.record = record
    print('RCWM JDCloud bridge ready on 127.0.0.1:' + str(args.port), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
