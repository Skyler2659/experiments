"""Protocol regression checks for the RCWM JDCloud bridge."""
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    'jdcloud_bridge', Path(__file__).resolve().parents[2] / 'scripts/jdcloud_responses_proxy.py')
BRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BRIDGE)


class BridgeTests(unittest.TestCase):
    def test_parallel_image_results_follow_all_tool_calls_before_user_images(self):
        items = [{'type': 'function_call', 'call_id': f'call_{i}', 'name': 'view_image', 'arguments': '{}'}
                 for i in range(17)]
        items += [{'type': 'function_call_output', 'call_id': f'call_{i}', 'output': [
            {'type': 'input_image', 'image_url': f'data:image/png;base64,image_{i}'}]} for i in range(17)]
        body, _ = BRIDGE.chat_request({'model': 'gpt-5', 'input': items,
            'tools': [{'type': 'function', 'name': 'view_image', 'parameters': {'type': 'object'}}]})
        messages = body['messages']
        self.assertEqual([x['role'] for x in messages], ['assistant'] + ['tool'] * 17 + ['user'])
        self.assertEqual([x['tool_call_id'] for x in messages[1:18]], [f'call_{i}' for i in range(17)])
        self.assertEqual([x['image_url']['url'] for x in messages[-1]['content']],
                         [f'data:image/png;base64,image_{i}' for i in range(17)])

    def test_custom_edit_tool_preserves_patch_text_in_both_directions(self):
        patch = '*** Begin Patch\n*** Add File: example.txt\n+hello\n*** End Patch'
        body, names = BRIDGE.chat_request({'model': 'gpt-5', 'tools': [
            {'type': 'custom', 'name': 'apply_patch', 'description': 'Edit files'}], 'input': [
            {'type': 'custom_tool_call', 'name': 'apply_patch', 'call_id': 'c1', 'input': patch},
            {'type': 'custom_tool_call_output', 'call_id': 'c1', 'output': 'Success'}]})
        args = body['messages'][0]['tool_calls'][0]['function']['arguments']
        self.assertEqual(json.loads(args)['input'], patch)
        output = BRIDGE.response_output({'choices': [{'message': {'tool_calls': [
            {'id': 'c2', 'function': {'name': 'apply_patch', 'arguments': args}}]}}]}, names)
        self.assertEqual(output[0]['type'], 'custom_tool_call')
        self.assertEqual(output[0]['input'], patch)

    def test_unknown_content_is_not_silently_discarded(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported message part'):
            BRIDGE.chat_request({'model': 'gpt-5', 'input': [
                {'role': 'user', 'content': [{'type': 'input_audio'}]}]})

    def test_anthropic_parallel_tool_results_and_images_preserve_identity(self):
        chat = {'model': 'Claude-Opus-4.8', 'reasoning_effort': 'high', 'messages': [
            {'role': 'developer', 'content': 'Instructions'},
            {'role': 'assistant', 'tool_calls': [
                {'id': 'one', 'function': {'name': 'view_image', 'arguments': '{}'}},
                {'id': 'two', 'function': {'name': 'view_image', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'one', 'content': 'First'},
            {'role': 'tool', 'tool_call_id': 'two', 'content': 'Second'},
            {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]}
        payload = BRIDGE.anthropic_request(chat)
        self.assertEqual(payload['system'], 'Instructions')
        self.assertEqual([b['id'] for b in payload['messages'][0]['content']], ['one', 'two'])
        blocks = payload['messages'][1]['content']
        self.assertEqual([b['type'] for b in blocks], ['tool_result', 'tool_result', 'image'])
        self.assertEqual([b['tool_use_id'] for b in blocks[:2]], ['one', 'two'])
        self.assertEqual(blocks[2]['source']['data'], 'YQ==')

    def test_null_tool_calls_are_a_valid_text_reply(self):
        output = BRIDGE.response_output({'choices': [{'message': {'content': 'OK', 'tool_calls': None}}]}, {})
        self.assertEqual(output[0]['content'][0]['text'], 'OK')


if __name__ == '__main__':
    unittest.main()
