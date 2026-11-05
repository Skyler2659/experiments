import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE=Path(__file__).resolve().parents[2]/'src/privilege/resources/recursive_cwm/runner/node_checkpoint.py'
spec=importlib.util.spec_from_file_location('node_checkpoint',FILE)
node=importlib.util.module_from_spec(spec);spec.loader.exec_module(node)


class NodeCheckpointTest(unittest.TestCase):
    def test_only_unchanged_completed_work_is_reused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);folder=root/'fractal/child';folder.mkdir(parents=True)
            prompt=root/'prompt.md';prompt.write_text('same instructions')
            for name in ['brief.md','target.png','view.json']:(folder/name).write_text('original')
            (folder/'component.js').write_text('export function build() {}')
            (folder/'part.json').write_text('{"module":"component.js","children":[]}')
            state=node.snapshot(root,'child','parent',prompt)
            (folder/'.delivery_checkpoint.json').write_text(json.dumps(state))
            self.assertTrue(node.check(root,'child','parent',prompt))
            (folder/'brief.md').write_text('refine the geometry')
            self.assertFalse(node.check(root,'child','parent',prompt))
            (folder/'brief.md').write_text('original')
            (folder/'component.js').write_text('changed geometry')
            self.assertFalse(node.check(root,'child','parent',prompt))


if __name__=='__main__':unittest.main()
