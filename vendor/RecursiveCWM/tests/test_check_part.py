"""runner/check_part.py accepts every delivery shape the solver produces and rejects broken ones (facts only)."""
import json, subprocess, sys
from pathlib import Path
CHECK = Path(__file__).resolve().parent.parent / 'runner' / 'check_part.py'

def run(part):
    r = subprocess.run([sys.executable, str(CHECK), str(part)], capture_output=True, text=True)
    return r.returncode, r.stdout.strip()

def node(fractal, nid, part, module='component.js'):
    d = fractal / nid; d.mkdir(parents=True, exist_ok=True)
    if module: (d / module).write_text('export function build(){}\n')
    (d / 'part.json').write_text(json.dumps(part)); return d / 'part.json'

def test_module_and_export(tmp_path):
    f = tmp_path / 'fractal'
    assert run(node(f, 'scene', {'module': 'component.js', 'export': 'build', 'children': []})) == (0, 'ok module=component.js:build children=0')

def test_entry_without_export(tmp_path):
    f = tmp_path / 'fractal'
    assert run(node(f, 'scene', {'entry': 'component.js', 'children': []}))[0] == 0

def test_components_list(tmp_path):
    f = tmp_path / 'fractal'
    assert run(node(f, 'scene', {'components': [{'id': 'a'}, {'id': 'b'}]}, module=None)) == (0, 'ok components=2 children=0')
    assert run(node(f, 'dup', {'components': [{'id': 'a'}, {'id': 'a'}]}, module=None)) == (1, 'duplicate_ids')

def test_children_flat_and_nested_ids(tmp_path):
    f = tmp_path / 'fractal'
    node(f, 'cottages', {'module': 'component.js', 'export': 'build', 'children': []})
    node(f, 'east-village/farmstead/timber-barn', {'module': 'component.js', 'export': 'build', 'children': []})
    # the root names a flat child; a nested node names its nested child by the full id (as in the paper's runs)
    assert run(node(f, 'scene', {'module': 'component.js', 'export': 'build', 'children': ['cottages', 'east-village/farmstead']}))[1].startswith('child_missing')
    node(f, 'east-village/farmstead', {'module': 'component.js', 'export': 'build', 'children': ['east-village/farmstead/timber-barn']})
    assert run(f / 'east-village/farmstead/part.json') == (0, 'ok module=component.js:build children=1')
    assert run(f / 'scene/part.json') == (0, 'ok module=component.js:build children=2')

def test_children_as_paths_and_objects(tmp_path):
    f = tmp_path / 'fractal'
    node(f, 'police-station', {'module': 'component.js', 'export': 'build', 'children': []})
    node(f, 'houses', {'entry': 'component.js', 'export': 'build', 'children': []})
    p = node(f, 'scene', {'entry': 'component.js', 'export': 'build',
                          'children': [{'name': 'police-station', 'part': '../police-station/part.json'}, {'id': 'houses'}, '../houses/part.json']})
    assert run(p) == (0, 'ok module=component.js:build children=3')

def test_missing_child_and_module(tmp_path):
    f = tmp_path / 'fractal'
    assert run(node(f, 'scene', {'module': 'component.js', 'export': 'build', 'children': ['ghost']})) == (1, 'child_missing: ghost')
    assert run(node(f, 'nomod', {'module': 'missing.js', 'export': 'build'}, module=None)) == (1, 'module_missing: missing.js')
    assert run(node(f, 'empty', {'children': []}, module=None)) == (1, 'empty_components')
