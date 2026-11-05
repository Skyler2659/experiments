"""Reuse a validated node only while its assigned work and delivered files match."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(chain, node, parent, prompt, visited=frozenset()):
    if node in visited: raise ValueError("Cyclic child graph")
    visited = visited | {node}
    chain = Path(chain); folder = chain / 'fractal' / node
    request = chain / 'fractal' / parent / 'refinement_requests.json'
    refinement = json.loads(request.read_text()) if request.is_file() else None
    if isinstance(refinement, dict): refinement = refinement.get(node)
    inputs = {name: digest(folder/name) for name in ['brief.md', 'target.png', 'view.json']}
    inputs.update(prompt=digest(Path(prompt)), refinement=refinement)
    part = json.loads((folder/'part.json').read_text())
    files = {'part.json': digest(folder/'part.json')}
    for p in folder.rglob('*'):
        if p.is_file() and (p.suffix in {'.js', '.mjs', '.glb', '.png', '.jpg', '.jpeg'} or p.name == 'parameters.json'):
            # Render-only output does not alter the delivered component.
            if any(x in p.relative_to(folder).parts for x in ['review', 'renders', 'screenshots']): continue
            files[str(p.relative_to(folder))] = digest(p)
    if not (folder/part['module']).is_file(): raise ValueError('Missing component module')
    children = {child: snapshot(chain,child,node,prompt,visited) for child in part.get('children', [])}
    return {'inputs': inputs, 'files': files, 'children': children}


def check(chain, node, parent, prompt):
    folder = Path(chain)/'fractal'/node
    try:
        return json.loads((folder/'.delivery_checkpoint.json').read_text()) == snapshot(chain,node,parent,prompt)
    except (OSError, ValueError, KeyError):
        return False


if __name__ == '__main__':
    mode, chain, node, parent, prompt = sys.argv[1:]
    if mode == 'check': sys.exit(0 if check(chain,node,parent,prompt) else 1)
    data = snapshot(chain,node,parent,prompt)
    (Path(chain)/'fractal'/node/'.delivery_checkpoint.json').write_text(json.dumps(data,indent=2))
