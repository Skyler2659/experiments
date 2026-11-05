"""Identify the actual imported implementation in every run receipt."""
import hashlib
from pathlib import Path


def implementation_receipt():
    root = Path(__file__).resolve().parent
    paths = sorted(root.glob('*.py')) + sorted((root/'resources/fixed_obstacles').glob('*.json'))
    versions = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    digest = hashlib.sha256('\n'.join(f'{name}:{value}' for name, value in versions.items()).encode()).hexdigest()
    return {'bundle_id': digest, 'module_root': str(root), 'files': versions}
