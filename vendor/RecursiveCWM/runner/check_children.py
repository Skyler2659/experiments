#!/usr/bin/env python3
"""Validate a delegation before launching any child with an ambiguous scope."""
import json
from pathlib import Path
import re
import sys


def validate(path,node,parent):
    path=Path(path);children=json.loads(path.read_text())
    if not isinstance(children,list):raise ValueError('children.json must be a JSON list, not an object')
    if not all(isinstance(c,str) and re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]*',c) and c not in (node,parent) for c in children):raise ValueError('Invalid child ID or recursion cycle')
    if len(children)!=len(set(children)):raise ValueError('Duplicate children')
    missing=[str(path.parent.parent/c/name) for c in children for name in ('brief.md','view.json','target.png') if not (path.parent.parent/c/name).is_file()]
    if missing:raise ValueError('Missing flat child input files; create them before delegating: '+', '.join(missing))
    return children

if __name__=='__main__':
    try:print(' '.join(validate(*sys.argv[1:4])))
    except (ValueError,OSError) as error:print(str(error),file=sys.stderr);sys.exit(1)
