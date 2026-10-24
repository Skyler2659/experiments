#!/usr/bin/env python3
"""trace/events.jsonl -> trace/tree.json + trace/recursion_report.md (the real call tree; usage = tokens summed over a node's sessions)."""
import json, sys, collections
from pathlib import Path
chain = Path(sys.argv[1])
evs = []
for l in (chain/'trace/events.jsonl').read_text().splitlines():
    if not l.strip(): continue
    try: evs.append(json.loads(l.replace('"usage_total":}','"usage_total":0}')))  # older runner wrote an empty usage field
    except Exception: pass
kids = collections.defaultdict(list)
info = {}
for e in evs:
    n = e['node_id']
    info.setdefault(n, {'depth': e['depth'], 'parent': e['parent_id'], 'sessions': 0, 'usage': 0, 'stop': None})
    if e['event'] == 'session_end':
        info[n]['sessions'] += 1
        info[n]['usage'] += e.get('usage_total') or 0   # per-session tokens, summed over the node's sessions
    if e['event'] == 'child_call' and e.get('child'):
        if e['child'] not in kids[n]: kids[n].append(e['child'])
    if e['event'] == 'stop': info[n]['stop'] = e.get('stop_reason')
roots = [n for n, i in info.items() if i['parent'] in ('-', None)]
def tree(n):
    return {'node': n, **info[n], 'children': [tree(k) for k in kids[n]]}
out = {'roots': [tree(r) for r in roots]}
(chain/'trace/tree.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
lines = ['# Recursion trace (real call tree)', '']
def emit(n, ind=0):
    i = info[n]
    lines.append(f"{'  '*ind}- solve({n}) depth={i['depth']} sessions={i['sessions']} usage≈{i['usage']} stop={i['stop']}")
    for k in kids[n]: emit(k, ind+1)
for r in roots: emit(r)
maxd = max((i['depth'] for i in info.values()), default=0)
lines += ['', f"max depth: {maxd}; nodes: {len(info)}; parents that made child calls: {sum(1 for n in kids if kids[n])}"]
(chain/'trace/recursion_report.md').write_text('\n'.join(lines))
print('\n'.join(lines))
