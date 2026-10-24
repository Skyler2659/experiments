"""Small CLI for native coding agents; no agent harness."""
import argparse,json
from .world import World
p=argparse.ArgumentParser();p.add_argument('workspace');p.add_argument('--branch')
s=p.add_subparsers(dest='command',required=True)
s.add_parser('inspect');q=s.add_parser('object');q.add_argument('id')
s.add_parser('audit-binding')
q=s.add_parser('bounds');q.add_argument('id')
q=s.add_parser('export');q.add_argument('output')
q=s.add_parser('update');q.add_argument('observation');q.add_argument('--expected-revision',type=int)
a=p.parse_args();w=World(a.workspace,a.branch)
if a.command=='inspect': result={'objects':w.objects(),'state_revision':w.state()['revision']}
elif a.command=='audit-binding': result=w.verify_binding()
elif a.command=='object': result=w.object(a.id)
elif a.command=='bounds': result=w.bounds(a.id)
elif a.command=='export': result=w.export_scene(a.output)
else: result={'state_revision':w.update_observed(a.observation,a.expected_revision)}
print(json.dumps(result,indent=2))
