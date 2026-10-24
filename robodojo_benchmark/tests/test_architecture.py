"""Guard the package boundaries so future fixes do not reintroduce import cycles."""
import ast
from robodojo_benchmark.catalog import ROOT


def test_dependencies_follow_harness_robot_world_direction():
    for package, forbidden in [('codeworld_workspace', {'codeworld_robot', 'robodojo_benchmark'}),
                               ('codeworld_robot', {'robodojo_benchmark'})]:
        for path in (ROOT/package).glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.level == 0 and node.module else []
                elif isinstance(node, ast.Import):modules = [a.name for a in node.names]
                else:continue
                assert not any(m.split('.')[0] in forbidden for m in modules), str(path)
