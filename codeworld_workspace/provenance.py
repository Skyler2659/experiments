"""Best-effort provenance diagnostics for agent-authored Python computations.

This is a lint, not an authorization boundary or a physical reasoning proof. Policy
constants, ordinary helpers and output construction must not be rejected because
this small analyzer cannot understand them. Runtime CodeWorld queries and command
binding provide the execution record; observation updates validate measurements.
"""
import ast
from dataclasses import dataclass, field

PHYSICAL = {'position', 'bounds', 'mesh', 'transform', 'object', 'state', 'measurement', 'proprioception'}


@dataclass
class Value:
    deps: set = field(default_factory=set)
    fields: dict = field(default_factory=dict)

    def all(self):
        return self.deps | set().union(*(v.all() for v in self.fields.values()))


def join(*values):
    keys = set().union(*(set(v.fields) for v in values))
    return Value(set().union(*(v.deps for v in values)),
                 {k: join(*(v.fields[k] for v in values if k in v.fields)) for k in keys})


class Analyzer:
    def __init__(self, source):
        tree = ast.parse(source)
        self.functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.active = set()

    def expr(self, n, env):
        if n is None: return Value()
        if isinstance(n, ast.Name): return env.get(n.id, Value())
        if isinstance(n, ast.Constant): return Value()
        if isinstance(n, ast.Dict):
            if any(not isinstance(k, ast.Constant) for k in n.keys):
                raise ValueError('Use explicit dictionary keys in decision outputs')
            return Value(fields={k.value: self.expr(v, env) for k, v in zip(n.keys, n.values)})
        if isinstance(n, (ast.List, ast.Tuple)):
            return Value(fields={i: self.expr(v, env) for i, v in enumerate(n.elts)})
        if isinstance(n, ast.Subscript):
            v = self.expr(n.value, env)
            if isinstance(n.slice, ast.Constant) and n.slice.value in v.fields:
                return join(Value(v.deps), v.fields[n.slice.value])
            return Value(v.all() | self.expr(n.slice, env).all())
        if isinstance(n, ast.IfExp):
            return join(self.expr(n.body, env), self.expr(n.orelse, env), Value(self.expr(n.test, env).all()))
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ctx':
                return Value({n.func.attr})
            args = [self.expr(a, env) for a in n.args] + [self.expr(k.value, env) for k in n.keywords]
            if isinstance(n.func, ast.Name) and n.func.id in self.functions:
                if n.func.id in self.active: raise ValueError('Recursive decision helpers are unsupported')
                f = self.functions[n.func.id];self.active.add(n.func.id)
                try: return self.block(f.body, dict(zip((a.arg for a in f.args.args), args)))[1]
                finally: self.active.remove(n.func.id)
            return Value(set().union(*(a.all() for a in args), self.expr(n.func, env).all()))
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            if any(isinstance(x, ast.Constant) and x.value == 0 for x in (n.left, n.right)): return Value()
        return Value(set().union(*(self.expr(c, env).all() for c in ast.iter_child_nodes(n))))

    def assign(self, target, value, env, control):
        value = join(value, Value(control))
        if isinstance(target, ast.Name): env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, t in enumerate(target.elts): self.assign(t, value.fields.get(i, Value(value.all())), env, control)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            old = env.get(target.value.id, Value())
            if isinstance(target.slice, ast.Constant):
                fields = dict(old.fields);fields[target.slice.value] = value
                env[target.value.id] = Value(old.deps, fields)
            else: raise ValueError('Dynamic output mutation cannot establish decision provenance; return explicit results')
        else: raise ValueError('Unsupported decision assignment; use local values and explicit results')

    def block(self, stmts, env, control=None):
        env = dict(env);control = set(control or []);returns = []
        for n in stmts:
            if isinstance(n, ast.Assign):
                v = self.expr(n.value, env)
                for t in n.targets: self.assign(t, v, env, control)
            elif isinstance(n, ast.AnnAssign): self.assign(n.target, self.expr(n.value, env), env, control)
            elif isinstance(n, ast.AugAssign): self.assign(n.target, join(self.expr(n.target, env), self.expr(n.value, env)), env, control)
            elif isinstance(n, ast.Return): returns.append(join(self.expr(n.value, env), Value(control)))
            elif isinstance(n, ast.If):
                ctrl = control | self.expr(n.test, env).all()
                a, ar = self.block(n.body, env, ctrl); b, br = self.block(n.orelse, env, ctrl)
                env = {k: join(a.get(k, Value()), b.get(k, Value())) for k in set(a) | set(b)}
                returns.extend([ar, br])
            elif isinstance(n, (ast.For, ast.While)):
                deps = self.expr(n.iter if isinstance(n, ast.For) else n.test, env).all()
                if isinstance(n, ast.For): self.assign(n.target, Value(deps), env, control)
                e, r = self.block(n.body, env, control | deps)
                env = {k: join(env.get(k, Value()), e.get(k, Value())) for k in set(env) | set(e)};returns.append(r)
            elif isinstance(n, ast.Expr):
                call = n.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ('append', 'extend'):
                    if not isinstance(call.func.value, ast.Name): raise ValueError('Use a local result list')
                    name = call.func.value.id;old = env.get(name, Value());v = self.expr(call.args[0], env)
                    if call.func.attr == 'extend': v = join(*v.fields.values())
                    fields = dict(old.fields);fields[len(fields)] = join(v, Value(control));env[name] = Value(old.deps, fields)
                else: self.expr(call, env)
            elif isinstance(n, (ast.Pass, ast.Break, ast.Continue)): pass
            elif isinstance(n, ast.FunctionDef): self.functions[n.name] = n
            else: raise ValueError('Unsupported decision statement: ' + type(n).__name__)
        return env, join(*returns)


def constant_judgments(source):
    """Recognize only explicit literal claims; unknown Python is diagnostic only."""
    compute=next((n for n in ast.parse(source).body
                  if isinstance(n,ast.FunctionDef) and n.name=='compute'),None)
    if compute is None:return []
    returns=[]
    def visit(node):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.Lambda)):return
        if isinstance(node,ast.Return):
            row={}
            if isinstance(node.value,ast.Dict):
                for key,values in zip(node.value.keys,node.value.values):
                    if isinstance(key,ast.Constant) and key.value=='judgments' and isinstance(values,ast.Dict):
                        for name,value in zip(values.keys,values.values):
                            if isinstance(name,ast.Constant):
                                try:row[name.value]=ast.literal_eval(value)
                                except (ValueError,TypeError):pass
            returns.append(row)
        for child in ast.iter_child_nodes(node):visit(child)
    for statement in compute.body:visit(statement)
    if not returns:return []
    keys=set.intersection(*(set(row) for row in returns))
    return sorted(key for key in keys if all(row[key]==returns[0][key] for row in returns))


def analyze_program(source, result):
    """Return dependency hints and warnings without rejecting executable Python.

Empty dependencies for a plainly constant judgment remain explicit so callers can
distinguish a declared policy value from a measured physical state. Unsupported
syntax is labelled rather than treated as proof that a program ignores the world.
"""
    warnings = []
    constants=constant_judgments(source)
    analyzer = Analyzer(source)
    try:
        if 'compute' not in analyzer.functions: raise ValueError('Define compute(ctx)')
        _, output = analyzer.block(analyzer.functions['compute'].body, {})
    except (ValueError, KeyError, TypeError, IndexError, RecursionError) as exc:
        queries = {n.func.attr for n in ast.walk(ast.parse(source))
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ctx'}
        return {'judgment_dependencies': {key: sorted(queries) for key in result.get('judgments', {})},
                'warnings': ['Static dependency analysis incomplete: ' + str(exc)],
                'constant_judgments':constants}
    judgments = output.fields.get('judgments', Value())
    provenance = {}
    for key in result.get('judgments', {}):
        deps = output.deps | judgments.deps | judgments.fields.get(key, Value()).all()
        if not deps: warnings.append('No static world dependency found for judgment: ' + key)
        provenance[key] = sorted(deps)
    actions = output.fields.get('actions', Value())
    def required(value, inherited, label):
        if not (value.all() | inherited) & PHYSICAL:
            warnings.append('Declared policy constant or unresolved dependency in action parameter: ' + label)

    def parameters(op, args, inherited):
        inherited = inherited | args.deps
        if op in ('move_tcp', 'check_ik'):
            pose = args.fields.get('pose_W', Value())
            if pose.fields:
                for key in ('translation_m', 'quaternion_xyzw'):
                    required(pose.fields.get(key, Value()), inherited | pose.deps, 'pose_W.' + key)
            else: required(pose, inherited, 'pose_W')
        elif op in ('move_tcp_group', 'joints_group', 'gripper_group', 'check_motion'):
            commands = args.fields.get('commands', Value())
            if commands.fields:
                for command in commands.fields.values():
                    child = op.removesuffix('_group') if op != 'check_motion' else ('move_tcp' if 'pose_W' in command.fields else 'joints')
                    parameters(child, command, inherited | commands.deps)
            else: required(commands, inherited, 'commands')
        else:
            key = {'joints':'joint_positions', 'gripper':'mode', 'step':'actions', 'select_arm':'arm'}.get(op)
            if key: required(args.fields.get(key, Value()), inherited, key)

    for index, actual in enumerate(result.get('actions', [])):
        # Loops may generate a variable count from one statically analysed template.
        a = actions.fields.get(index)
        if a is None and len(actions.fields) == 1: a = next(iter(actions.fields.values()))
        if a is None:
            warnings.append('Static action construction is unresolved; inspect recorded runtime queries')
            continue
        deps = output.deps | actions.deps | a.deps | a.fields.get('args', Value()).all()
        if not deps & PHYSICAL: warnings.append('No static measured-world dependency found for action ' + str(index))
        parameters(actual['op'], a.fields.get('args', Value()), output.deps | actions.deps | a.deps)
    return {'judgment_dependencies': provenance, 'warnings': warnings,'constant_judgments':constants}


def output_dependencies(source, result):
    """Compatibility API returning diagnostic dependency hints only."""
    return analyze_program(source, result)['judgment_dependencies']
