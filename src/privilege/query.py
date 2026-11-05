"""Geometry snapshots for computations in the host agent's own Python runtime."""

import copy

import numpy as np


class Snapshot:
    def __init__(self, world, *, state=None):
        self._world = world
        self._state = copy.deepcopy(world.state() if state is None else state)
        self._reads = set()
        self.revision = self._state["revision"]
        self.frame = world.manifest["frame"]

    def objects(self):
        self._reads.add((None, 'membership'))
        return list(self._state["objects"])

    def transform(self, object_id, part_id=None):
        self._reads.add((object_id, 'geometry' if part_id is not None else 'pose'))
        row = self._state["objects"][object_id]
        transform = np.array(row["T_W_object"])
        if part_id is not None:
            transform = transform @ np.array(row["assembly"]["parts"][part_id]["T_object_part"])
        return transform

    def mesh(self, object_id, part_id=None):
        self._reads.add((object_id, 'geometry'))
        if part_id is not None:
            return self._world._part_mesh(object_id, part_id, self._state)
        return self._world._mesh(object_id, self._state)

    def bounds(self, object_id, part_id=None):
        return self.mesh(object_id, part_id).bounds.copy()

    def position(self, object_id, part_id=None):
        """Mesh centroid, not necessarily a grasp point or the object origin."""
        return self.mesh(object_id, part_id).centroid.copy()

    def uncertainty(self, object_id):
        self._reads.add((object_id, 'uncertainty'))
        row = self._state["objects"][object_id]
        return copy.deepcopy(
            {k: row.get(k) for k in ("basis", "review", "sensor_validation", "evidence", "pose_refresh")}
        )

    def local_point(self, object_id, point, part_id=None):
        t = self.transform(object_id, part_id)
        p = np.asarray(point, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError("Expected finite local 3-vector")
        return t[:3, :3] @ p + t[:3, 3]


def dependency_value(state, object_id, aspect):
    """Compare inputs, not global revisions or review timestamps."""
    if aspect == 'membership':
        return sorted(state['objects'])
    row = state['objects'].get(object_id)
    if row is None:
        return None
    value = {'pose': row['T_W_object']}
    if aspect == 'geometry':
        assembly = row.get('assembly', {})
        value['parts'] = {pid: {k: part.get(k) for k in ('sha256', 'local_mesh', 'T_object_part')}
                          for pid, part in assembly.get('parts', {}).items()}
    if aspect == 'uncertainty':
        value = {k: row.get(k) for k in ('basis', 'review', 'pose_refresh', 'sensor_validation', 'evidence')}
    # Regrasp/release can change how a target must be computed even at an
    # unchanged object pose. Do not invalidate unrelated objects on arm motion.
    value['attachments'] = {arm: {'T_E_object': item['T_E_object']}
                            for arm, item in state.get('attachments', {}).items()
                            if item['object_id'] == object_id}
    return value


class DerivedValue(np.lib.mixins.NDArrayOperatorsMixin):
    """Dependency-aware numeric expression, evaluated from one current snapshot."""
    __array_priority__ = 1000

    def __init__(self, world, compute):
        if not callable(compute):
            raise TypeError('compute requires a function of a fresh snapshot')
        self._world, self._compute = world, compute
        self._inputs = None
        self._value = None
        self.evaluations = 0
        self.revision = None

    def _validate(self, value):
        value = np.asarray(value)
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            raise ValueError('expression must return finite numeric values')
        return value

    def value(self):
        state = self._world.state()
        if self._inputs is None or any(dependency_value(state, *key) != old
                                       for key, old in self._inputs.items()):
            snapshot = Snapshot(self._world, state=state)
            self._value = self._validate(self._compute(snapshot)).copy()
            self._inputs = {key: copy.deepcopy(dependency_value(state, *key)) for key in snapshot._reads}
            self.evaluations += 1
            self.revision = snapshot.revision
        return self._value.copy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method not in ('__call__', 'reduce') or kwargs.get('out') is not None:
            return NotImplemented
        return self._compose(getattr(ufunc, method), inputs, kwargs)

    def __array_function__(self, func, types, args, kwargs):
        if kwargs.get('out') is not None:
            return NotImplemented
        return self._compose(func, args, kwargs)

    def _compose(self, func, args, kwargs):
        def capture(value):
            if isinstance(value, DerivedValue):
                if value._world is not self._world:
                    raise ValueError('Cannot combine expressions from different worlds')
                return value
            if isinstance(value, (list, tuple)):
                return type(value)(capture(v) for v in value)
            if isinstance(value, dict):
                return {k: capture(v) for k, v in value.items()}
            return copy.deepcopy(value)
        def resolve(value, snapshot):
            if isinstance(value, DerivedValue):
                return value._compute(snapshot)
            if isinstance(value, (list, tuple)):
                return type(value)(resolve(v, snapshot) for v in value)
            if isinstance(value, dict):
                return {k: resolve(v, snapshot) for k, v in value.items()}
            return value
        captured_args, captured_kwargs = capture(args), capture(kwargs)
        return DerivedValue(self._world, lambda s: func(
            *resolve(captured_args, s), **resolve(captured_kwargs, s)))

    def __array__(self, dtype=None, copy=None):
        # Explicit NumPy conversion is a numeric snapshot, never a live proxy.
        return np.array(self.value(), dtype=dtype, copy=True)

    def __iter__(self):
        return iter(self.value().tolist())

    def __len__(self):
        return len(self.value())

    def __getitem__(self, index):
        index = copy.deepcopy(index)
        return DerivedValue(self._world, lambda s: self._compute(s)[index])

    def __float__(self):
        return float(self.value())

    def __bool__(self):
        return bool(self.value())

    @property
    def T(self):
        return DerivedValue(self._world, lambda s: np.asarray(self._compute(s)).T)

    def reshape(self, *shape):
        return DerivedValue(self._world, lambda s: np.asarray(self._compute(s)).reshape(*shape))


class DerivedTarget(DerivedValue):
    """Backward-compatible finite xyz expression."""
    def _validate(self, value):
        value = super()._validate(value)
        if value.shape != (3,):
            raise ValueError('target expression must return finite xyz')
        return value


class LiveQuery:
    """Opt-in live computations; Snapshot remains explicitly frozen."""
    def __init__(self, world):
        self._world = world

    def _query(self, method, *args):
        args = copy.deepcopy(args)
        return DerivedValue(self._world, lambda s: getattr(s, method)(*args))

    def position(self, object_id, part_id=None):
        return self._query('position', object_id, part_id)

    def transform(self, object_id, part_id=None):
        return self._query('transform', object_id, part_id)

    def bounds(self, object_id, part_id=None):
        return self._query('bounds', object_id, part_id)

    def local_point(self, object_id, point, part_id=None):
        return self._query('local_point', object_id, point, part_id)
