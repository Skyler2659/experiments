// Resolve each historical frame's geometry, including agent-published articulated parts.
export function geometryKey(row) {
  return JSON.stringify([row.id, row.part ?? null, row.mesh]);
}
export function frameGeometry(objects, state) {
  const rows = [];
  for (const object of objects) {
    const live = state.objects[object.id];
    if (!live) continue;
    if (live.assembly) {
      for (const [part, spec] of Object.entries(live.assembly.parts)) {
        rows.push({id: object.id, part, mesh: spec.local_mesh,
          T_W_object: live.T_W_object, T_object_part: spec.T_object_part});
      }
    } else {
      rows.push({id: object.id, mesh: object.mesh, T_W_object: live.T_W_object});
    }
  }
  return rows;
}
export function geometryAssets(objects, frames) {
  const assets = new Map();
  for (const frame of frames) {
    for (const row of frameGeometry(objects, frame.state)) {
      assets.set(geometryKey(row), {id: row.id, part: row.part, mesh: row.mesh});
    }
  }
  return [...assets.values()];
}
