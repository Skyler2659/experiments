// Recover only host-authored metadata from the original saved observation.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const digest = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');

export function restoreRobotBinding(runtime, name, out) {
  const input = path.join(runtime, 'runs', name, 'input');
  const target = path.join(input, 'robot_binding.json');
  try { read(target); return {restored: false}; }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  // Older/non-robot scenes have no host reference metadata and need no binding.
  if (!fs.existsSync(path.join(input, 'known_robot.glb'))) return {restored: false};
  const meta = read(path.join(input, 'input.json'));
  const views = read(path.join(input, 'views.json'));
  if (digest(path.join(input, 'reference.png')) !== meta.sha256)
    throw Error('Cannot restore robot binding: reference image changed');
  const observations = path.resolve(runtime, '../../observations');
  const matches = [];
  for (const entry of fs.readdirSync(observations, {withFileTypes: true})) {
    if (!entry.isDirectory()) continue;
    const capturePath = path.join(observations, entry.name, 'capture.json');
    if (!fs.existsSync(capturePath)) continue;
    const capture = read(capturePath);
    if (capture.frame === views.frame && capture.views.some(v => v.rgb_sha256 === meta.sha256))
      matches.push({capture, capturePath});
  }
  if (matches.length !== 1) throw Error('Cannot restore robot binding: missing or ambiguous original capture');
  const {capture, capturePath} = matches[0];
  if (!capture.robot_state) throw Error('Cannot restore robot binding: missing public robot state');
  const robotHash = digest(path.join(input, 'known_robot.glb'));
  if (robotHash !== digest(path.resolve(runtime, '../../known_robot.glb')))
    throw Error('Cannot restore robot binding: robot asset differs from original');
  const binding = {reference_sha256: meta.sha256, robot_node: {asset: {sha256: robotHash}},
    frame: capture.frame, robot_state: capture.robot_state};
  const temporary = target + '.restore-' + process.pid;
  try {
    const fd = fs.openSync(temporary, 'wx', 0o600);
    try { fs.writeFileSync(fd, JSON.stringify(binding, null, 2) + '\n'); fs.fsyncSync(fd); }
    finally { fs.closeSync(fd); }
    fs.renameSync(temporary, target);
    if (JSON.stringify(read(target)) !== JSON.stringify(binding)) throw Error('Restored binding readback mismatch');
  } finally { if (fs.existsSync(temporary)) fs.unlinkSync(temporary); }
  const receipt = {restored: true, file: target, source_capture: capturePath,
    source: 'saved public robot state and unchanged reference assets', time: new Date().toISOString()};
  fs.mkdirSync(out, {recursive: true});
  fs.writeFileSync(path.join(out, 'robot_binding_recovery.json'), JSON.stringify(receipt, null, 2) + '\n');
  return receipt;
}
