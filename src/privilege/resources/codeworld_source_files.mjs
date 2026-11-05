// Hash executable scene inputs, excluding diagnostic histories and private snapshots.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const excluded = new Set(['correction','feedback','trace','.adaptive_best','.best','__pycache__']);
export function sourceFiles(dir, prefix='') {
  const result={};
  for(const entry of fs.readdirSync(dir,{withFileTypes:true})) {
    if(excluded.has(entry.name) || entry.name.startsWith('correction_')) continue;
    const rel=path.posix.join(prefix,entry.name),full=path.join(dir,entry.name);
    if(entry.isDirectory()) Object.assign(result,sourceFiles(full,rel));
    else if(/\.(js|mjs|json|html|glb)$/.test(entry.name) && !entry.name.includes('log'))
      result[rel]=crypto.createHash('sha256').update(fs.readFileSync(full)).digest('hex');
  }
  return result;
}
