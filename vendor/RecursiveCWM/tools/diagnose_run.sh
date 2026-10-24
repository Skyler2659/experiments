#!/bin/bash
# diagnose_run.sh <run-dir> : what happened in a run, from its own files (no model, no network).
# Prints the tree shape and cost, every node's delivery/render state, the stop reasons, and greps every session log
# for the failure signatures that silently ruin a run: Chromium not launching (missing system libraries or browsers),
# sandbox denials, a full disk, quota/rate limits, dropped connections. Paste its output when asking for help.
set -uo pipefail
RUN="$(cd "${1:?usage: diagnose_run.sh <run-dir>}" && pwd)"; F="$RUN/fractal"; EV="$RUN/trace/events.jsonl"
[ -d "$F" ] || { echo "not a run directory (no fractal/): $RUN"; exit 1; }
echo "run: $RUN"
[ -f "$RUN/conditions.json" ] && python3 -c "import json;d=json.load(open('$RUN/conditions.json'));print('conditions:',', '.join(f'{k}={d[k]}' for k in ('instruction_sha256','max_depth','max_cycles','model','reasoning_effort','codex_cli','node','three','playwright') if k in d))"
echo; echo "== tree and cost"
if [ -f "$EV" ]; then python3 - "$EV" "$F" <<'P'
import json,sys,collections,glob,os,datetime
ev,F=sys.argv[1],sys.argv[2]; dep={}; ts=[]; stops=collections.Counter(); sess=collections.Counter(); tok=collections.Counter()
for l in open(ev):
    try: e=json.loads(l.replace('"usage_total":}','"usage_total":0}'))
    except Exception: continue
    ts.append(e['ts'])
    if e['event']=='session_start': dep[e['node_id']]=e['depth']
    if e['event']=='session_end': sess[e['node_id']]+=1; tok[e['node_id']]+=e.get('usage_total') or 0
    if e['event']=='stop': stops[e.get('stop_reason')]+=1
c=collections.Counter(dep.values())
print(f"nodes={len(dep)} max_depth={max(c) if c else 0} per_level={[c[i] for i in range(max(c)+1)] if c else []}")
print(f"sessions={sum(sess.values())} tokens_M={sum(tok.values())/1e6:.2f} stop_reasons={dict(stops)}")
if ts:
    t0=datetime.datetime.fromisoformat(ts[0]); t1=datetime.datetime.fromisoformat(ts[-1]); print(f"wall_min={int((t1-t0).total_seconds()/60)}  first={ts[0]}  last={ts[-1]}")
print("(stop_reason invalid_artifact only labels the trace: the runner's fact check disagreed with the part.json shape; the delivery still counts)")
P
else echo "no trace/events.jsonl"; fi
echo; echo "== nodes: delivered / renders it made / sessions   (a delivered node with 0 renders worked blind)"
find "$F" -name part.json | sort | while read p; do
  d=$(dirname "$p"); n=${d#$F/}
  r=$(find "$d" -maxdepth 2 -name '*.png' ! -name target.png 2>/dev/null | wc -l)
  s=$(grep -a -c "tokens used" "$d/codex-run.log" 2>/dev/null || echo 0)
  printf "  %-50s part=yes renders=%-4s sessions=%s\n" "$n" "$r" "$s"
done
for d in $(find "$F" -mindepth 1 -type d -name '*' | sort); do
  [ -f "$d/task.md" ] && [ ! -f "$d/part.json" ] && printf "  %-50s part=NO  (started, never delivered)\n" "${d#$F/}"
done
echo; echo "== failure signatures in session logs (nodes whose log matches, and the first matching line; 0 = not seen)"
LOGS=$(find "$F" -name codex-run.log)
sig() { local n first; n=$(grep -a -i -l -- "$2" $LOGS 2>/dev/null | wc -l); printf "  %-30s nodes=%s" "$1" "$n"
        if [ "$n" -gt 0 ]; then first=$(grep -a -i -h -m1 -- "$2" $LOGS 2>/dev/null | head -1 | cut -c1-110); printf "   e.g. %s" "$first"; fi; echo; }
sig "chromium launch failed"        "Failed to launch\|browserType.launch\|Executable doesn't exist\|error while loading shared libraries\|libnss3\|libatk\|libgbm"
sig "browser/page closed"           "Target page, context or browser has been closed\|Protocol error"
sig "sandbox denied write"          "Read-only file system\|Operation not permitted\|EACCES\|EPERM"
sig "disk full"                     "ENOSPC\|No space left"
sig "quota / rate limit"            "hit your usage limit\|rate limit\|HTTP 429\|status 429\|Too Many Requests\|insufficient_quota"
sig "connection dropped"            "stream disconnected\|ECONNRESET\|ETIMEDOUT\|reconnecting"
sig "python import failed"          "ModuleNotFoundError\|No module named"
sig "three.js not found"            "Cannot find module 'three'\|three.module.js.*404\|ERR_MODULE_NOT_FOUND"
echo; echo "== nodes whose session never printed a session id (codex did not start)"
for f in $LOGS; do grep -a -q "session id" "$f" || echo "  ${f#$F/}"; done
echo; echo "== first 3 lines of the root's log (executor, sandbox, effort)"
sed -n '2,5p' "$F/scene/codex-run.log" 2>/dev/null | sed 's/^/  /'
echo; echo "renders to look at:  $F/scene/final.png (or whatever the root's part.json names);  compare with $F/scene/target.png"
