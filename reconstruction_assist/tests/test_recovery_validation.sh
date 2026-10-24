#!/bin/bash
set -euo pipefail
CODE="$1";RT="$2";WS=$(mktemp -d /tmp/rcwm-recovery-scope.XXXXXX)
"$CODE/tools/new_workspace.sh" "$WS" "$RT" >/dev/null
ND="$WS/runs/recovery/fractal/scene";mkdir -p "$ND" "$WS/runs/recovery/trace" "$WS/fake-bin"
"$RT/.venv/bin/python" -c "from PIL import Image;Image.new('RGB',(32,32)).save('$ND/target.png')"
printf 'deadbeef\n' > "$ND/.sid"
printf '["unprepared"]\n' > "$ND/children.json.prev"
printf '{"node_id":"scene","event":"session_start","cycle":1}\n' > "$WS/runs/recovery/trace/events.jsonl"
cat > "$WS/fake-bin/codex" <<'CLI'
#!/bin/bash
ND="$RCWM_ROOT/runs/recovery/fractal/scene"
printf 'called\n' > "$ND/called"
printf 'export function build(){}\n' > "$ND/component.js"
printf '{"module":"component.js","children":[]}\n' > "$ND/part.json"
printf 'session id: deadbeef\ntokens used\n1\n'
CLI
chmod +x "$WS/fake-bin/codex"
env PATH="$WS/fake-bin:$PATH" RCWM_ROOT="$WS" RCWM_CODE="$CODE" RCWM_MAXCYC=4 RCWM_MAXD=3 bash "$CODE/runner/solve_recursive.sh" runs/recovery scene - 0 recovery > "$WS/runner.log" 2>&1
[ -f "$ND/called" ]
[ "$(rg -c 'children_protocol_error' "$WS/runs/recovery/trace/events.jsonl")" = 1 ]
echo "recovery validation returns control to the model ($WS)"
