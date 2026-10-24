#!/bin/bash
set -euo pipefail
CODE="$1"; RT="$2"; WS=$(mktemp -d /tmp/rcwm-service-failure.XXXXXX)
"$CODE/tools/new_workspace.sh" "$WS" "$RT" >/dev/null
"$RT/.venv/bin/python" -c "from PIL import Image; Image.new('RGB',(64,64)).save('$WS/ref.png')"
printf '#!/bin/bash\nexit 20\n' > "$WS/failing-hook"; chmod +x "$WS/failing-hook"
set +e
env PATH="$CODE/tests/fake-codex:$PATH" RCWM_ROOT="$WS" RCWM_CLEAN_CODEX_HOME=0 RCWM_FEEDBACK_HOOK="$WS/failing-hook" bash "$CODE/rcwm.sh" "$WS/ref.png" failcase 2 3 > "$WS/run.log" 2>&1
STATUS=$?
set -e
if [ "$STATUS" -ne 20 ]; then cat "$WS/run.log"; echo "Expected service error 20, got $STATUS"; exit 1; fi
echo "service failure correctly propagated; fixture retained at $WS"
