#!/usr/bin/env bash
# Ubuntu: unpack Chromium's missing libraries into the private runtime.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RCWM_RUNTIME="${RCWM_ROOT:-$ROOT/runtime}"
mkdir -p "$RCWM_RUNTIME/.system-libs/debs"
cd "$RCWM_RUNTIME/.system-libs/debs"
apt-get download libatk1.0-0 libatk-bridge2.0-0 libxcomposite1 libxdamage1 libatspi2.0-0
for deb in ./*.deb; do dpkg-deb -x "$deb" ..; done
