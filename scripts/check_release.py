#!/usr/bin/env python3
"""Check the publishable source set without reading credentials or experiment outputs."""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SKIP = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "build",
    "dist",
    "runs",
    "outputs",
    "runtime",
    "node_modules",
}
problems = []
count = size = 0
for path in ROOT.rglob("*"):
    rel = path.relative_to(ROOT)
    if any(part in SKIP or part.endswith(".egg-info") for part in rel.parts):
        continue
    if path.is_symlink():
        problems.append(f"External dependency/symlink: {rel}")
        continue
    if not path.is_file():
        continue
    count += 1
    size += path.stat().st_size
    if path.name in {"auth.json", ".env", ".DS_Store"} or path.suffix in {".pem", ".key", ".pyc"}:
        problems.append(f"Private or generated file: {rel}")
    if path.stat().st_size > 5_000_000:
        problems.append(f"Unexpected large source file: {rel}")
    if path.suffix not in {".py", ".md", ".json", ".toml", ".sh", ".mjs", ".html", ".txt", ".yml"}:
        continue
    text = path.read_text(errors="replace")
    if re.search(r"/(?:Users|private/lc)/[A-Za-z0-9_-]+/", text):
        problems.append(f"Machine-specific path: {rel}")
    if re.search(r"\bsk-[A-Za-z0-9_-]{24,}\b|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----", text):
        problems.append(f"Credential-like literal: {rel}")
print(f"Checked {count} source files, {size / 1024**2:.2f} MiB")
if problems:
    print("\n".join(problems))
    sys.exit(1)
print("Release source checks passed")
