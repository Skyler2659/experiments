"""Canonical serialization and identifiers shared by current pipeline modules."""
import hashlib,json,re
from pathlib import Path
from typing import Any
from .errors import Real2CodeError
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

def safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not ID.fullmatch(value) or value in {".", ".."}:
        raise Real2CodeError(f"invalid {label}")
    return value

def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value) + b"\n")

def read_json(path: Path) -> Any:
    return json.loads(path.read_text())
