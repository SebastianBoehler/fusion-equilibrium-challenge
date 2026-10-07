"""Content identities and atomic writes for immutable experiment artifacts."""

from hashlib import sha256
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def identity(value: dict) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def code_digest(paths: list[Path]) -> str:
    return identity({str(p): digest(p) for p in sorted(paths)})


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)
