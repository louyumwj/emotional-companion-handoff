"""Resolve packaged model paths without depending on the original checkout."""
import ntpath
from pathlib import Path


def resolve_base(root, saved_base, override=""):
    root = Path(root)
    if override:
        candidates = [Path(override)]
        if not candidates[0].is_absolute():
            candidates = [root / override]
    else:
        name = ntpath.basename(str(saved_base).rstrip("/\\"))
        candidates = [root / "models" / name]
        if not ntpath.isabs(str(saved_base)):
            candidates.append(root / saved_base)
        else:
            candidates.append(Path(saved_base))
    for path in candidates:
        if (path / "config.json").is_file():
            return str(path.resolve())
    raise FileNotFoundError(
        "Local base model not found. Restore models/ first or pass --base. "
        + "Checked: " + ", ".join(map(str, candidates))
    )


def portable_base(root, base):
    path = Path(base).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return "models/" + path.name
