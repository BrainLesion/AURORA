from __future__ import annotations

from pathlib import Path


def remove_path_suffixes(path: Path | str) -> Path:
    """Remove all suffixes from a path

    Args:
        path (Path | str): path to remove suffixes from

    Returns:
        Path: path without suffixes
    """
    path_stem = Path(path)
    while path_stem.suffix:
        path_stem = path_stem.with_suffix("")
    return path_stem
