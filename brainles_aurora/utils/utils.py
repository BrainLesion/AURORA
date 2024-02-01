from pathlib import Path
import os
from typing import IO
import sys


def turbo_path(path: str | Path) -> Path:
    """Make path absolute and normed

    Args:
        path (str | Path): input path

    Returns:
        Path: absolute and normed path
    """
    return Path(
        os.path.normpath(
            os.path.abspath(
                path,
            )
        )
    )


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
