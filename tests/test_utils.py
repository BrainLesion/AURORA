import pytest
from pathlib import Path
from brainles_aurora.utils import remove_path_suffixes


class TestAuroraUtils:
    def test_remove_path_suffixes_single_suffix(self):
        suffixes = [".nii", ".nii.gz", ".what.is.this.nii.gz", ""]
        path_stem = "path/to/test"

        paths_str = [path_stem + suffix for suffix in suffixes]
        paths_path = [Path(path) for path in paths_str]

        assert all(
            Path(path_stem) == remove_path_suffixes(p1) == remove_path_suffixes(p2)
            for p1, p2 in zip(paths_path, paths_str)
        )
