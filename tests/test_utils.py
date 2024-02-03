from pathlib import Path
from brainles_aurora.utils import remove_path_suffixes


class TestAuroraUtils:
    def test_remove_path_suffixes(self):
        suffixes = [".nii", ".nii.gz", ".what.is.this.nii.gz", ""]
        path_stem = "path/to/test"

        paths_str = [path_stem + suffix for suffix in suffixes]

        assert all(
            Path(path_stem) == remove_path_suffixes(p) == remove_path_suffixes(Path(p))
            for p in paths_str
        )
