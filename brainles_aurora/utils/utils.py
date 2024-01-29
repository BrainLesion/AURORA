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


class DualStdErrOutput:
    """Class to write to stderr and a file at the same time"""

    stderr: IO = None

    @staticmethod
    def enable(file_stream: IO):
        """Enable DualStdErrOutput"""
        if DualStdErrOutput.stderr is None:
            DualStdErrOutput.stderr = sys.stderr
            sys.stderr = DualStdErrOutput(sys.stderr, file_stream)
        else:
            sys.stderr.set_file_handler_stream(file_stream)

    def __init__(self, stderr: IO, file_handler_stream: IO = None):
        if DualStdErrOutput.stderr is None:
            DualStdErrOutput.stderr = stderr
        self.stderr = DualStdErrOutput.stderr
        # print("init", flush=True)
        # self.stderr.write("DualStdErrOutput initialized\n")
        self.file_handler_stream = file_handler_stream

    def set_file_handler_stream(self, file_handler_stream: IO):
        self.file_handler_stream = file_handler_stream

    def write(self, text: str):
        self.stderr.write(text)
        if self.file_handler_stream:
            self.file_handler_stream.write(text)

    def flush(self):
        self.stderr.flush()
        if self.file_handler_stream:
            self.file_handler_stream.flush()
