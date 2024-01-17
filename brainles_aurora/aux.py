from pathlib import Path
import os
from typing import IO


def turbo_path(the_path):
    turbo_path = Path(
        os.path.normpath(
            os.path.abspath(
                the_path,
            )
        )
    )
    return turbo_path


class DualStdErrOutput:
    def __init__(self, stderr: IO, file_handler_stream: IO = None):
        self.stderr = stderr
        self.file_handler_stream = file_handler_stream

    def set_file_handler_stream(self, file_handler_stream: IO):
        self.file_handler_stream = file_handler_stream

    def write(self, text):
        self.stderr.write(text)
        if self.file_handler_stream:
            self.file_handler_stream.write(text)

    def flush(self):
        self.stderr.flush()
        if self.file_handler_stream:
            self.file_handler_stream.flush()
