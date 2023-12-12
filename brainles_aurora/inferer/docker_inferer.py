from brainles_aurora.inferer.abstract_inferer import AbstractInferer
from brainles_aurora.inferer.dataclasses import BaseConfig


class DockerInferer(AbstractInferer):
    def __init__(self, config: BaseConfig) -> None:
        super().__init__(config)

    def infer(self):
        # _setup
        # _infer
        # _handle_results
        pass
