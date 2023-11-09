
from abc import ABC, abstractmethod


class AuroraInferer(ABC):
    @abstractmethod
    def __call__(self, input_data):
        pass


class GPUInferer(AuroraInferer):
    def __init__(self) -> None:
        AuroraInferer.__init__(self)

    def __call__(self):
        pass
