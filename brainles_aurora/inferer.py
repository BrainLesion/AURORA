
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
from brainles_aurora.aux import turbo_path
from brainles_aurora.download import download_model_weights
from brainles_aurora.enums import ModalityMode, ModelSelection
import logging


class AuroraInferer(ABC):

    def __init__(self) -> None:
        pass

    def _setup_logger(self):
        logging.basicConfig(

            format='%(asctime)s %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=self.log_level,
            encoding='utf-8',
            handlers=[logging.StreamHandler(), logging.FileHandler(
                'aurora_inferer.log')]
        )

    def _check_files(self):
        pass

    @abstractmethod
    def infer(self, input_data):
        pass


class GPUInferer(AuroraInferer):

    def __init__(self,
                 segmentation_file: str | np.ndarray,
                 t1_file: str | np.ndarray | None = None,
                 t1c_file: str | np.ndarray | None = None,
                 t2_file: str | np.ndarray | None = None,
                 fla_file: str | np.ndarray | None = None,
                 cuda_devices: str = "0",
                 tta: bool = True,
                 sliding_window_batch_size: int = 1,
                 workers: int = 0,
                 threshold: float = 0.5,
                 sliding_window_overlap: float = 0.5,
                 crop_size: Tuple[int, int, int] = (192, 192, 32),
                 model_selection: ModelSelection = ModelSelection.BEST,
                 whole_network_outputs_file: str | None = None,
                 metastasis_network_outputs_file: str | None = None,
                 log_level: int | str = logging.INFO,
                 ) -> None:
        self.segmentation_file = segmentation_file
        self.t1_file = t1_file
        self.t1c_file = t1c_file
        self.t2_file = t2_file
        self.fla_file = fla_file
        self.tta = tta
        self.sliding_window_batch_size = sliding_window_batch_size
        self.workers = workers
        self.threshold = threshold
        self.sliding_window_overlap = sliding_window_overlap
        self.crop_size = crop_size
        self.model_selection = model_selection
        self.whole_network_outputs_file = whole_network_outputs_file
        self.metastasis_network_outputs_file = metastasis_network_outputs_file
        self.log_level = log_level
        self.cuda_devices = cuda_devices

        self._setup_logger()
        self._check_files()

    def infer(self):
        logging.info("Infering on GPU")
        pass
