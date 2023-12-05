import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from brainles_aurora.constants import DataMode, ModelSelection


@dataclass
class BaseConfig:
    output_mode: DataMode = DataMode.NIFTI_FILE
    output_folder: str | Path = "aurora_output"
    log_level: int | str = logging.INFO


@dataclass
class AuroraInfererConfig(BaseConfig):
    t1: str | Path | np.ndarray | None = None
    t1c: str | Path | np.ndarray | None = None
    t2: str | Path | np.ndarray | None = None
    fla: str | Path | np.ndarray | None = None
    output_whole_network: bool = False
    output_metastasis_network: bool = False
    tta: bool = True
    sliding_window_batch_size: int = 1
    workers: int = 0
    threshold: float = 0.5
    sliding_window_overlap: float = 0.5
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    model_selection: ModelSelection = ModelSelection.BEST
