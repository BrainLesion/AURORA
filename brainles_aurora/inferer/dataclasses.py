import logging
from dataclasses import dataclass
from typing import Tuple


from brainles_aurora.inferer import DataMode, ModelSelection


@dataclass
class BaseConfig:
    """Base configuration for the Aurora model inferer.

    Attributes:
        log_level (int, optional): Logging level. Defaults to logging.INFO.
    """

    log_level: int = logging.INFO


@dataclass
class AuroraInfererConfig(BaseConfig):
    """Configuration for the Aurora model inferer.

    Attributes:
        log_level (int | str, optional): Logging level. Defaults to logging.INFO.
        tta (bool, optional): Whether to apply test-time augmentations. Defaults to True.
        sliding_window_batch_size (int, optional): Batch size for sliding window inference. Defaults to 1.
        workers (int, optional): Number of workers for data loading. Defaults to 0.
        threshold (float, optional): Threshold for binarizing the model outputs. Defaults to 0.5.
        sliding_window_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.5.
        crop_size (Tuple[int, int, int], optional): Crop size for sliding window inference. Defaults to (192, 192, 32).
        model_selection (ModelSelection, optional): Model selection strategy. Defaults to ModelSelection.BEST.
    """

    tta: bool = True
    sliding_window_batch_size: int = 1
    workers: int = 0
    threshold: float = 0.5
    sliding_window_overlap: float = 0.5
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    model_selection: ModelSelection = ModelSelection.BEST
