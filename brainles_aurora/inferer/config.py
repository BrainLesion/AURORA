import logging
from dataclasses import dataclass
from typing import Tuple

from brainles_aurora.inferer.constants import Device, ModelSelection


@dataclass
class BaseConfig:
    """Base configuration for the Aurora model inferer."""

    log_level: int = logging.INFO
    """Logging level. Defaults to logging.INFO."""
    device: Device = Device.AUTO
    """Device for model inference. Defaults to Device.AUTO."""
    cuda_devices: str = "0"
    """CUDA devices to use when using CUDA. Defaults to "0"."""


@dataclass
class AuroraInfererConfig(BaseConfig):
    """Configuration for the Aurora model inferer."""

    tta: bool = True
    """Whether to apply test-time augmentations. Defaults to True."""
    sliding_window_batch_size: int = 1
    """Batch size for sliding window inference. Defaults to 1."""
    workers: int = 0
    """Number of workers for data loading. Defaults to 0."""
    threshold: float = 0.5
    """Threshold for binarizing the model outputs. Defaults to 0.5."""
    sliding_window_overlap: float = 0.5
    """Overlap ratio for sliding window inference. Defaults to 0.5."""
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    """Crop size for sliding window inference. Defaults to (192, 192, 32)."""
    model_selection: ModelSelection = ModelSelection.BEST
    """Model selection strategy. Defaults to ModelSelection.BEST."""
