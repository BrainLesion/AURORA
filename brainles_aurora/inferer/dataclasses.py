import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from brainles_aurora.inferer.constants import DataMode, ModelSelection


@dataclass
class BaseConfig:
    """Base configuration for the Aurora model inferer.

    Attributes:
        output_mode (DataMode, optional): Output mode for the inference results. Defaults to DataMode.NIFTI_FILE.
        output_folder (str | Path, optional): Output folder for the results. Defaults to "aurora_output".
        log_level (int | str, optional): Logging level. Defaults to logging.INFO.
        segmentation_file_name (str, optional): File name for the segmentation result. Defaults to "segmentation.nii.gz". (The segmentation will be saved in output_folder/{timestamp}/segmentation_file_name)
        t1 (str | Path | np.ndarray | None, optional): Path or NumPy array for T1 image. Defaults to None.
        t1c (str | Path | np.ndarray | None, optional): Path or NumPy array for T1 contrast-enhanced image. Defaults to None.
        t2 (str | Path | np.ndarray | None, optional): Path or NumPy array for T2 image. Defaults to None.
        fla (str | Path | np.ndarray | None, optional): Path or NumPy array for FLAIR image. Defaults to None.
    """

    output_mode: DataMode = DataMode.NIFTI_FILE
    output_folder: str | Path = "aurora_output"
    segmentation_file_name: str | None = "segmentation.nii.gz"
    log_level: int | str = logging.INFO
    t1: str | Path | np.ndarray | None = None
    t1c: str | Path | np.ndarray | None = None
    t2: str | Path | np.ndarray | None = None
    fla: str | Path | np.ndarray | None = None


@dataclass
class AuroraInfererConfig(BaseConfig):
    """Configuration for the Aurora model inferer.

    Attributes:
        output_mode (DataMode, optional): Output mode for the inference results. Defaults to DataMode.NIFTI_FILE.
        output_folder (str | Path, optional): Output folder for the results. Defaults to "aurora_output".
        segmentation_file_name (str, optional): File name for the segmentation result. Defaults to "segmentation.nii.gz". (The segmentation will be saved in output_folder/{timestamp}/segmentation_file_name)
        log_level (int | str, optional): Logging level. Defaults to logging.INFO.
        t1 (str | Path | np.ndarray | None, optional): Path or NumPy array for T1 image. Defaults to None.
        t1c (str | Path | np.ndarray | None, optional): Path or NumPy array for T1 contrast-enhanced image. Defaults to None.
        t2 (str | Path | np.ndarray | None, optional): Path or NumPy array for T2 image. Defaults to None.
        fla (str | Path | np.ndarray | None, optional): Path or NumPy array for FLAIR image. Defaults to None.
        output_whole_network (bool, optional): Whether to output the whole network results. Defaults to False.
        output_metastasis_network (bool, optional): Whether to output the metastasis network results. Defaults to False.
        tta (bool, optional): Whether to apply test-time augmentations. Defaults to True.
        sliding_window_batch_size (int, optional): Batch size for sliding window inference. Defaults to 1.
        workers (int, optional): Number of workers for data loading. Defaults to 0.
        threshold (float, optional): Threshold for binarizing the model outputs. Defaults to 0.5.
        sliding_window_overlap (float, optional): Overlap ratio for sliding window inference. Defaults to 0.5.
        crop_size (Tuple[int, int, int], optional): Crop size for sliding window inference. Defaults to (192, 192, 32).
        model_selection (ModelSelection, optional): Model selection strategy. Defaults to ModelSelection.BEST.
    """

    output_whole_network: bool = False
    output_metastasis_network: bool = False
    tta: bool = True
    sliding_window_batch_size: int = 1
    workers: int = 0
    threshold: float = 0.5
    sliding_window_overlap: float = 0.5
    crop_size: Tuple[int, int, int] = (192, 192, 32)
    model_selection: ModelSelection = ModelSelection.BEST
