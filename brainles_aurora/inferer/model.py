import logging
import os
from pathlib import Path
from typing import List

import monai
import numpy as np
from monai.data import list_data_collate
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Lambdad,
    LoadImageD,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)
from torch.utils.data import DataLoader

from brainles_aurora.inferer.config import AuroraInfererConfig
from brainles_aurora.inferer.constants import IMGS_TO_MODE_DICT, DataMode, InferenceMode

logger = logging.getLogger(__name__)


class ModelHandler:
    """TODO"""
