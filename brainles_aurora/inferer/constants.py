from enum import Enum


class InferenceMode(str, Enum):
    """
    Enum representing different modes of inference based on available image inputs.\n
    In General You should aim to use as many modalities as possible to get the best results.
    """

    T1_T1C_T2_FLA = "t1-t1c-t2-fla"
    """All four modalities are available."""
    T1_T1C_FLA = "t1-t1c-fla"
    """T1, T1C, and FLAIR are available."""
    T1_T1C = "t1-t1c"
    """T1 and T1C are available."""
    T1C_FLA = "t1c-fla"
    """T1C and FLAIR are available."""
    T1C_O = "t1c-o"
    """T1C is available."""
    FLA_O = "fla-o"
    """FLAIR is available."""
    T1_O = "t1-o"
    """T1 is available."""


class ModelSelection(str, Enum):
    """Enum representing different strategies for model selection."""

    BEST = "best"
    """Select the best performing model."""
    LAST = "last"
    """Select the last model."""
    VANILLA = "vanilla"
    """Select the vanilla model."""


class DataMode(str, Enum):
    """Enum representing different modes for handling input and output data."""

    NIFTI_FILE = "NIFTI_FILEPATH"
    """Input data is provided as NIFTI file paths/ output is writte to NIFTI files."""
    NUMPY = "NP_NDARRAY"
    """Input data is provided as NumPy arrays/ output is returned as NumPy arrays."""


class Output(str, Enum):
    """Enum representing different types of output."""

    SEGMENTATION = "segmentation"
    """Segmentation mask"""
    WHOLE_NETWORK = "whole_network"
    """Whole network output."""
    METASTASIS_NETWORK = "metastasis_network"
    """Metastasis network output."""


MODALITIES = ["t1", "t1c", "t2", "fla"]
"""List of modality names in standard order: T1 T1C T2 FLAIR (['t1', 't1c', 't2', 'fla'])"""


# booleans indicate presence of files in order: T1 T1C T2 FLAIR
IMGS_TO_MODE_DICT = {
    (True, True, True, True): InferenceMode.T1_T1C_T2_FLA,
    (True, True, False, True): InferenceMode.T1_T1C_FLA,
    (True, True, False, False): InferenceMode.T1_T1C,
    (False, True, False, True): InferenceMode.T1C_FLA,
    (False, True, False, False): InferenceMode.T1C_O,
    (False, False, False, True): InferenceMode.FLA_O,
    (True, False, False, False): InferenceMode.T1_O,
}
"""Dictionary mapping tuples of booleans representing presence of the modality in order [t1,t1c,t2,fla] to InferenceMode values."""


class Device(str, Enum):
    """Enum representing device for model inference."""

    CPU = "cpu"
    """Use CPU"""
    GPU = "cuda"
    """Use GPU (CUDA)"""
    AUTO = "auto"
    """Attempt to use GPU, fallback to CPU."""


WEIGHTS_DIR_PATTERN = "weights_v*.*.*"
"""Directory name pattern to store model weights. E.g. weights_v1.0.0"""
