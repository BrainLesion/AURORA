from enum import Enum


class InferenceMode(str, Enum):
    """Enum representing different modes of inference based on available image inputs."""

    T1_T1C_T2_FLA = "t1-t1c-t2-fla"
    T1_T1C_FLA = "t1-t1c-fla"
    T1_T1C = "t1-t1c"
    T1C_FLA = "t1c-fla"
    T1C_O = "t1c-o"
    FLA_O = "fla-o"
    T1_O = "t1-o"


class ModelSelection(str, Enum):
    """Enum representing different strategies for model selection."""

    BEST = "best"
    LAST = "last"
    VANILLA = "vanilla"


class DataMode(str, Enum):
    """Enum representing different modes for handling input and output data.

    Enum Values:
        NIFTI_FILE (str): Input data is provided as NIFTI file paths/ output is writte to NIFTI files.
        NUMPY (str): Input data is provided as NumPy arrays/ output is returned as NumPy arrays.
    """

    NIFTI_FILE = "NIFTI_FILEPATH"
    NUMPY = "NP_NDARRAY"


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
