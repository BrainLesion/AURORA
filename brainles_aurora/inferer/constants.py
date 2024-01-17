from enum import Enum


class InferenceMode(str, Enum):
    """Enum representing different modes of inference based on available image inputs

    Enum Values:
        T1_T1C_T2_FLA (str): All four modalities are available.
        T1_T1C_FLA (str): T1, T1C, and FLAIR are available.
        T1_T1C (str): T1 and T1C are available.
        T1C_FLA (str): T1C and FLAIR are available.
        T1C_O (str): T1C is available.
        FLA_O (str): FLAIR is available.
        T1_O (str): T1 is available.
    """

    T1_T1C_T2_FLA = "t1-t1c-t2-fla"
    T1_T1C_FLA = "t1-t1c-fla"
    T1_T1C = "t1-t1c"
    T1C_FLA = "t1c-fla"
    T1C_O = "t1c-o"
    FLA_O = "fla-o"
    T1_O = "t1-o"


class ModelSelection(str, Enum):
    """Enum representing different strategies for model selection.

    Enum Values:
        BEST (str): Select the best performing model.
        LAST (str): Select the last model.
        VANILLA (str): Select the vanilla model.
    """

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


class Output(str, Enum):
    """Enum representing different types of output.

    Enum Values:
        SEGMENTATION (str): Segmentation mask.
        WHOLE_NETWORK (str): Whole network output.
        METASTASIS_NETWORK (str): Metastasis network output.
    """

    SEGMENTATION = "segmentation"
    WHOLE_NETWORK = "whole_network"
    METASTASIS_NETWORK = "metastasis_network"


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
