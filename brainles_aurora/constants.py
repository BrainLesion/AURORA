from enum import Enum, auto


class ModalityMode(str, Enum):
    T1_T1C_T2_FLA = 't1-t1c-t2-fla'
    T1_T1C_FLA = 't1-t1c-fla'
    T1_T1C = 't1-t1c'
    T1C_FLA = 't1c-fla'
    T1C_O = 't1c-o'
    FLA_O = 'fla-o'
    T1_O = 't1-o'


class ModelSelection(str, Enum):
    BEST = 'best'
    LAST = 'last'
    VANILLA = 'vanilla'


# booleans indicate presence of files in order: T1 T1C T2 FLAIR
FILES_TO_MODE_DICT = {
    (True, True, True, True): ModalityMode.T1_T1C_T2_FLA,
    (True, True, False, True): ModalityMode.T1_T1C_FLA,
    (True, True, False, False): ModalityMode.T1_T1C,
    (False, True, False, True): ModalityMode.T1C_FLA,
    (False, True, False, False): ModalityMode.T1C_O,
    (False, False, False, True): ModalityMode.FLA_O,
    (True, False, False, False): ModalityMode.T1_O,
}
