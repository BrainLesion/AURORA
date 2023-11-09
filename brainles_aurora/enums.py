from enum import Enum, auto


class ModalityMode(str, Enum):
    T1_T1C_T2_FLA = 't1-t1c-t2-fla'
    T1C_T1_FLA = 't1c-t1-fla'
    T1C_T1 = 't1c-t1'
    T1C_FLA = 't1c-fla'
    T1C_O = 't1c-o'
    FLA_O = 'fla-o'
    T1_O = 't1-o'


class ModelSelection(str, Enum):
    BEST = 'best'
    LAST = 'last'
    VANILLA = 'vanilla'
