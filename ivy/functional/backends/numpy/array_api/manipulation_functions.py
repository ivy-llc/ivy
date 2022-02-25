# global
from typing import Union, Tuple
import numpy as np

def roll(x: np.ndarray, shift: Union[int, Tuple[int]], axis: Union[int, Tuple[int]]=None)\
        -> np.ndarray:
    return np.roll(x, shift, axis)

