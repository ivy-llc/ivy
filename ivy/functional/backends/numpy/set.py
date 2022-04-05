# global
import numpy as np
#from numpy import array_api as npa
from typing import Tuple



def unique_all(x : np.ndarray)\
                -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    return np.asarray(np.unique_all(np.asarray(x)))