# global
import numpy as np
from typing  import Union, Optional, Tuple, Literal

# local
from ivy import inf


def vector_norm(x: np.ndarray, 
                p: Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                 -> np.ndarray:

    np_normalized_vector = None

    if axis == None:
        np_normalized_vector = np.linalg.norm(x.flatten(),p, axis,keepdims)

    else:
        np_normalized_vector = np.linalg.norm(x,p, axis,keepdims)

    if np_normalized_vector.shape == tuple():
        return np.expand_dims(np_normalized_vector, 0)
    return np_normalized_vector
