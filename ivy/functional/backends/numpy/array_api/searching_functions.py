import numpy as np

from typing import Union, Optional

def argmax(
    x:np.ndarray,
    axis: Optional[int] = None,
    out: Optional[int] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.argmax(x,axis=axis,out=out, keepdims=keepdims)
    