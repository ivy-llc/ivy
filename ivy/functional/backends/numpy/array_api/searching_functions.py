# global
import numpy as np
from typing import Optional

def argmin(x : np.ndarray,
            axis: Optional[int] = None, 
            out: Optional[np.ndarray] = None, 
            keepdims: bool = False
            ) -> np.ndarray:
            
            ret = np.argmin(x,axis=axis,out=out,keepdims=keepdims)
            return ret
