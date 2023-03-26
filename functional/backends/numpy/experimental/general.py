<<<<<<< HEAD
from typing import Optional
=======
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
import numpy as np
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"1.23.0 and below": ("bfloat16",)}, backend_version)
def isin(
    elements: np.ndarray,
    test_elements: np.ndarray,
    /,
    *,
<<<<<<< HEAD
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
=======
    assume_unique: bool = False,
    invert: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
) -> np.ndarray:
    return np.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True
