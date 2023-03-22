<<<<<<< HEAD
from typing import Optional
=======
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
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
>>>>>>> a3fa5ae9c4567371f82de20b15479e535a867ead
) -> np.ndarray:
    return np.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True
