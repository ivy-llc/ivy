from typing import Optional
import numpy as np


def isin(
    elements: np.ndarray,
    test_elements: np.ndarray,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> np.ndarray:
    return np.isin(
        elements,
        test_elements,
        assume_unique=assume_unique,
        invert=invert,
    )


isin.support_native_out = True
