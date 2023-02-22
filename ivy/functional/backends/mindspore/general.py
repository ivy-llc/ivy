from typing import Optional
import mindspore as ms
import mindspore.numpy as np
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def isin(
    elements: ms.Tensor,
    test_elements: ms.Tensor,
    /,
    *,
    assume_unique: Optional[bool] = False,
    invert: Optional[bool] = False,
) -> ms.Tensor:
    # Numpy argument assume_unique is not supported since the implementation does not
    # rely on the uniqueness of the input arrays.
    return np.isin(
        elements,
        test_elements,
        invert=invert,
    )


