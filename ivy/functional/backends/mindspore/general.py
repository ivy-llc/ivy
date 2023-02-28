from typing import Optional, Union, List
import mindspore as ms
import mindspore.numpy as np
import numpy as orig_np


# local
import ivy
from ivy.functional.ivy.gradients import _is_variable
from ivy.functional.ivy.general import _parse_ellipsis
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

def is_native_array(x, /, *, exclusive=False):
    if isinstance(x, ms.Tensor):
        return True
    return False

def current_backend_str() -> str:
    return "mindspore"


def to_numpy(
    x: Union[ms.Tensor, List[ms.Tensor]], /, *, copy: bool = True
) -> Union[orig_np.ndarray, List[orig_np.ndarray]]:
    if isinstance(x, (float, int, bool)):
        return x
    elif isinstance(x, orig_np.ndarray):
        if copy:
            return x.copy()
        else:
            return x
    elif isinstance(x, ms.Tensor):
        if copy:
            return orig_np.array(x)
        else:
            return orig_np.asarray(x)
    elif isinstance(x, list):
        return [ivy.to_numpy(u) for u in x]
    raise ivy.utils.exceptions.IvyException("Expected a Mindspore Tensor.")