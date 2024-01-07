# global
from typing import Optional, Tuple
import numpy as np

# local
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


@with_supported_dtypes({"1.26.3 and below": ("int32", "int64")}, backend_version)
def unravel_index(
    indices: np.ndarray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray]:
    ret = np.asarray(np.unravel_index(indices, shape), dtype=np.int32)
    return tuple(ret)


unravel_index.support_native_out = False
