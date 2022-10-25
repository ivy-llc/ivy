# global
from typing import Optional, Tuple

import numpy as np

# local
from ivy.functional.backends.numpy.device import _to_device


# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: str,
) -> Tuple[np.ndarray]:
    return tuple(
        _to_device(np.asarray(np.triu_indices(n=n_rows, k=k, m=n_cols)), device=device)
    )
