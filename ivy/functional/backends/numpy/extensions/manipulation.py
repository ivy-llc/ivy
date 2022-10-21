from typing import Optional, Union, Sequence
import numpy as np


def moveaxis(
    a: np.ndarray,
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.moveaxis(a, source, destination)


moveaxis.support_native_out = False


def heaviside(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.heaviside(
        x1,
        x2,
        out=out,
    )


heaviside.support_native_out = True


def flipud(
    m: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.flipud(m)


flipud.support_native_out = False
