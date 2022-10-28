# global
from typing import Optional, Union, Sequence, Tuple, NamedTuple
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


def vstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.vstack(arrays)


def hstack(
    arrays: Sequence[np.ndarray],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.hstack(arrays)


def rot90(
    m: np.ndarray,
    /,
    *,
    k: Optional[int] = 1,
    axes: Optional[Tuple[int, int]] = (0, 1),
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.rot90(m, k, axes)


def top_k(
    x: np.ndarray,
    k: int,
    /,
    *,
    axis: Optional[int] = -1,
    largest: Optional[bool] = True,
    out: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not largest:
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
    else:
        x *= -1
        indices = np.argsort(x, axis=axis)
        indices = np.take(indices, np.arange(k), axis=axis)
        x *= -1
    topk_res = NamedTuple("top_k", [("values", np.ndarray), ("indices", np.ndarray)])
    val = np.take_along_axis(x, indices, axis=axis)
    return topk_res(val, indices)
