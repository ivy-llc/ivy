from typing import Optional, Union, Tuple, Sequence
import numpy as np

import ivy  # noqa
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version


def median(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
=======
    keepdims: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if out is not None:
        out = np.reshape(out, input.shape)
    return np.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )


median.support_native_out = True


def nanmean(
    a: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
=======
    keepdims: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    return np.nanmean(a, axis=axis, keepdims=keepdims, dtype=dtype, out=out)


nanmean.support_native_out = True


@with_supported_dtypes({"1.23.0 and below": ("int32", "int64")}, backend_version)
def unravel_index(
    indices: np.ndarray,
    shape: Tuple[int],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> Tuple:
    ret = np.asarray(np.unravel_index(indices, shape), dtype=np.int32)
    return tuple(ret)


unravel_index.support_native_out = False


def quantile(
    a: np.ndarray,
    q: Union[float, np.ndarray],
    /,
    *,
    axis: Optional[Union[int, Sequence[int]]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
=======
    keepdims: bool = False,
    interpolation: str = "linear",
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    # quantile method in numpy backend, always return an array with dtype=float64.
    # in other backends, the output is the same dtype as the input.

    tuple(axis) if isinstance(axis, list) else axis

    return np.quantile(
        a, q, axis=axis, method=interpolation, keepdims=keepdims, out=out
    ).astype(a.dtype)


def corrcoef(
    x: np.ndarray,
    /,
    *,
    y: Optional[np.ndarray] = None,
<<<<<<< HEAD
    rowvar: Optional[bool] = True,
=======
    rowvar: bool = True,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.corrcoef(x, y=y, rowvar=rowvar, dtype=x.dtype)


def nanmedian(
    input: np.ndarray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
<<<<<<< HEAD
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
=======
    keepdims: bool = False,
    overwrite_input: bool = False,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nanmedian(
        input, axis=axis, keepdims=keepdims, overwrite_input=overwrite_input, out=out
    )


nanmedian.support_native_out = True


def bincount(
    x: np.ndarray,
    /,
    *,
    weights: Optional[np.ndarray] = None,
<<<<<<< HEAD
    minlength: Optional[int] = 0,
=======
    minlength: int = 0,
>>>>>>> 389dca45a1e0481907cf9d0cc56aecae3e740c69
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if weights is not None:
        ret = np.bincount(x, weights=weights, minlength=minlength)
        ret = ret.astype(weights.dtype)
    else:
        ret = np.bincount(x, minlength=minlength)
        ret = ret.astype(x.dtype)
    return ret


bincount.support_native_out = False
