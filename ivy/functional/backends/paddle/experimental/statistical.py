# global
from typing import Optional, Union, Tuple, Sequence
import paddle
from ivy.utils.exceptions import IvyNotImplementedException

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


def median(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nanmean(
    a: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[paddle.dtype] = None,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def quantile(
    a: paddle.Tensor,
    q: Union[paddle.Tensor, float],
    /,
    *,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: Optional[bool] = False,
    interpolation: Optional[str] = "linear",
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if interpolation not in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
        raise ValueError("interpolation must be 'linear', 'lower', 'higher', 'midpoint', or 'nearest'")
    sorted_a = paddle.sort(a, axis=axis if axis is not None else 0)
    n = paddle.shape(sorted_a)[axis if axis is not None else 0]
    if isinstance(q, float):
        if not 0 <= q <= 1:
            raise ValueError("q must be between 0 and 1")
        indices = q * (n - 1)
        if interpolation == 'lower':
            indices = paddle.floor(indices)
        elif interpolation == 'higher':
            indices = paddle.ceil(indices)
        elif interpolation == 'midpoint':
            indices_floor = paddle.floor(indices)
            delta = q * (n - 1) - indices_floor
            if isinstance(q, paddle.Tensor):
                indices_floor = indices_floor.astype(q)
            indices = indices_floor + delta
        elif interpolation == 'nearest':
            indices = paddle.round(indices)
        lower_indices = paddle.floor(indices)
        upper_indices = paddle.ceil(indices)
        lower_indices = paddle.cast(lower_indices, 'int64')
        upper_indices = paddle.cast(upper_indices, 'int64')
        weights = indices - lower_indices
        weights = paddle.cast(weights, a.dtype)
        lower_quantiles = paddle.index_select(sorted_a, lower_indices, axis=axis if axis is not None else 0)
        upper_quantiles = paddle.index_select(sorted_a, upper_indices, axis=axis if axis is not None else 0)
        result = lower_quantiles * (1 - weights) + upper_quantiles * weights
        if keepdims:
            result = paddle.unsqueeze(result, axis if axis is not None else 0)
        return result
    else:
        indices = paddle.floor(q * (n - 1))
        indices = paddle.cast(indices, 'int64')
        result = paddle.index_select(sorted_a, indices, axis=axis if axis is not None else 0)
        if keepdims:
            result = paddle.unsqueeze(result, axis)
        return result


def corrcoef(
    x: paddle.Tensor,
    /,
    *,
    y: Optional[paddle.Tensor] = None,
    rowvar: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def nanmedian(
    input: paddle.Tensor,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    overwrite_input: Optional[bool] = False,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()


def unravel_index(
    indices: paddle.Tensor,
    shape: Tuple[int],
    /,
    *,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    raise IvyNotImplementedException()
