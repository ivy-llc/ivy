# global
import paddle
import paddle.fluid as fluid
from typing import Tuple, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_dtypes

# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(
    x: paddle.Tensor,
    /,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8", "int16", "uint8", "uint16", "bfloat16", "float16", "float32", "float64", "bool")},
    backend_version,
)
def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    unique, counts = paddle.unique(x, return_counts=True)
    nan_count = paddle.count_nonzero(paddle.where(paddle.isnan(x) > 0)).numpy()[0]

    if nan_count > 0:
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float('nan')).cast(x.dtype)
        counts_nan = paddle.full(shape=[1, nan_count], fill_value=1).cast(x.dtype)
        unique = paddle.concat(input=[unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])], axis=0)
        counts = paddle.concat(input=[counts.astype(x.dtype), paddle.reshape(counts_nan, [nan_count])], axis=0)

    Results = namedtuple("Results", ["values", "counts"])
    return Results(unique, counts)


@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8", "int16", "uint8", "uint16", "bfloat16", "float16", "float32", "float64", "bool")},
    backend_version,
)
def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    unique, inverse_val = paddle.unique(x, return_inverse=True)
    nan_idx = paddle.where(paddle.isnan(x) > 0)
    nan_count = paddle.count_nonzero(nan_idx).numpy()[0]

    if nan_count > 0:
        inverse_val[nan_idx] = len(unique)
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float('nan')).cast(x.dtype)
        unique = fluid.layers.concat(input=[unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])], axis=0)

    Results = namedtuple("Results", ["values", "inverse_indices"])
    return Results(unique, inverse_val)


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    raise IvyNotImplementedException()
