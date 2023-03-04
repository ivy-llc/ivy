# global
import paddle
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
    {"2.4.2 and below": ("int8", "int16", "uint8", "uint16", "bfloat16",
                         "float16", "complex64", "complex128", "bool")},
    backend_version,
)
def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    unique, counts = paddle.unique(x, return_counts=True)
    nan_count = paddle.count_nonzero(paddle.where(paddle.isnan(x) > 0)).numpy()[0]

    if nan_count > 0:
        unique_nan = paddle.full(shape=[1, nan_count],
                                 fill_value=float('nan')).cast(x.dtype)
        counts_nan = paddle.full(shape=[1, nan_count], fill_value=1).cast(x.dtype)
        unique = paddle.concat(
            [unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])], axis=0)
        counts = paddle.concat(
            [counts.astype(x.dtype), paddle.reshape(counts_nan, [nan_count])], axis=0)

    Results = namedtuple("Results", ["values", "counts"])
    return Results(unique, counts)


def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    raise IvyNotImplementedException()


def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    nan_count = paddle.sum(paddle.isnan(x))
    unique = paddle.unique(x)
    if nan_count > 0:
        nans = paddle.full(shape=[nan_count], fill_value=float('nan')).cast(x.dtype)
        unique = paddle.concat([unique, nans])
    return unique
