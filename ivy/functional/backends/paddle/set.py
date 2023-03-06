# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_dtypes
# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException


def unique_all(x: paddle.Tensor, /,) -> Tuple[paddle.Tensor, paddle.Tensor, int ,paddle.Tensor]:
    # Flatten the tensor to 1D
    flat_x = paddle.flatten(x)
    sorted_x = paddle.sort(flat_x)[0]
    indices = paddle.where(sorted_x[1:] != sorted_x[:-1])[0] + 1
    indices = paddle.concat([paddle.to_tensor([0]), indices, paddle.to_tensor([len(flat_x)])])
    nan_indices = paddle.where(paddle.isnan(sorted_x))[0]
    nan_count = len(nan_indices)
    counts = None
    if nan_count > 0:
        nan_values = paddle.full([nan_count], float('nan'), dtype=x.dtype)
        sorted_x = paddle.concat([sorted_x, nan_values], axis=0)
        nan_counts = paddle.to_tensor([paddle.count_nonzero(paddle.isnan(flat_x)).numpy()])
        counts = nan_counts
    if len(indices) == 2:
        unique_x = sorted_x[indices[:-1]]
        if counts is None:
            counts = paddle.to_tensor([len(flat_x)])
    else:
        unique_x = sorted_x[indices[:-1]]
        if counts is None:
            counts = paddle.to_tensor([len(flat_x)])
        else:
            counts = indices[1:] - indices[:-1]

    ret = namedtuple("Results", ["values", "counts", "nan_count", "nan_indices"])
    return ret(unique_x, counts, nan_count, nan_indices)


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
    raise IvyNotImplementedException()
