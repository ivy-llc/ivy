# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_dtypes
# local

from . import backend_version
from ivy.utils.exceptions import IvyNotImplementedException

@with_unsupported_dtypes(
    {"2.4.2 and below": ("int8", "int16", "uint8", "uint16", "bfloat16",
                         "float16", "complex64", "complex128", "complex")},
    backend_version,
)
def unique_all(x: paddle.Tensor, /,) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    flat_x = paddle.flatten(x)
    sorted_x, indices = paddle.sort(flat_x)
    unique_indices = paddle.concat([paddle.to_tensor([0]), paddle.nonzero(sorted_x[1:] != sorted_x[:-1])[:, 0] + 1,
                                    paddle.to_tensor([len(sorted_x)])])
    unique_x = paddle.gather(sorted_x, unique_indices[:-1])
    counts = unique_indices[1:] - unique_indices[:-1]
    inverse_indices = paddle.argsort(indices)
    nan_indices = paddle.where(paddle.isnan(sorted_x))[0]
    nan_count = len(nan_indices)
    if nan_count > 0:
        nan_values = paddle.full([nan_count], float('nan'), dtype=x.dtype)
        unique_x = paddle.concat([unique_x, nan_values], axis=0)
        nan_counts = paddle.to_tensor([paddle.count_nonzero(paddle.isnan(flat_x)).numpy()])
        counts = paddle.concat([counts, nan_counts], axis=0)

    Results = namedtuple("Results", ["values", "indices", "inverse_indices", "counts"])
    return Results(unique_x, unique_indices[:-1], inverse_indices, counts)


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

    
@with_unsupported_dtypes(
    {"2.4.2 and below": ("complex64", "complex128")},
    backend_version,
)
def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    x_dtype = x.dtype
    nan_count = paddle.sum(paddle.isnan(x.cast("float64")))
    unique = paddle.unique(x.cast('float64'))
    if nan_count > 0:
        nans = paddle.full(shape=[nan_count], fill_value=float('nan')).cast(unique.dtype)
        unique = paddle.concat([unique, nans])
    return unique.cast(x_dtype)
