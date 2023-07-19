# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import with_unsupported_device_and_dtypes

# local
from . import backend_version


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}}, backend_version
)
def unique_all(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        x, x_dtype = x.cast("float32"), x.dtype
    else:
        x_dtype = x.dtype
    if axis is not None:
        axis = axis % x.ndim
    values, indices, inverse_indices, counts = paddle.unique(
        x,
        return_index=True,
        return_counts=True,
        return_inverse=True,
        axis=axis,
    )

    nan_count = paddle.sum(paddle.isnan(x))
    if nan_count.item() > 0:
        nan = paddle.to_tensor([float("nan")] * nan_count.item(), dtype=values.dtype)
        values = paddle.concat((values, nan))
        nan_idx = paddle.nonzero(paddle.isnan(x).astype(float).flatten()).flatten()
        indices = paddle.concat((indices, nan_idx))
        inverse_indices = paddle.put_along_axis(
            arr=inverse_indices, indices=nan_idx, values=values.shape, axis=0
        )
        counts = paddle.concat(
            (counts, paddle.ones(shape=nan_count, dtype=counts.dtype))
        )

    if not by_value:
        sort_idx = paddle.argsort(indices)
        values = paddle.gather(values, sort_idx, axis=axis)
        counts = paddle.gather(counts, sort_idx)
        indices = paddle.gather(indices, sort_idx)
        inv_sort_idx = paddle_backend.invert_permutation(sort_idx)
        inverse_indices = paddle_backend.vmap(lambda y: paddle.gather(inv_sort_idx, y))(
            inverse_indices
        )

    return Results(
        values.cast(x_dtype),
        indices,
        inverse_indices,
        counts,
    )


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}}, backend_version
)
def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        x, x_dtype = x.cast("float32"), x.dtype
    else:
        x_dtype = x.dtype

    unique, counts = paddle.unique(x, return_counts=True)
    nan_count = paddle.count_nonzero(paddle.where(paddle.isnan(x) > 0)).numpy()[0]

    if nan_count > 0:
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float("nan")).cast(
            x.dtype
        )
        counts_nan = paddle.full(shape=[1, nan_count], fill_value=1).cast(x.dtype)
        unique = paddle.concat(
            [unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])], axis=0
        )
        counts = paddle.concat(
            [counts.astype(x.dtype), paddle.reshape(counts_nan, [nan_count])], axis=0
        )

    Results = namedtuple("Results", ["values", "counts"])
    return Results(unique.cast(x_dtype), counts)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}}, backend_version
)
def unique_inverse(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        x, x_dtype = x.cast("float32"), x.dtype
    else:
        x_dtype = x.dtype
    unique, inverse_val = paddle.unique(x, return_inverse=True)
    nan_idx = paddle.where(paddle.isnan(x) > 0)
    nan_count = paddle.count_nonzero(nan_idx).numpy()[0]

    if nan_count > 0:
        inverse_val[nan_idx] = len(unique)
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float("nan")).cast(
            x.dtype
        )
        unique = paddle.concat(
            input=[unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])],
            axis=-1,
        )
    inverse_val = paddle.reshape(inverse_val, shape=x.shape)
    Results = namedtuple("Results", ["values", "inverse_indices"])
    return Results(unique.cast(x_dtype), inverse_val)


@with_unsupported_device_and_dtypes(
    {"2.5.0 and below": {"cpu": ("complex",)}}, backend_version
)
def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        x, x_dtype = x.cast("float32"), x.dtype
    else:
        x_dtype = x.dtype
    nan_count = paddle.sum(paddle.isnan(x))
    unique = paddle.unique(x)
    if nan_count > 0:
        nans = paddle.full(shape=[nan_count], fill_value=float("nan")).cast(
            unique.dtype
        )
        unique = paddle.concat([unique, nans])
    return unique.cast(x_dtype)
