# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
import ivy.functional.backends.paddle as paddle_backend
from ivy.func_wrapper import with_supported_dtypes

# local
from . import backend_version


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
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

    x_dtype = x.dtype
    if axis is not None:
        axis = axis % x.ndim
    values, inverse_indices, counts = paddle.unique(
        x,
        return_index=False,  # which occurrences of the unique values are picked is
        # inconsistent in some cases, so calculate the indices manually below
        return_counts=True,
        return_inverse=True,
        axis=axis,
    )

    unique_nan = paddle.isnan(values)
    idx_dtype = inverse_indices.dtype
    if paddle.any(unique_nan):
        nan_index = paddle.where(paddle.isnan(x))
        non_nan_index = [
            x.tolist().index(val) for val in values if not paddle.isnan(val)
        ]
        indices = values.clone().to(idx_dtype)
        indices[unique_nan] = nan_index[0]
        inverse_indices[paddle.isnan(x)] = paddle.where(unique_nan)[0][0]
        counts[unique_nan] = 1
        indices[~unique_nan] = paddle.to_tensor(non_nan_index, dtype=idx_dtype)
    else:
        decimals = paddle.arange(inverse_indices.numel()) / inverse_indices.numel()
        inv_sorted = (inverse_indices.astype(decimals.dtype) + decimals).argsort()
        tot_counts = paddle.concat(
            (paddle.zeros((1,), dtype=counts.dtype), counts.cumsum(axis=0))
        )[:-1]
        indices = inv_sorted[tot_counts].astype(idx_dtype)

    if not by_value:
        sort_idx = paddle.argsort(indices)
    else:
        if axis is None:
            axis = 0
        values_ = paddle.moveaxis(values, axis, 0)
        values_ = paddle.reshape(values_, (values_.shape[0], -1))
        sort_idx = paddle.to_tensor(
            [
                i[0]
                for i in sorted(
                    enumerate(values_.numpy().tolist()), key=lambda x: tuple(x[1])
                )
            ]
        )
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


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def unique_counts(x: paddle.Tensor, /) -> Tuple[paddle.Tensor, paddle.Tensor]:
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
    return Results(unique.cast(x.dtype), counts)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def unique_inverse(
    x: paddle.Tensor,
    /,
    *,
    axis: Optional[int] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    if x.dtype not in [paddle.int32, paddle.int64, paddle.float32, paddle.float64]:
        x = x.cast("float32")

    if axis is not None:
        unique, inverse_val = paddle.unique(x, return_inverse=True, axis=axis)

    if axis is None:
        axis = 0

    nan_idx = paddle.where(paddle.isnan(x) > 0)
    nan_count = paddle.count_nonzero(nan_idx).numpy()[0]

    if nan_count > 0:
        inverse_val[nan_idx] = len(unique)
        unique_nan = paddle.full(shape=[1, nan_count], fill_value=float("nan")).cast(
            x.dtype
        )
        unique = paddle.concat(
            [unique.astype(x.dtype), paddle.reshape(unique_nan, [nan_count])],
            axis=-1,
        )
    inverse_val = paddle.reshape(inverse_val, shape=x.shape)
    Results = namedtuple("Results", ["values", "inverse_indices"])
    return Results(unique.cast(x.dtype), inverse_val)


@with_supported_dtypes(
    {"2.6.0 and below": ("float32", "float64", "int32", "int64")}, backend_version
)
def unique_values(
    x: paddle.Tensor, /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    nan_count = paddle.sum(paddle.isnan(x))
    unique = paddle.unique(x)
    if nan_count > 0:
        nans = paddle.full(shape=[nan_count], fill_value=float("nan")).cast(
            unique.dtype
        )
        unique = paddle.concat([unique, nans])
    return unique.cast(x.dtype)
