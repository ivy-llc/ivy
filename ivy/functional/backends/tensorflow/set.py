# global
import tensorflow as tf
from typing import Tuple, Union, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def unique_all(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    by_value: bool = True,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    Results = namedtuple(
        "Results",
        ["values", "indices", "inverse_indices", "counts"],
    )

    if axis is None:
        x = tf.reshape(x, shape=(-1,))
        axis = 0

    values, inverse_indices, counts = tf.raw_ops.UniqueWithCountsV2(
        x=x,
        axis=tf.constant([axis], dtype=tf.int32),
    )

    tensor_list = x.numpy().tolist()
    if x.dtype.is_floating and tf.reduce_any(tf.math.is_nan(values)):
        unique_nan = tf.math.is_nan(values)
        nan_index = tf.reshape(tf.where(tf.math.is_nan(x)), [-1])
        non_nan_index = tf.experimental.numpy.array(
            [tensor_list.index(val) for val in values if not tf.math.is_nan(val)]
        )
        indices = tf.experimental.numpy.full(
            fill_value=float("NaN"), shape=values.shape
        ).numpy()
        indices[unique_nan] = nan_index
        indices[~unique_nan] = non_nan_index
    else:
        decimal = tf.range(tf.size(inverse_indices)) / tf.size(inverse_indices)
        inv_sorted = tf.argsort(
            tf.cast(inverse_indices, dtype=decimal.dtype) + decimal
        ).numpy()
        total_counts = tf.concat(
            [tf.zeros((1,), dtype=counts.dtype), tf.cumsum(counts, axis=0)[:-1]], 0
        )
        indices = inv_sorted[total_counts]

    if by_value:
        values_ = tf.experimental.numpy.moveaxis(values, axis, 0)
        values_ = tf.reshape(values_, (values_.shape[0], -1))
        sort_idx = tf.stack(
            [i[0] for i in sorted(list(enumerate(values_)), key=lambda x: tuple(x[1]))]
        )
        sort_idx = tf.cast(sort_idx, tf.int32)
        values = tf.gather(values, sort_idx, axis=axis)
        counts = tf.gather(counts, sort_idx)
        indices = tf.gather(indices, sort_idx)
        inv_sort_idx = tf.math.invert_permutation(sort_idx)
        inverse_indices = tf.map_fn(
            lambda y: tf.gather(inv_sort_idx, y), inverse_indices
        )

    return Results(
        tf.cast(values, dtype=x.dtype),
        tf.cast(indices, dtype=tf.int64),
        tf.cast(inverse_indices, dtype=tf.int64),
        tf.cast(counts, dtype=tf.int64),
    )


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def unique_counts(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    Results = namedtuple("Results", ["values", "counts"])
    v, _, c = tf.unique_with_counts(tf.sort(tf.reshape(x, [-1])))
    v = tf.cast(v, dtype=x.dtype)
    c = tf.cast(c, dtype=tf.int64)
    return Results(v, c)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def unique_inverse(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    Results = namedtuple("Results", ["values", "inverse_indices"])
    flat_tensor = tf.reshape(x, -1)
    values = tf.unique(tf.sort(flat_tensor))[0]
    values = tf.cast(values, dtype=x.dtype)

    values_list = values.numpy().tolist()
    inverse_indices = [values_list.index(val) for val in flat_tensor.numpy().tolist()]
    inverse_indices = tf.reshape(tf.convert_to_tensor(inverse_indices), x.shape)
    inverse_indices = tf.cast(inverse_indices, dtype=tf.int64)
    return Results(values, inverse_indices)


@with_unsupported_dtypes({"2.13.0 and below": ("complex",)}, backend_version)
def unique_values(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.unique(tf.reshape(x, [-1]))[0]
    return tf.sort(ret)
