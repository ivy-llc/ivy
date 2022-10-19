from typing import Union, Tuple
from collections import namedtuple

import tensorflow as tf

from . import version
from ivy.func_wrapper import with_unsupported_dtypes


@with_unsupported_dtypes(
    {
        "1.11.0 and below": ("float16",),
    },
    version,
)
def unique_all(
    x: Union[tf.Tensor, tf.Variable],
    /,
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
    flat_tensor = tf.reshape(x, [-1])
    values, inverse_indices, counts = tf.unique_with_counts(tf.sort(flat_tensor))
    tensor_list = flat_tensor.numpy().tolist()
    if (
        x.dtype.is_floating
        and tf.math.reduce_sum(tf.cast(tf.math.is_nan(values), "float32")).numpy()
    ):
        unique_nan = tf.math.is_nan(values).numpy()

        nan_index = tf.where(tf.math.is_nan(flat_tensor)).numpy().reshape([-1])
        non_nan_index = tf.experimental.numpy.array(
            [tensor_list.index(val) for val in values if not tf.math.is_nan(val)]
        )

        indices = tf.experimental.numpy.full(
            fill_value=float("NaN"), shape=values.shape
        ).numpy()

        indices[unique_nan] = nan_index
        indices[~unique_nan] = non_nan_index
        indices = tf.convert_to_tensor(indices)
    else:
        values_list = values.numpy().tolist()
        indices = [tensor_list.index(val) for val in values_list]
        indices = tf.convert_to_tensor(indices)
        inverse_indices = [values_list.index(val) for val in tensor_list]
        inverse_indices = tf.convert_to_tensor(inverse_indices)

    return Results(
        tf.cast(values, x.dtype),
        tf.cast(indices, dtype=tf.int64),
        tf.cast(tf.reshape(inverse_indices, x.shape), dtype=tf.int64),
        tf.cast(counts, dtype=tf.int64),
    )
