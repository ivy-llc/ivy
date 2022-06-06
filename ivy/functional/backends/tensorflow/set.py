# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Tuple
from collections import namedtuple

import ivy


def unique_all(
    x: Union[tf.Tensor, tf.Variable]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    UniqueAll = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
    )

    flat_tensor = tf.reshape(x, [-1])
    values, inverse_indices, counts = tf.unique_with_counts(flat_tensor)
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
        indices = tf.experimental.numpy.array(
            [tensor_list.index(val) for val in values]
        )
        indices = tf.convert_to_tensor(indices)

    return UniqueAll(
        tf.cast(values, x.dtype),
        tf.cast(indices, dtype="int32"),
        tf.reshape(inverse_indices, x.shape),
        counts,
    )


def unique_inverse(
    x: Union[tf.Tensor, tf.Variable],
) -> Tuple[tf.Tensor, tf.Tensor]:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = tf.unique(tf.reshape(x, -1))
    inverse_indices = tf.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(
    x: Union[tf.Tensor, tf.Variable],
    out: Union[tf.Tensor, tf.Variable] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.unique(tf.reshape(x, [-1]))[0]
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


def unique_counts(
    x: Union[tf.Tensor, tf.Variable],
) -> Tuple[tf.Tensor, tf.Tensor]:
    uc = namedtuple("uc", ["values", "counts"])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)
