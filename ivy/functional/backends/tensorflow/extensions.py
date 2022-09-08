import ivy
from ivy.functional.ivy.extensions import _verify_coo_components
import tensorflow as tf


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


def native_sparse_array(data=None, *, indices=None, values=None, dense_shape=None):
    if data is not None:
        if ivy.exists(indices) or ivy.exists(values) or ivy.exists(dense_shape):
            raise Exception(
                "only specify either data or or all three components: \
                indices, values and dense_shape"
            )
        assert ivy.is_native_sparse_array(data), "not a sparse array"
        return data
    _verify_coo_components(indices=indices, values=values, dense_shape=dense_shape)
    return tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return x.indices, x.values, x.dense_shape
    raise Exception("not a SparseTensor")
