import tensorflow as tf


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


def init_data_sparse_array(indices, values, shape):
    return tf.SparseTensor(indices=indices.data, values=values.data, dense_shape=shape)


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return x.indices, x.values, x.dense_shape
    raise Exception("not a SparseTensor")
