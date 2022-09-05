import tensorflow as tf


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


def get_sparse_components(x):
    if isinstance(x, tf.SparseTensor):
        return x.indices, x.values, x.dense_shape
    raise Exception("not a SparseTensor")
