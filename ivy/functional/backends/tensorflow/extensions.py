import tensorflow as tf


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)
