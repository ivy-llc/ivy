# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor
from typing import Union, Tuple
from collections import namedtuple


def unique_all(x: Tensor) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
    UniqueAll = namedtuple(typename = 'unique_all', field_names = ['values', 'indices', 'inverse_indices', 'counts'])
    
    flat_tensor = tf.reshape(x, [-1])
    
    values, inverse_indices, counts = tf.unique_with_counts(flat_tensor)
    
    unique_nan = tf.math.is_nan(values).numpy()
    idx_dtype = inverse_indices.dtype

    if tf.math.reduce_sum(tf.cast(tf.math.is_nan(values), 'int32')).numpy():
        nan_index = tf.where(tf.math.is_nan(flat_tensor)).numpy().reshape([-1])
        non_nan_index = [flat_tensor.numpy().tolist().index(val) for val in values if not tf.math.is_nan(val)]

        indices = tf.experimental.numpy.full(fill_value = float('NaN'), shape = values.shape).numpy()
        
        indices[unique_nan] = nan_index
        indices[~unique_nan] = tf.constant(non_nan_index, dtype = idx_dtype).numpy()
    else:
        tensor_list = flat_tensor.numpy().tolist()
        indices = tf.constant([tensor_list.index(val) for val in values])
    
    return UniqueAll(tf.cast(values, x.dtype), tf.constant(indices, dtype = idx_dtype), tf.reshape(inverse_indices, x.shape), counts)


def unique_inverse(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    out = namedtuple('unique_inverse', ['values', 'inverse_indices'])
    values, inverse_indices = tf.unique(tf.reshape(x, -1))
    inverse_indices = tf.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: Tensor) \
        -> Tensor:
    return tf.unique(tf.reshape(x, [-1]))[0]


def unique_counts(x: Tensor) \
        -> Tuple[Tensor, Tensor]:
    uc = namedtuple('uc', ['values', 'counts'])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)
