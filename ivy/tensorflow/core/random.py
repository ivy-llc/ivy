"""
Collection of TensorFlow random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

random_uniform = lambda low, high, size, _=None: _tf.random.uniform(size, low, high)
randint = lambda low, high, size, _=None: _tf.random.uniform(shape=size, minval=low, maxval=high, dtype=_tf.int32)
seed = lambda seed_value: _tf.random.set_seed(seed_value)
shuffle = lambda x: _tf.random.shuffle(x)
