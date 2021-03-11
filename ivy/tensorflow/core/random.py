"""
Collection of TensorFlow random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

random_uniform = lambda low=0., high=1., shape=None, dev=None: _tf.random.uniform(shape, low, high)
randint = lambda low, high, shape, dev=None: _tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=_tf.int32)
seed = lambda seed_value=0: _tf.random.set_seed(seed_value)
shuffle = lambda x: _tf.random.shuffle(x)
