"""
Collection of TensorFlow random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

random_uniform = lambda low=0., high=1., shape=None, dev_str='cpu':\
    _tf.random.uniform(shape if shape else (), low, high)


def multinomial(population_size, num_samples, probs=None, replace=True):
    if not replace:
        raise Exception('TensorFlow does not support multinomial without replacement')
    if probs is None:
        probs = _tf.ones((1, population_size,)) / population_size
    return _tf.random.categorical(_tf.math.log(probs), num_samples)


randint = lambda low, high, shape, dev_str='cpu':\
    _tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=_tf.int32)
seed = lambda seed_value=0: _tf.random.set_seed(seed_value)
shuffle = lambda x: _tf.random.shuffle(x)
