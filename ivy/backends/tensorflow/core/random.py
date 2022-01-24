"""
Collection of TensorFlow random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

# local
from ivy.core.device import default_device


def random_uniform(low=0., high=1., shape=None, dev_str=None):
    dev_str = default_device(dev_str)
    with _tf.device('/' + dev_str.upper()):
        return _tf.random.uniform(shape if shape else (), low, high)


def random_normal(mean=0., std=1., shape=None, dev_str=None):
    dev_str = default_device(dev_str)
    with _tf.device('/' + dev_str.upper()):
        return _tf.random.normal(shape if shape else (), mean, std)


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, dev_str=None):
    if not replace:
        raise Exception('TensorFlow does not support multinomial without replacement')
    dev_str = default_device(dev_str)
    with _tf.device('/' + dev_str.upper()):
        if probs is None:
            probs = _tf.ones((batch_size, population_size,)) / population_size
        return _tf.random.categorical(_tf.math.log(probs), num_samples)


def randint(low, high, shape, dev_str=None):
    dev_str = default_device(dev_str)
    with _tf.device('/' + dev_str.upper()):
        return _tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=_tf.int32)


seed = lambda seed_value=0: _tf.random.set_seed(seed_value)
shuffle = lambda x: _tf.random.shuffle(x)
