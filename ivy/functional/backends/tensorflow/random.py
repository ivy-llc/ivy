"""
Collection of TensorFlow random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as tf

# local
from ivy.functional.ivy.device import default_device


# Extra #
# ------#

def random_uniform(low=0., high=1., shape=None, dev=None):
    with tf.device(default_device(dev)):
        return tf.random.uniform(shape if shape else (), low, high)


def random_normal(mean=0., std=1., shape=None, dev=None):
    dev = default_device(dev)
    with tf.device('/' + dev.upper()):
        return tf.random.normal(shape if shape else (), mean, std)


def multinomial(population_size, num_samples, batch_size, probs=None, replace=True, dev=None):
    if not replace:
        raise Exception('TensorFlow does not support multinomial without replacement')
    dev = default_device(dev)
    with tf.device('/' + dev.upper()):
        if probs is None:
            probs = tf.ones((batch_size, population_size,)) / population_size
        return tf.random.categorical(tf.math.log(probs), num_samples)


def randint(low, high, shape, dev=None):
    dev = default_device(dev)
    with tf.device('/' + dev.upper()):
        return tf.random.uniform(shape=shape, minval=low, maxval=high, dtype=tf.int32)


seed = lambda seed_value=0: tf.random.set_seed(seed_value)
shuffle = lambda x: tf.random.shuffle(x)
