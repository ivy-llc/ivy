"""
Collection of Numpy random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

random_uniform = lambda low=0., high=1., shape=None, dev_str=None: _np.asarray(_np.random.uniform(low, high, shape))
randint = lambda low, high, shape, dev_str=None: _np.random.randint(low, high, shape)
seed = lambda seed_value=0: _np.random.seed(seed_value)


def shuffle(x):
    _np.random.shuffle(x)
    return x
