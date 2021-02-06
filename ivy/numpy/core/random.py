"""
Collection of Numpy random functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np

random_uniform = lambda low, high, size, _=None: _np.random.uniform(low, high, size)
randint = lambda low, high, size, _=None: _np.random.randint(low, high, size)
seed = lambda seed_value: _np.random.seed(seed_value)


def shuffle(x):
    _np.random.shuffle(x)
    return x
