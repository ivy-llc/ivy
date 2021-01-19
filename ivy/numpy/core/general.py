"""
Collection of Numpy general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import numpy as _np


def to_dev(x, dev):
    if dev is not None:
        if 'gpu' in dev:
            raise Exception('Native Numpy does not support GPU placement, consider using Jax instead')
        elif 'cpu' in dev:
            pass
        else:
            raise Exception('Invalid device specified, must be in the form [ "cpu:idx" | "gpu:idx" ]')
    return x


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _np.__dict__[dtype_str]
    else:
        dtype = None
    return to_dev(_np.array(object_in, dtype=dtype), dev)
