"""
Collection of Jax general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax
import math as _math
import numpy as _onp
import jax.numpy as jnp
import jaxlib as _jaxlib
from numbers import Number
from collections import Iterable
from operator import mul as _mul
from functools import reduce as _reduce
from jaxlib.xla_extension import Buffer
import multiprocessing as _multiprocessing
from haiku._src.data_structures import FlatMapping

# local
import ivy
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype
from ivy.functional.backends.jax.device import to_dev, dev as callable_dev

DTYPE_TO_STR = {jnp.dtype('int8'): 'int8',
                jnp.dtype('int16'): 'int16',
                jnp.dtype('int32'): 'int32',
                jnp.dtype('int64'): 'int64',
                jnp.dtype('uint8'): 'uint8',
                jnp.dtype('uint16'): 'uint16',
                jnp.dtype('uint32'): 'uint32',
                jnp.dtype('uint64'): 'uint64',
                jnp.dtype('bfloat16'): 'bfloat16',
                jnp.dtype('float16'): 'float16',
                jnp.dtype('float32'): 'float32',
                jnp.dtype('float64'): 'float64',
                jnp.dtype('bool'): 'bool',

                jnp.int8: 'int8',
                jnp.int16: 'int16',
                jnp.int32: 'int32',
                jnp.int64: 'int64',
                jnp.uint8: 'uint8',
                jnp.uint16: 'uint16',
                jnp.uint32: 'uint32',
                jnp.uint64: 'uint64',
                jnp.bfloat16: 'bfloat16',
                jnp.float16: 'float16',
                jnp.float32: 'float32',
                jnp.float64: 'float64',
                jnp.bool_: 'bool'}

DTYPE_FROM_STR = {'int8': jnp.dtype('int8'),
                  'int16': jnp.dtype('int16'),
                  'int32': jnp.dtype('int32'),
                  'int64': jnp.dtype('int64'),
                  'uint8': jnp.dtype('uint8'),
                  'uint16': jnp.dtype('uint16'),
                  'uint32': jnp.dtype('uint32'),
                  'uint64': jnp.dtype('uint64'),
                  'bfloat16': jnp.dtype('bfloat16'),
                  'float16': jnp.dtype('float16'),
                  'float32': jnp.dtype('float32'),
                  'float64': jnp.dtype('float64'),
                  'bool': jnp.dtype('bool')}


# Helpers #
# --------#



def _to_array(x):
    if isinstance(x, jax.interpreters.ad.JVPTracer):
        return _to_array(x.primal)
    elif isinstance(x, jax.interpreters.partial_eval.DynamicJaxprTracer):
        return _to_array(x.aval)
    return x


# API #
# ----#











def dtype_bits(dtype_in):
    dtype_str = dtype_to_str(dtype_in)
    if 'bool' in dtype_str:
        return 1
    return int(dtype_str.replace('uint', '').replace('int', '').replace('bfloat', '').replace('float', ''))




minimum = jnp.minimum
maximum = jnp.maximum




def cast(x, dtype):
    return x.astype(dtype_from_str(dtype))


astype = cast





































# noinspection PyShadowingNames
def identity(n, dtype='float32', batch_shape=None, dev=None):
    dtype = jnp.__dict__[dtype]
    mat = jnp.identity(n, dtype=dtype)
    if batch_shape is None:
        return_mat = mat
    else:
        reshape_dims = [1] * len(batch_shape) + [n, n]
        tile_dims = list(batch_shape) + [1, 1]
        return_mat = jnp.tile(jnp.reshape(mat, reshape_dims), tile_dims)
    return to_dev(return_mat, default_device(dev))








def dtype(x, as_str=False):
    dt = x.dtype
    if as_str:
        return dtype_to_str(dt)
    return dt


def dtype_to_str(dtype_in):
    if isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_TO_STR[dtype_in]


def dtype_from_str(dtype_in):
    if not isinstance(dtype_in, str):
        return dtype_in
    return DTYPE_FROM_STR[dtype_in]


compile = lambda fn, dynamic=True, example_inputs=None, static_argnums=None, static_argnames=None: \
    jax.jit(fn, static_argnums=static_argnums, static_argnames=static_argnames)
current_framework_str = lambda: 'jax'
current_framework_str.__name__ = 'current_framework_str'


