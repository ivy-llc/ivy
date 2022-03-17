# global
import ivy
_round = round
import logging
import mxnet as _mx
import numpy as _np
import math as _math
from numbers import Number
from operator import mul as _mul
from functools import reduce as _reduce
import multiprocessing as _multiprocessing

# local
from ivy.functional.ivy.old import default_dtype
from ivy.functional.ivy.device import default_device
from ivy.functional.backends.mxnet.device import _callable_dev
from ivy.functional.backends.mxnet import _handle_flat_arrays_in_out, _mxnet_init_context,\
    _scalar_or_flat_array_to_scalar, _handle_flat_arrays_in


def is_array(x, exclusive=False):
    if isinstance(x, _mx.ndarray.ndarray.NDArray):
        if exclusive and x.grad is not None:
            return False
        return True
    return False


copy_array = lambda x: x.copy()

@_handle_flat_arrays_in_out
def array_equal(x0, x1):
    if ivy.dtype(x0, as_str=True) == 'bool':
        x0 = x0.astype('int32')
    if ivy.dtype(x1, as_str=True) == 'bool':
        x1 = x1.astype('int32')
    return _mx.nd.min(_mx.nd.broadcast_equal(x0, x1)) == 1