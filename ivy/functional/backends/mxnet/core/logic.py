"""
Collection of MXNet logic functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

# local
from ivy.functional.backends.mxnet.core.general import _handle_flat_arrays_in_out

@_handle_flat_arrays_in_out
def logical_and(x1, x2):
    return _mx.nd.logical_and(x1, x2)


@_handle_flat_arrays_in_out
def logical_or(x1, x2):
    return _mx.nd.logical_or(x1, x2)


@_handle_flat_arrays_in_out
def logical_not(x):
    return _mx.nd.logical_not(x)
