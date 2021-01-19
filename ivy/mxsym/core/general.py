"""
Collection of MXNet general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import time as _time
import mxnet as _mx
_round = round


def array(object_in, dtype_str=None, dev=None):
    if isinstance(object_in, _mx.symbol.symbol.Symbol):
        return object_in
    _mx_nd_array = _mx.nd.array(object_in)
    return _mx.sym.BlockGrad(_mx.symbol.Variable(str(_time.time()).replace('.', ''), shape=_mx_nd_array.shape,
                                                 dtype=dtype_str, init=_mx.init.Constant(value=_mx_nd_array)))
