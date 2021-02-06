"""
Collection of MXNet logic functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

logical_and = lambda x1, x2: _mx.symbol.broadcast_logical_and(x1, x2)
logical_or = lambda x1, x2: _mx.symbol.broadcast_logical_or(x1, x2)
logical_not = _mx.symbol.logical_not
