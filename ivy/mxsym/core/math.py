"""
Collection of MXNet math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

sin = _mx.symbol.sin
cos = _mx.symbol.cos
tan = _mx.symbol.tan
asin = _mx.symbol.arcsin
acos = _mx.symbol.arccos
atan = _mx.symbol.arctan


def atan2(*_):
    raise Exception('mxnet symbolic does not support ivy.math.atan2().')


sinh = _mx.symbol.sinh
cosh = _mx.symbol.cosh
tanh = _mx.symbol.tanh
asinh = _mx.symbol.arcsinh
acosh = _mx.symbol.arccosh
atanh = _mx.symbol.arctanh
log = _mx.symbol.log
exp = _mx.symbol.exp
