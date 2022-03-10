"""
Collection of MXNet math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx
import math as _math

cos = lambda x: _math.cos(x) if isinstance(x, float) else _mx.nd.cos(x)
tan = lambda x: _math.tan(x) if isinstance(x, float) else _mx.nd.tan(x)
asin = lambda x: _math.asin(x) if isinstance(x, float) else _mx.nd.arcsin(x)
acos = lambda x: _math.acos(x) if isinstance(x, float) else _mx.nd.arccos(x)
atan = lambda x: _math.atan(x) if isinstance(x, float) else _mx.nd.arctan(x)
atan2 = lambda x, y: _math.atan2(x, y) if isinstance(x, float) else _mx.np.arctan2(x.as_np_ndarray(), y.as_np_ndarray()).as_nd_ndarray()
sinh = lambda x: _math.sinh(x) if isinstance(x, float) else _mx.nd.sinh(x)
cosh = lambda x: _math.cosh(x) if isinstance(x, float) else _mx.nd.cosh(x)
tanh = lambda x: _math.tanh(x) if isinstance(x, float) else _mx.nd.tanh(x)
asinh = lambda x: _math.asinh(x) if isinstance(x, float) else _mx.nd.arcsinh(x)
acosh = lambda x: _math.acosh(x) if isinstance(x, float) else _mx.nd.arccosh(x)
atanh = lambda x: _math.atanh(x) if isinstance(x, float) else _mx.nd.arctanh(x)
log = lambda x: _math.log(x) if isinstance(x, float) else _mx.nd.log(x)
exp = lambda x: _math.exp(x) if isinstance(x, float) else _mx.nd.exp(x)
erf = lambda x: _math.erf(x) if isinstance(x, float) else _mx.nd.erf(x)
