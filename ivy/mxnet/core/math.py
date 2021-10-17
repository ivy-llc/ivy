"""
Collection of MXNet math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import mxnet as _mx

sin = _mx.nd.sin
cos = _mx.nd.cos
tan = _mx.nd.tan
asin = _mx.nd.arcsin
acos = _mx.nd.arccos
atan = _mx.nd.arctan
atan2 = lambda x1, x2: _mx.np.arctan2(x1.as_np_ndarray(), x2.as_np_ndarray()).as_nd_ndarray()
sinh = _mx.nd.sinh
cosh = _mx.nd.cosh
tanh = _mx.nd.tanh
asinh = _mx.nd.arcsinh
acosh = _mx.nd.arccosh
atanh = _mx.nd.arctanh
log = _mx.nd.log
exp = _mx.nd.exp
erf = _mx.nd.erf
