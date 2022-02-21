"""
Collection of TensorFlow math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf

sin = _tf.sin


def cos(x: _tf.Tensor) -> _tf.Tensor:
    return _tf.cos(x)


tan = _tf.tan
asin = _tf.asin
acos = _tf.acos
atan = _tf.atan
atan2 = _tf.atan2
sinh = _tf.math.sinh
cosh = _tf.math.cosh
tanh = _tf.math.tanh
asinh = _tf.math.asinh
acosh = _tf.math.acosh
atanh = _tf.math.atanh
log = _tf.math.log
exp = _tf.math.exp
erf = _tf.math.erf
