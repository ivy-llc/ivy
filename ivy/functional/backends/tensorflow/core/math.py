"""
Collection of TensorFlow math functions, wrapped to fit Ivy syntax and signature.
"""

# global
import math
import tensorflow as _tf

sin = _tf.sin
cos = _tf.cos
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
erf = _tf.math.erf

def exp(x):
    if isinstance(x, float):
        return math.exp(x)
    
    if x.dtype.is_integer or (x.dtype == _tf.float16):
        return _tf.math.exp(_tf.cast(x, dtype=_tf.float32))
    return _tf.math.exp(x)