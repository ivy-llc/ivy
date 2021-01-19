"""
Collection of TensorFlow general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import tensorflow as _tf


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    dtype = _tf.__dict__[dtype_str] if dtype_str else dtype_str
    if dev:
        with _tf.device('/' + dev.upper()):
            ret_val = _tf.convert_to_tensor(object_in, dtype=dtype)
    else:
        ret_val = _tf.convert_to_tensor(object_in, dtype=dtype)
    return ret_val
