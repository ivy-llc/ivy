"""
Collection of Jax general functions, wrapped to fit Ivy syntax and signature.
"""

# global
import jax as _jax
import jax.numpy as _jnp


def to_dev(x, dev):
    if dev is not None:
        if 'cpu' in dev or 'gpu' in dev:
            dev_split = dev.split(':')
            dev_str = dev_split[0]
            if len(dev_split) > 1:
                idx = int(dev_split[1])
            else:
                idx = 0
            _jax.device_put(x, _jax.devices(dev_str)[idx])
        else:
            raise Exception('Invalid device specified, must be in the form [ "cpu:idx" | "gpu:idx" ]')
    return x


# noinspection PyShadowingNames
def array(object_in, dtype_str=None, dev=None):
    if dtype_str:
        dtype = _jnp.__dict__[dtype_str]
    else:
        dtype = None
    return to_dev(_jnp.array(object_in, dtype=dtype), dev)
