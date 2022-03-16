import jax as _jax
import jax.numpy as _jnp
import jaxlib as _jaxlib

# noinspection PyUnresolvedReferences,PyProtectedMember
def is_array(x, exclusive=False):
    if exclusive:
        return isinstance(x, (_jax.interpreters.xla._DeviceArray,
                              _jaxlib.xla_extension.DeviceArray, Buffer))
    return isinstance(x, (_jax.interpreters.xla._DeviceArray,
                          _jaxlib.xla_extension.DeviceArray, Buffer,
                          _jax.interpreters.ad.JVPTracer,
                          _jax.core.ShapedArray,
                          _jax.interpreters.partial_eval.DynamicJaxprTracer))

copy_array = _jnp.array