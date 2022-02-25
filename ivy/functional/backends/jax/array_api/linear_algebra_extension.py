# global
from typing  import Union, Optional, Tuple, Literal
import jax as _jax
# noinspection PyPackageRequirements
import jaxlib
import jax.numpy as jnp
from jaxlib.xla_extension import Buffer


# local
from ivy import inf

JaxArray = (_jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer)


def vector_norm(x: JaxArray, 
                p: Union[int, float, Literal[inf, - inf]] = 2, 
                axis: Optional[Union[int, Tuple[int]]] = None, 
                keepdims: bool = False)\
                    -> JaxArray:
                

    jnp_normalized_vector = None

    if axis == None:
       jnp_normalized_vector = jnp.linalg.norm(jnp.ravel(x), p, axis, keepdims)

    else:
        jnp_normalized_vector = jnp.linalg.norm(x, p, axis, keepdims)

    if jnp_normalized_vector.shape  == tuple():
        return  jnp.expand_dims(jnp_normalized_vector, 0)
    return jnp_normalized_vector
