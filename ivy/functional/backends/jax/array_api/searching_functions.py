# global
import jax.numpy as jnp
from typing import Optional


#local
from ivy.functional.backends.jax import JaxArray

def argmin(x : JaxArray, 
    axis: Optional[int] = None, 
    out: Optional[JaxArray] = None, 
    keepdims: bool = False
    ) -> JaxArray:

    ret = jnp.argmin(x,axis=axis,out=out,keepdims=keepdims)
    return (ret)
