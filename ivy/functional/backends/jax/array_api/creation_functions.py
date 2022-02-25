# global
import jax.numpy as jnp
from typing import Union, Optional, Tuple

# local
import ivy
from ivy.functional.backends.jax import JaxArray


# noinspection PyShadowingNames
def ones(shape: Union[int, Tuple[int, ...]],
         dtype: Optional[jnp.dtype] = None,
         device: Optional[str] = None) \
        -> JaxArray:
    return ivy.to_dev(jnp.ones(shape, ivy.dtype_from_str(ivy.default_dtype(dtype))), ivy.default_device(device))
