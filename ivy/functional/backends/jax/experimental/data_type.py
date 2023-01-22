# global
from typing import Union

import jax.numpy as jnp

# local
from ivy.functional.backends.jax import ivy_dtype_dict


def is_native_dtype(dtype_in: Union[jnp.dtype, str], /) -> bool:
    if dtype_in in ivy_dtype_dict:
        return True
    else:
        return False
