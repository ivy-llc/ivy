from typing import Optional, Tuple, Union

import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray


def median(
    input: JaxArray,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.median(
        input,
        axis=axis,
        keepdims=keepdims,
        out=out,
    )
