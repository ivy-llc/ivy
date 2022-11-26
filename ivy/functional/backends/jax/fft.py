from typing import Optional, Sequence
import jax.numpy as jnp

import ivy
from ivy.functional.backends.jax import JaxArray

def fft2(
        a: ivy.Array,
        # s: Optional[Sequence[int]] = None,
        # axes: Sequence[int] = [-2, -1],
        # norm: Optional[str] = None
) -> JaxArray:
    s = None
    axes = [-2, -1]
    norm = None
    return jnp.fft.fft2(a, s=s, axes=axes, norm=norm)
