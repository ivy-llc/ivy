from typing import Optional, Sequence
import jax.numpy as jnp

from ivy.functional.backends.jax import JaxArray


def fft2(
        a: JaxArray,
        s: Optional[Sequence[int]] = None,
        axes: Sequence[int] = [-2, -1],
        norm: Optional[str] = None
) -> JaxArray:
    return jnp.fft.fft2(a, s=s, axes=axes, norm=norm)
