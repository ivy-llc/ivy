# global
from typing import Optional, Tuple
import math
import jax.numpy as jnp
import jaxlib.xla_extension

# local
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device
import ivy

# Array API Standard #
# ------------------ #


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    /,
    *,
    device: jaxlib.xla_extension.Device,
) -> Tuple[JaxArray]:
    return _to_device(
        jnp.triu_indices(n=n_rows, k=k, m=n_cols),
        device=device,
    )


def vorbis_window(
    window_length: JaxArray,
    *,
    dtype: Optional[jnp.dtype] = jnp.float32,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.array(
        [
            round(
                math.sin(
                    (ivy.pi / 2) * (math.sin(ivy.pi * (i) / (window_length * 2)) ** 2)
                ),
                8,
            )
            for i in range(1, window_length * 2)[0::2]
        ],
        dtype=dtype,
    )


def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[jnp.dtype] = None,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    window_length = window_length + 1 if periodic is True else window_length
    return jnp.array(jnp.hanning(window_length), dtype=dtype)
