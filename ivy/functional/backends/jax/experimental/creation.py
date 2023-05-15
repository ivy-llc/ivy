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
    k: int = 0,
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
    dtype: jnp.dtype = jnp.float32,
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
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if size == 1:
        return jnp.array([1], dtype=dtype)
    if size == 2 and periodic is False:
        return jnp.array([0, 1], dtype=dtype)
    if periodic is False:
        return jnp.array(jnp.hanning(size), dtype=dtype)
    else:
        return jnp.array(jnp.hanning(size + 1)[:-1], dtype=dtype)


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if periodic is False:
        return jnp.array(jnp.kaiser(M=window_length, beta=beta), dtype=dtype)
    else:
        return jnp.array(jnp.kaiser(M=window_length + 1, beta=beta)[:-1], dtype=dtype)


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: jaxlib.xla_extension.Device,
) -> Tuple[JaxArray, ...]:
    return _to_device(
        jnp.tril_indices(n=n_rows, k=k, m=n_cols),
        device=device,
    )
