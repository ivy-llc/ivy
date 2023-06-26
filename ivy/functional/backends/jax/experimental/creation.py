# global
from typing import Optional, Tuple
import math
import jax
import jax.numpy as jnp
import jaxlib.xla_extension

# local
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device
import ivy

# Array API Standard #
# ------------------ #


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
    if size < 2:
        return jnp.ones([size], dtype=dtype)
    if periodic:
        count = jnp.arange(size) / size
    else:
        count = jnp.linspace(start=0, stop=size, num=size)
    return (0.5 - 0.5 * jnp.cos(2 * jnp.pi * count)).astype(dtype)


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if window_length < 2:
        return jnp.ones([window_length], dtype=dtype)
    if periodic is False:
        return jnp.kaiser(M=window_length, beta=beta).astype(dtype)
    else:
        return jnp.kaiser(M=window_length + 1, beta=beta)[:-1].astype(dtype)


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


def unsorted_segment_min(
    data: JaxArray,
    segment_ids: JaxArray,
    num_segments: int,
) -> JaxArray:
    # added these check to keep the same behaviour as tensorflow
    if not (isinstance(num_segments, int)):
        raise ValueError("num_segments must be of integer type")

    valid_dtypes = [jnp.int32, jnp.int64]

    if segment_ids.dtype not in valid_dtypes:
        raise ValueError("segment_ids must have an int32 or int64 dtype")

    if num_segments <= 0:
        raise ValueError("num_segments must be positive")

    if data.shape[0] != segment_ids.shape[0]:
        raise ValueError("The length of segment_ids should be equal to data.shape[0].")

    if jnp.max(segment_ids) >= num_segments:
        error_message = (
            f"segment_ids[{jnp.argmax(segment_ids)}] = "
            f"{jnp.max(segment_ids)} is out of range [0, {num_segments})"
        )
        raise ValueError(error_message)

    return jax.ops.segment_min(data, segment_ids, num_segments)
