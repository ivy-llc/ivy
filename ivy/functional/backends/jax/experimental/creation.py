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
    # added this check to keep the same behaviour as tensorflow
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )
    return jax.ops.segment_min(data, segment_ids, num_segments)


def unsorted_segment_sum(
    data: JaxArray,
    segment_ids: JaxArray,
    num_segments: int,
) -> JaxArray:
    # Used the same check which is used for unsorted_segment_min as
    # the check should be same
    # Might require to change the assertion function name to
    # check_unsorted_segment_valid_params
    ivy.utils.assertions.check_unsorted_segment_min_valid_params(
        data, segment_ids, num_segments
    )
    return jax.ops.segment_sum(data, segment_ids, num_segments)



def blackman_window(
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
    return (0.42 - 0.5 * jnp.cos(2 * jnp.pi * count)) + (
        0.08 * jnp.cos(2 * jnp.pi * 2 * count)
    )

  
def trilu(
    x: JaxArray, /, *, k: int = 0, upper: bool = True, out: Optional[JaxArray] = None
) -> JaxArray:
    if upper:
        return jnp.triu(x, k)
    return jnp.tril(x, k)


def mel_weight_matrix(
    num_mel_bins: int,
    dft_length: int,
    sample_rate: int,
    lower_edge_hertz: float = 0.0,
    upper_edge_hertz: float = 3000.0,
):
    lower_edge_hertz = jnp.array(lower_edge_hertz)
    upper_edge_hertz = jnp.array(upper_edge_hertz)
    zero = jnp.array(0.0)
    hz_to_mel = lambda f: 2595 * jnp.log10(1 + f / 700)
    nyquist_hz = sample_rate / 2
    linear_freqs = jnp.linspace(0, nyquist_hz, dft_length, dtype=jnp.float32)[1:]
    spec_bin_mels = hz_to_mel(linear_freqs)[..., None]
    mel_edges = jnp.linspace(
        hz_to_mel(lower_edge_hertz),
        hz_to_mel(upper_edge_hertz),
        num_mel_bins + 2,
        dtype=jnp.float32,
    )
    mel_edges = jnp.stack([mel_edges[i : i + 3] for i in range(num_mel_bins)])
    lower_edge_mel, center_mel, upper_edge_mel = [
        t.reshape((1, num_mel_bins)) for t in jnp.split(mel_edges, 3, axis=1)
    ]
    lower_slopes = (spec_bin_mels - lower_edge_mel) / (center_mel - lower_edge_mel)
    upper_slopes = (upper_edge_mel - spec_bin_mels) / (upper_edge_mel - center_mel)
    mel_weights = jnp.maximum(zero, jnp.minimum(lower_slopes, upper_slopes))
    return jnp.pad(mel_weights, [[1, 0], [0, 0]])
