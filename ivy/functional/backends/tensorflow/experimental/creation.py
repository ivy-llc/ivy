# global

from typing import Union, Optional, Tuple

import tensorflow as tf

# local
import ivy
from ivy.func_wrapper import with_unsupported_device_and_dtypes
from .. import backend_version

# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {"2.13.0 and below": {"cpu": ("bfloat16",)}},
    backend_version,
)
def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if window_length < 2:
        return tf.ones([window_length], dtype=dtype)
    if periodic is False:
        return tf.signal.kaiser_window(window_length, beta, dtype=dtype)
    else:
        return tf.signal.kaiser_window(window_length + 1, beta, dtype=dtype)[:-1]


def kaiser_bessel_derived_window(
    window_length: int,
    beta: float = 12.0,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.signal.kaiser_bessel_derived_window(window_length, beta, dtype)


def vorbis_window(
    window_length: Union[tf.Tensor, tf.Variable],
    *,
    dtype: tf.DType = tf.dtypes.float32,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.signal.vorbis_window(window_length, dtype=dtype, name=None)


def hann_window(
    size: int,
    /,
    *,
    periodic: bool = True,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if size < 2:
        return tf.ones([size], dtype=dtype)
    if periodic:
        return tf.signal.hann_window(size + 1, periodic=False, dtype=dtype)[:-1]
    else:
        return tf.signal.hann_window(size, periodic=False, dtype=dtype)


def tril_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: str,
) -> Tuple[Union[tf.Tensor, tf.Variable], ...]:
    n_cols = n_rows if n_cols is None else n_cols

    if n_rows < 0 or n_cols < 0:
        n_rows, n_cols = 0, 0

    ret = [[], []]

    for i in range(-min(k, 0), n_rows, 1):
        for j in range(0, min(n_cols, k + i + 1), 1):
            ret[0].append(i)
            ret[1].append(j)

    if device is not None:
        with tf.device(ivy.as_native_dev(device)):
            return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))

    return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))


def unsorted_segment_min(
    data: tf.Tensor,
    segment_ids: tf.Tensor,
    num_segments: Union[int, tf.Tensor],
) -> tf.Tensor:
    return tf.math.unsorted_segment_min(data, segment_ids, num_segments)


def unsorted_segment_sum(
    data: tf.Tensor,
    segment_ids: tf.Tensor,
    num_segments: Union[int, tf.Tensor],
) -> tf.Tensor:
    return tf.math.unsorted_segment_sum(data, segment_ids, num_segments)


def mel_weight_matrix(
    num_mel_bins: int,
    dft_length: int,
    sample_rate: int,
    lower_edge_hertz: float = 125.0,
    upper_edge_hertz: float = 3000.0,
):
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,
        dft_length,
        sample_rate,
        lower_edge_hertz=lower_edge_hertz,
        upper_edge_hertz=upper_edge_hertz,
    )
