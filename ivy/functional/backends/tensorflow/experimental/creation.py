# global

from typing import Union, Optional, Tuple

import tensorflow as tf

# local
import ivy

# Array API Standard #
# -------------------#


def triu_indices(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: int = 0,
    /,
    *,
    device: str,
) -> Tuple[Union[tf.Tensor, tf.Variable]]:
    n_cols = n_rows if n_cols is None else n_cols

    if n_rows < 0 or n_cols < 0:
        n_rows, n_cols = 0, 0

    ret = [[], []]

    for i in range(0, min(n_rows, n_cols - k), 1):
        for j in range(max(0, k + i), n_cols, 1):
            ret[0].append(i)
            ret[1].append(j)

    if device is not None:
        with tf.device(ivy.as_native_dev(device)):
            return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))

    return tuple(tf.convert_to_tensor(ret, dtype=tf.int64))


def kaiser_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if periodic is False:
        return tf.signal.kaiser_window(
            window_length, beta, dtype=tf.dtypes.float32, name=None
        )
    else:
        return tf.signal.kaiser_window(window_length + 1, beta, dtype=dtype, name=None)[
            :-1
        ]


def kaiser_bessel_derived_window(
    window_length: int,
    periodic: bool = True,
    beta: float = 12.0,
    *,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if periodic is True:
        return tf.signal.kaiser_bessel_derived_window(
            window_length + 1, beta, dtype, name=None
        )[:-1]
    else:
        return tf.signal.kaiser_bessel_derived_window(
            window_length, beta, dtype, name=None
        )


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
    return tf.signal.hann_window(size, periodic=(not periodic), dtype=dtype)


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
