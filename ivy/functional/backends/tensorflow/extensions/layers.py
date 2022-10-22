from typing import Union, Optional, Tuple
import tensorflow as tf


def vorbis_window(
    window_length: Union[tf.Tensor, tf.Variable],
    *,
    dtype: Optional[tf.DType] = tf.dtypes.float32,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.signal.vorbis_window(window_length, dtype=dtype, name=None)


def hann_window(
    window_length: int,
    periodic: Optional[bool] = True,
    dtype: Optional[tf.DType] = None,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.signal.hann_window(
        window_length, periodic=periodic, dtype=dtype, name=None
    )


def max_pool2d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NHWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCHW":
        x = tf.transpose(x, (0, 2, 3, 1))
    res = tf.nn.max_pool2d(x, kernel, strides, padding)
    if data_format == "NCHW":
        return tf.transpose(res, (0, 3, 1, 2))
    return res


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
