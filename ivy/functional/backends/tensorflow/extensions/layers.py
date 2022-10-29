from typing import Union, Optional, Tuple, Literal
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


def max_pool1d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int]],
    strides: Union[int, Tuple[int]],
    padding: str,
    /,
    *,
    data_format: str = "NWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:

    if data_format == "NCW":
        x = tf.transpose(x, (0, 2, 1))
    res = tf.nn.max_pool1d(x, kernel, strides, padding)

    if data_format == "NCW":
        res = tf.transpose(res, (0, 2, 1))
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


def dct(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    type: Optional[Literal[1, 2, 3, 4]] = 2,
    n: Optional[int] = None,
    axis: Optional[int] = -1,
    norm: Optional[Literal["ortho"]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> tf.Tensor:
    if x.dtype not in (tf.float32, tf.float64):
        x = tf.cast(x, tf.float32)
    if axis != -1:
        new_dims = list(range(len(x.shape)))
        new_dims[axis], new_dims[-1] = new_dims[-1], axis
        x = tf.transpose(x, new_dims)
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
        dct_out = tf.transpose(dct_out, new_dims)
    else:
        dct_out = tf.signal.dct(x, type=type, n=n, axis=-1, norm=norm)
    return dct_out
