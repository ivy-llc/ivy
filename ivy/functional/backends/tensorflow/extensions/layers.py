from typing import Union, Optional, Tuple
import tensorflow as tf
from ivy.func_wrapper import with_unsupported_dtypes
from .. import backend_version


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


@with_unsupported_dtypes({"2.9.1 and below":
                         ("bfloat16", "float64")},
                         backend_version
                         )
def max_pool3d(
    x: Union[tf.Tensor, tf.Variable],
    kernel: Union[int, Tuple[int], Tuple[int, int, int]],
    strides: Union[int, Tuple[int], Tuple[int, int, int]],
    padding: str,
    /,
    *,
    data_format: str = "NDHWC",
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if data_format == "NCDHW":
        x = tf.transpose(x, (0, 2, 3, 4, 1))
    res = tf.nn.max_pool3d(x, kernel, strides, padding)
    if data_format == "NCDHW":
        return tf.transpose(res, (0, 4, 1, 2, 3))
    return res
