# global
import tensorflow as tf
from typing import Union, Optional

# local
import ivy


def argsort(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    direction = "DESCENDING" if descending else "ASCENDING"
    x = tf.convert_to_tensor(x)
    is_bool = x.dtype.is_bool
    if is_bool:
        x = tf.cast(x, tf.int32)
    ret = tf.argsort(x, axis=axis, direction=direction, stable=stable)
    return tf.cast(ret, dtype=tf.int64)


def sort(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    # TODO: handle stable sort when it's supported in tensorflow
    # currently it supports only quicksort (unstable)
    direction = "DESCENDING" if descending else "ASCENDING"
    x = tf.convert_to_tensor(x)
    is_bool = x.dtype.is_bool
    if is_bool:
        x = tf.cast(x, tf.int32)
    ret = tf.sort(x, axis=axis, direction=direction)
    if is_bool:
        ret = tf.cast(ret, dtype=tf.bool)
    return ret


def searchsorted(
    x: Union[tf.Tensor, tf.Variable],
    v: Union[tf.Tensor, tf.Variable],
    /,
    *,
    side="left",
    sorter=None,
    ret_dtype=tf.int64,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    assert ivy.is_int_dtype(ret_dtype), ValueError(
        "only Integer data types are supported for ret_dtype."
    )
    is_supported_int_ret_dtype = ret_dtype in [tf.int32, tf.int64]
    if sorter is not None:
        x = tf.gather(x, sorter)
    if x.ndim == 1 and v.ndim != 1:
        if is_supported_int_ret_dtype:
            fn = lambda inner_v: tf.searchsorted(
                x, inner_v, side=side, out_type=ret_dtype
            )
        else:
            fn = lambda inner_v: tf.cast(
                tf.searchsorted(x, inner_v, side=side), ret_dtype
            )
        return tf.map_fn(fn=fn, elems=v, dtype=ret_dtype)
    if is_supported_int_ret_dtype:
        return tf.searchsorted(x, v, side=side, out_type=ret_dtype)
    return tf.cast(tf.searchsorted(x, v, side=side), ret_dtype)
