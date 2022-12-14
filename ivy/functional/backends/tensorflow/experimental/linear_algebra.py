from typing import Union, Optional, Tuple
import tensorflow as tf

import ivy


def diagflat(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    offset: Optional[int] = 0,
    padding_value: Optional[float] = 0,
    align: Optional[str] = "RIGHT_LEFT",
    num_rows: Optional[int] = None,
    num_cols: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
):
    if len(x.shape) > 1:
        x = tf.reshape(x, [-1])

    if num_rows is None:
        num_rows = -1
    if num_cols is None:
        num_cols = -1

    ret = tf.linalg.diag(
        x,
        name="diagflat",
        k=offset,
        num_rows=num_rows,
        num_cols=num_cols,
        padding_value=padding_value,
        align=align,
    )

    if ivy.exists(out):
        ivy.inplace_update(out, ret)

    return ret


def kron(
    a: Union[tf.Tensor, tf.Variable],
    b: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.kron(a, b)


def eig(
    x: Union[tf.Tensor, tf.Variable],
    /,
) -> Tuple[Union[tf.Tensor, tf.Variable], ...]:
    if not ivy.dtype(x) in (ivy.float32, ivy.float64, ivy.complex64, ivy.complex128):
        return tf.linalg.eig(tf.cast(x, tf.float64))
    return tf.linalg.eig(x)
