from typing import Union, Optional
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

def matrix_exp(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    max_squarings: Optional[int] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if max_squarings is not None:
        c = 0
        if x.dtype == tf.float32:
            c = 1.97
        elif x.dtype == tf.float64:
            c = 2.42
        else:
            return tf.linalg.expm(x)

        if (
            max(
                0,
                tf.math.ceil(
                    tf.experimental.numpy.log2(tf.norm(x, ord="euclidean")) - c
                ),
            )
            > max_squarings
        ):
            return tf.constant(float("NaN"), dtype=tf.float32)
        else:
            return tf.linalg.expm(x)
    return tf.linalg.expm(x)