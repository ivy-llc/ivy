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
    k: Optional[int] = 0,
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
