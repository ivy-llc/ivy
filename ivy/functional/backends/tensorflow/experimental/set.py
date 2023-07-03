# global
import tensorflow as tf
from typing import Union, Tuple

# local
import ivy.functional.backends.tensorflow as tf_backend
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.12.0 and below": ("complex", "float16")}, backend_version)
def intersection(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    x1 = tf.reshape(x1, [-1])
    x2 = tf.reshape(x2, [-1])
    if not assume_unique:
        if return_indices:
            x1, ind1, _, _ = tf_backend.unique_all(x1)
            x2, ind2, _, _ = tf_backend.unique_all(x2)
        else:
            x1, _, _, _ = tf_backend.unique_all(x1)
            x2, _, _, _ = tf_backend.unique_all(x2)

    aux = tf.concat([x1, x2], 0)
    if return_indices:
        aux_sort_indices = tf.argsort(aux)
        aux = tf.gather(aux, aux_sort_indices)
    else:
        aux = tf.sort(aux)

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]
    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - x1.shape
        if not assume_unique:
            ar1_indices = tf.gather(ind1, ar1_indices)
            ar2_indices = tf.gather(ind2, ar2_indices)
        return int1d, tf.cast(ar1_indices, tf.int64), tf.cast(ar2_indices, tf.int64)
    else:
        return int1d
