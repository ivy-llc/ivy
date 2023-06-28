import ivy
import tensorflow as tf
from typing import Union

from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@with_unsupported_dtypes({"2.12.0 and below": ("complex", "float16")}, backend_version)
def intersection(
        a: Union[tf.Tensor, tf.Variable],
        b: Union[tf.Tensor, tf.Variable],
        /,
        *,
        assume_unique: bool = False,
        return_indices: bool = False,
) -> Union[tf.Tensor, tf.Variable]:
    a = tf.reshape(a, [-1])
    b = tf.reshape(b, [-1])
    if not assume_unique:
        ivy_tf = ivy.current_backend()
        if return_indices:
            a, ind1, _, _ = ivy_tf.unique_all(a)
            b, ind2, _, _ = ivy_tf.unique_all(b)
        else:
            a, _, _, _ = ivy_tf.unique_all(a)
            b, _, _, _ = ivy_tf.unique_all(b)

    aux = tf.concat([a, b], 0)
    if return_indices:
        aux_sort_indices = tf.argsort(aux)
        aux = tf.gather(aux, aux_sort_indices)
    else:
        aux = tf.sort(aux)

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]
    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - a.shape
        if not assume_unique:
            ar1_indices = tf.gather(ind1, ar1_indices)
            ar2_indices = tf.gather(ind2, ar2_indices)
        return int1d, tf.cast(ar1_indices, tf.int64), tf.cast(ar2_indices, tf.int64)
    else:
        return int1d
