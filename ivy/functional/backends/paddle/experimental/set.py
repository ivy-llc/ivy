# global
from typing import Tuple
import paddle
from ivy.func_wrapper import with_unsupported_device_and_dtypes

# local
import ivy.functional.backends.paddle as paddle_backend
from . import backend_version


@with_unsupported_device_and_dtypes(
    {
        "2.5.0 and below": {
            "cpu": (
                "complex64",
                "complex128",
                "int8",
                "uint8",
                "int16",
                "float16",
                "bfloat16",
            )
        }
    },
    backend_version,
)
def intersection(
    x1: paddle.Tensor,
    x2: paddle.Tensor,
    /,
    *,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    x1 = paddle.reshape(x1, [-1])
    x2 = paddle.reshape(x2, [-1])
    if not assume_unique:
        x1, ind1, _, _ = paddle_backend.unique_all(x1)
        x2, ind2, _, _ = paddle_backend.unique_all(x2)

    aux = paddle.concat([x1, x2], 0)
    if return_indices:
        values_ = paddle.reshape(aux, (aux.shape[0], -1))
        aux_sort_indices = paddle.to_tensor(
            [i[0] for i in sorted(list(enumerate(values_)), key=lambda x: tuple(x[1]))]
        )
        aux = aux[aux_sort_indices]
    else:
        aux = paddle.sort(aux)

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask]
        if not ar2_indices.size == 0:
            ar2_indices = ar2_indices - x1.size
        if not assume_unique:
            ar1_indices = paddle.gather(ind1, ar1_indices)
            ar2_indices = paddle.gather(ind2, ar2_indices)
        return (
            int1d,
            paddle.cast(ar1_indices, paddle.int64),
            paddle.cast(ar2_indices, paddle.int64),
        )
    else:
        return int1d
