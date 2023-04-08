# global
import paddle
from typing import Optional, Union


# msort
def msort(
    a: Union[paddle.Tensor, list, tuple], /, *, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    paddle.sort(a, axis=0)


# lexsort
def lexsort(
    keys: paddle.Tensor, /, *, axis: int = -1, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    shape = keys.shape
    if len(shape) == 1:
        return paddle.argsort(keys, axis=axis)

    if shape[0] == 0:
        raise TypeError("need sequence of keys with len > 0 in lexsort")

    if len(shape) == 2 and shape[1] == 1:
        return paddle.to_tensor([0], dtype=paddle.int64)

    result = paddle.argsort(keys[0], axis=axis)
    if shape[0] == 1:
        return result

    for i in range(1, shape[0]):
        key = keys[i]
        ind = paddle.take_along_axis(key, result, axis=axis)
        temp = paddle.argsort(ind, axis=axis)
        result = paddle.take_along_axis(result, temp, axis=axis)

    return result
