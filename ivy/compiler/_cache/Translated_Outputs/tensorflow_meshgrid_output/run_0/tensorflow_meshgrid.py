import tensorflow

from typing import Optional
from typing import Union


def tensorflow_meshgrid(
    *arrays: Union[tensorflow.Tensor, tensorflow.Variable],
    sparse: bool = False,
    indexing: str = "xy",
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if not sparse:
        return tensorflow.meshgrid(*arrays, indexing=indexing)
    sd = (1,) * len(arrays)
    ag__result_list_0 = []
    for i, a in enumerate(arrays):
        res = tensorflow.reshape(
            tensorflow.convert_to_tensor(a), sd[:i] + (-1,) + sd[i + 1 :]
        )
        ag__result_list_0.append(res)
    res = ag__result_list_0
    if indexing == "xy" and len(arrays) > 1:
        res[0] = tensorflow.reshape(res[0], (1, -1) + sd[2:])
        res[1] = tensorflow.reshape(res[1], (-1, 1) + sd[2:])
    return res
