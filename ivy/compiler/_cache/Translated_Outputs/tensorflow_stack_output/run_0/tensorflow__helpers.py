import tensorflow

from typing import Tuple
from typing import Union
from typing import List
from typing import Optional


def tensorflow_stack_1(
    arrays: Union[Tuple[tensorflow.Tensor], List[tensorflow.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    try:
        return tensorflow.experimental.numpy.stack(arrays, axis)
    except ValueError as e:
        raise Exception(e) from e
