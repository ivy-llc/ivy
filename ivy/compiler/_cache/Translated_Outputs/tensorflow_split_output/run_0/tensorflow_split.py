import tensorflow

import math
from typing import Sequence
from typing import Optional
from typing import Union

from .tensorflow__helpers import tensorflow_handle_array_like_without_promotion


@tensorflow_handle_array_like_without_promotion
def tensorflow_split(
    x: Union[tensorflow.Tensor, tensorflow.Variable],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[
        Union[int, Sequence[int], Union[tensorflow.Tensor, tensorflow.Variable]]
    ] = None,
    axis: int = 0,
    with_remainder: bool = False,
):
    if x.shape == ():
        if num_or_size_splits is not None and num_or_size_splits != 1:
            raise Exception(
                f"input array had no shape, but num_sections specified was {num_or_size_splits}"
            )
        return [x]
    if num_or_size_splits is None:
        dim_size = tensorflow.shape(x)[axis]
        num_or_size_splits = int(dim_size)
    if isinstance(num_or_size_splits, (tensorflow.Tensor, tensorflow.Variable)):
        num_or_size_splits = tensorflow.cast(num_or_size_splits, tensorflow.int32)
    elif isinstance(num_or_size_splits, int) and with_remainder:
        num_chunks = x.shape[axis] / num_or_size_splits
        num_chunks_int = math.floor(num_chunks)
        remainder = num_chunks - num_chunks_int
        if remainder != 0:
            num_or_size_splits = [num_or_size_splits] * num_chunks_int + [
                int(remainder * num_or_size_splits)
            ]
    return tensorflow.split(x, num_or_size_splits, axis)
