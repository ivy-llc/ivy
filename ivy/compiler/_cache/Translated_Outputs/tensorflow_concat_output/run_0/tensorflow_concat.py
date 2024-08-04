import tensorflow

from typing import Union
from typing import Tuple
from typing import List
from typing import Optional

from .tensorflow__helpers import tensorflow_concat


def tensorflow_concat(
    xs: Union[Tuple[tensorflow.Tensor, ...], List[tensorflow.Tensor]],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[tensorflow.Tensor, tensorflow.Variable]] = None,
):
    if axis is not None:
        try:
            return tensorflow.concat(xs, axis)
        except tensorflow.errors.InvalidArgumentError as error:
            if "(zero-based) was expected to be" in error.message:
                highest_dtype = xs[0].dtype
                return tensorflow.concat(
                    [
                        (
                            tensorflow.cast(x, highest_dtype)
                            if x.dtype != highest_dtype
                            else x
                        )
                        for x in xs
                    ],
                    axis,
                )
            else:
                raise
    return tensorflow_concat([tensorflow.reshape(x, -1) for x in xs], axis=0)
