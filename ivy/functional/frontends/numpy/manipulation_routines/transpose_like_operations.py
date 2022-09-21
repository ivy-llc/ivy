import ivy

from types import NoneType
from typing import Optional, Union, Tuple, List


def transpose(array, /, *, axes: Optional[Union[NoneType, Tuple[int], List[int]]]=None) -> ivy.Array:
    if axes is None:
        axes = list(range(len(array.shape)))[::-1]
    try:
        assert len(axes) > 1
    except AssertionError:
        raise ValueError("`axes` should have the same size the input array.ndim")

    return ivy.permute_dims(array, axes, out=None)

