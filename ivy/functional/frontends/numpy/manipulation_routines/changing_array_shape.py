import ivy
from typing import Sequence, Any, Union


def reshape(a: Sequence[Any], newshape: Union[Sequence[int], int]) -> Sequence[Any]:
    return ivy.reshape(a, newshape)
