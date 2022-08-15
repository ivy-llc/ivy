import ivy
from typing import Sequence, Any, Union


def reshape(a: Sequence[Any], newshape: Union[Sequence[int], int], order='C') -> Sequence[Any]:
    return ivy.reshape(a, newshape)
