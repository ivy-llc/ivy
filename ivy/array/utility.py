# global
from typing import Optional, Union, Tuple, List
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithUtility(abc.ABC):
    def all(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.all(self, axis, keepdims, out=out)

    def any(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.any(self, axis, keepdims, out=out)
