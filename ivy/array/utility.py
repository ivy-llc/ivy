# global
from typing import Optional, Union, Sequence
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithUtility(abc.ABC):
    def all(
        self: ivy.Array,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.all(self._data, axis, keepdims, out=out)

    def any(
        self: ivy.Array,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.any(self._data, axis, keepdims, out=out)
