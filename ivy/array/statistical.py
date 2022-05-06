# global
from typing import Optional, Union, Tuple
import abc

# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithStatistical(abc.ABC):
    def min(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.min(self, axis, keepdims, out=out)

    def max(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.max(self, axis, keepdims, out=out)

    def mean(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.mean(self, axis, keepdims, out=out)
