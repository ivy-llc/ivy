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

    def var(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.var(self, axis, keepdims, out=out)

    def prod(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        dtype: Optional[Union[ivy.Dtype, str]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.prod(self, axis, dtype, keepdims, out=out)

    def sum(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        dtype: Optional[Union[ivy.Dtype, str]] = None,
        keepdims: bool = False,
        out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
    ) -> ivy.Array:
        return ivy.sum(self, axis, dtype, keepdims, out=out)
