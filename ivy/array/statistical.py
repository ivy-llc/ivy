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
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.min(self._data, axis=axis, keepdims=keepdims, out=out)

    def max(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.max(self._data, axis, keepdims, out=out)

    def mean(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.mean(self._data, axis, keepdims, out=out)

    def var(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.var(self._data, axis, correction, keepdims, out=out)

    def prod(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.prod(self._data, axis, keepdims, dtype=dtype, out=out)

    def sum(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        keepdims: bool = False,
        *,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.sum(self._data, axis=axis, dtype=dtype, keepdims=keepdims, out=out)

    def std(
        self: ivy.Array,
        axis: Union[int, Tuple[int]] = None,
        correction: Union[int, float] = 0.0,
        keepdims: bool = False,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.std(self._data, axis, correction, keepdims, out=out)

    def einsum(
        self: ivy.Array,
        equation: str,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.einsum(equation, self._data, out=out)
