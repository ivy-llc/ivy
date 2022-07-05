# global
import abc
from typing import Tuple, Optional
# local
import ivy

# ToDo: implement all methods here as public instance methods


class ArrayWithDataTypes(abc.ABC):
    def can_cast(self: ivy.Array, to: ivy.Dtype) -> bool:
        """
        `ivy.Array` instance method variant of `ivy.can_cast`. This method simply wraps
        the function, and so the docstring for `ivy.can_cast` also applies to this
        method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1., 2., 3.])
        >>> print(x.dtype)
        float32

        >>> print(x.can_cast(ivy.float64))
        True
        """
        return ivy.can_cast(from_=self._data, to=to)

    def broadcast_to(
        self: ivy.Array,
        shape: Tuple[int, ...],
        out: Optional[ivy.Array] = None
    ):
        return ivy.broadcast_to(x=self._data, shape= shape, out=out)

    def dtype(self: ivy.Array, as_native: Optional[bool] = False) -> ivy.Dtype:
        return ivy.dtype(self._data, as_native)

    def astype(
        self: ivy.Array,
        dtype: ivy.Dtype,
        copy: bool = True,
        out: ivy.Array = None
    ) -> ivy.Array:
        return ivy.astype(self._data, dtype=dtype, copy=copy, out=out)

    def dtype_bits(self: ivy.Array) -> int:
        return ivy.dtype_bits(self._dtype)

    def as_ivy_dtype(self: ivy.Array) -> ivy.Dtype:
        return ivy.as_ivy_dtype(self._dtype)

    def as_native_dtype(self: ivy.Array) -> ivy.NativeDtype:
        return ivy.as_native_dtype(self._dtype)

    def is_int_dtype(self: ivy.Array) -> bool:
        return ivy.is_int_dtype(self._data)

    def is_float_dtype(self: ivy.Array) -> bool:
        return ivy.is_float_dtype(self._data)





