# global
import abc

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

    def iinfo(self: ivy.Array) -> Iinfo:
        return ivy.iinfo(type=self._dtype)

    def finfo(self: ivy.Array) -> Finfo:
        return ivy.finfo(type=self._dtype)

    def broadcast_to(
        self: ivy.Array,
        shape: Tuple[int, ...],
        out: Optional[ivy.Array] = None
    ):
        return ivy.broadcast_to(x=self._data, shape= shape, out=out)

    def broadcast_arrays(self, *arrays: Union[ivy.Array, ivy.NativeArray]) -> List[ivy.Array]:
        return ivy.broadcast_arrays(self._data, arrays)

    def dtype(self: ivy.Array, as_native: bool = False) -> ivy.Dtype:
        return ivy.dtype(self._data, as_native)

    def astype(
        self: ivy.Array,
        dtype: ivy.Dtype,
        copy: bool = True,
        out: ivy.Array = None
    ) -> ivy.Array:
        return ivy.astype(self._data, dtype=dtype, copy=copy, out=out)



