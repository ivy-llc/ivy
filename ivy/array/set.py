# global
import abc
from typing import Optional, NamedTuple


import ivy


class ArrayWithSet(abc.ABC):
    def unique_counts(self: ivy.Array) -> NamedTuple:
        """
        ivy.Array instance method variant of ivy.unique_counts. This method simply
        wraps the function, and so the docstring for ivy.unique_counts also applies
        to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2. , 1. , 0.])
        >>> y = x.unique_counts()
        >>> print(y)
        uc(values=ivy.array([0.,1.,2.]),counts=ivy.array([2,2,1]))
        """
        return ivy.unique_counts(self._data)

    def unique_values(self: ivy.Array, out: Optional[ivy.Array] = None) -> ivy.Array:
        return ivy.unique_values(self._data, out=out)

    def unique_all(
        self: ivy.Array,
    ) -> NamedTuple:
        return ivy.unique_all(self._data)

    def unique_inverse(self: ivy.Array) -> NamedTuple:
        return ivy.unique_inverse(self._data)
