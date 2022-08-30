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

        Parameters
        ----------
        self
            input array. If ``x`` has more than one dimension, the function must flatten
            ``x`` and return the unique elements of the flattened array.

        Returns
        -------
        ret
            a namedtuple ``(values, counts)`` whose

            - first element must have the field name ``values`` and must be an
            array containing the unique elements of ``x``.
            The array must have the same data type as ``x``.
            - second element must have the field name ``counts`` and must be an array
            containing the number of times each unique element occurs in ``x``.
            The returned array must have same shape as ``values`` and must
            have the default array index data type.

        Examples
        --------
        >>> x = ivy.array([0., 1., 2. , 1. , 0.])
        >>> y = x.unique_counts()
        >>> print(y)
        uc(values=ivy.array([0.,1.,2.]),counts=ivy.array([2,2,1]))
        """
        return ivy.unique_counts(self._data)

    def unique_values(
        self: ivy.Array, /, *, out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        return ivy.unique_values(self._data, out=out)

    def unique_all(
        self: ivy.Array,
    ) -> NamedTuple:
        return ivy.unique_all(self._data)

    def unique_inverse(self: ivy.Array) -> NamedTuple:
        return ivy.unique_inverse(self._data)
