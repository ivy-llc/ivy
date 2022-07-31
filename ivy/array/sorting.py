# global
import abc
from typing import Optional

# local

import ivy


class ArrayWithSorting(abc.ABC):
    def argsort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.argsort(self._data, axis, descending, stable, out=out)
        """
        ivy.Array instance method variant of ivy.argsort. This method simply wraps the
        function, and so the docstring for ivy.argsort also applies to this method
        with minimal changes.
        
        Examples
        --------
        >>> x = ivy.array([1, 5, 2])
        >>> y = x.argsort(-1, True, False)
        >>> print(y)
        ivy.array([1, 2, 0])
        
        >>> x = ivy.array([9.6, 2.7, 5.2])
        >>> y = x.argsort(-1, True, False)
        >>> print(y)
        ivy.array([0, 2, 1])
        
        """
    def sort(
        self: ivy.Array,
        axis: int = -1,
        descending: bool = False,
        stable: bool = True,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sort. This method simply wraps the
        function, and so the docstring for ivy.sort also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([7, 8, 6])
        >>> y = x.sort(-1, True, False)
        >>> print(y)
        ivy.array([8, 7, 6])

        >>> x = ivy.array([8.5, 8.2, 7.6])
        >>> y = x.sort(-1, True, False)
        >>> print(y)
        ivy.array([8.5, 8.2, 7.6])

        """
        return ivy.sort(self._data, axis, descending, stable, out=out)
