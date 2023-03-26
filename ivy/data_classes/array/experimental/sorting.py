# global
import abc
from typing import Optional

# local

import ivy


class _ArrayWithSortingExperimental(abc.ABC):
    def msort(
        self: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.msort. This method simply wraps the
        function, and so the docstring for ivy.msort also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            sorted array of the same type and shape as a

        Examples
        --------
        >>> a = ivy.asarray([[8, 9, 6],[6, 2, 6]])
        >>> a.msort()
        ivy.array(
            [[6, 2, 6],
            [8, 9, 6]]
            )
        """
        return ivy.msort(self._data, out=out)

    def lexsort(
        self: ivy.Array,
        /,
        *,
        axis: int = -1,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lexsort. This method simply wraps the
        function, and so the docstring for ivy.lexsort also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis of each key to be indirectly sorted.
            By default, sort over the last axis of each key.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            array of integer indices with shape N, that sort the input array as keys.

        Examples
        --------
        >>> a = [1,5,1,4,3,4,4] # First column
        >>> b = [9,4,0,4,0,2,1] # Second column
        >>> keys = ivy.asarray([b,a])
        >>> keys.lexsort() # Sort by a, then by b
        array([2, 0, 4, 6, 5, 3, 1])
        """
        return ivy.lexsort(self._data, axis=axis, out=out)
