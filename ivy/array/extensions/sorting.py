# global
import abc
from typing import Optional

# local

import ivy


class ArrayWithSortingExtensions(abc.ABC):
    # msort
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
        >>> a = ivy.randint(10, size=(2,3))
        >>> a.msort()
        ivy.array(
            [[6, 2, 6],
            [8, 9, 6]]
            )
        """
        return ivy.msort(self._data, out=out)
