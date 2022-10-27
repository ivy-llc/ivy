# global
import abc
from typing import Optional

# local
import ivy


class ArrayWithLinalgExtensions(abc.ABC):
    def diagflat(
        self: ivy.Array,
        /,
        *,
        k: int = 0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.diagflat.
        This method simply wraps the function, and so the docstring for
        ivy.diagflat also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1,2])
        >>> x.diagflat(k=1)
        ivy.array([[0, 1, 0],
                   [0, 0, 2],
                   [0, 0, 0]])
        """
        return ivy.diagflat(self._data, k=k, out=out)
