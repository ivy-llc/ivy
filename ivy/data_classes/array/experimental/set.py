# global
import abc
from typing import Optional

# local
import ivy


class _ArrayWithSetExperimental(abc.ABC):
    def difference(
        self: ivy.Array,
        x2: ivy.Array = None,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array method variant of ivy.difference. This method simply wraps the function, and so the
        docstring for ivy.difference also applies to this method with minimal changes.
        Parameters
        ----------
        self
            a 1D or 2D input array, nativearray, or container, with a numeric data type.
            
        x2
            optional second 1D or 2D input array, nativearray, or container, with a numeric data type.
            Must have the same shape as ``self``.
        out
            optional output array, nativearray, or container, with a numeric data type. Must have the
            same shape as ``self``.
        Returns
        -------
        ret
            a container containing the set difference between two containers.
        Examples
        --------
        >>> x = ivy.array([1., 2., 3., 5.])
        >>> y = ivy.array([3., 45., 3., 4.])
        >>> z = x.difference(y)
        >>> print(z)
        ivy.container([1., 2.])
        """
        return ivy.difference(self._data, x2)
