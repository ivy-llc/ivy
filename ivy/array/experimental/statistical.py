# global
import abc
from typing import Optional, Union, Tuple

# local
import ivy


class ArrayWithStatisticalExperimental(abc.ABC):
    def median(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[Tuple[int], int]] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.median. This method simply
        wraps the function, and so the docstring for ivy.median also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which the medians are computed. The default is to compute
            the median along a flattened version of the array.
        keepdims
            If this is set to True, the axes which are reduced are left in the result
            as dimensions with size one.
        out
            optional output array, for writing the result to.

        Returns
        -------
        ret
            The median of the array elements.

        Examples
        --------
        >>> a = ivy.array([[10, 7, 4], [3, 2, 1]])
        >>> a.median()
        3.5
        >>> a.median(axis=0)
        ivy.array([6.5, 4.5, 2.5])
        """
        return ivy.median(self._data, axis=axis, keepdims=keepdims, out=out)
