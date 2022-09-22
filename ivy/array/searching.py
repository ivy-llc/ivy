# global
import abc
from typing import Optional, Union, Tuple

# local
import ivy


class ArrayWithSearching(abc.ABC):
    def argmax(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> Union[ivy.Array, int]:
        """
        ivy.Array instance method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the maximum value of the flattened array. Default  None.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the array.
        out
            If provided, the result will be inserted into this array. It should be of
            the appropriate shape and dtype.

        Returns
        -------
        ret
            if axis is None, a zero-dimensional array containing the index of the first
            occurrence of the maximum value; otherwise, a non-zero-dimensional array
            containing the indices of the maximum values. The returned array must have
            the default array index data type.

        """
        return ivy.argmax(self._data, axis=axis, keepdims=keepdims, out=out)

    def argmin(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> Union[ivy.Array, int]:
        """
        ivy.Array instance method variant of ivy.argmin. This method simply
        wraps the function, and so the docstring for ivy.argmin also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        axis
            axis along which to search. If None, the function must return the index of
            the minimum value of the flattened array. Default = None.
        keepdims
            if True, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see Broadcasting). Otherwise, if False, the reduced axes
            (dimensions) must not be included in the result. Default = False.
        out
            if axis is None, a zero-dimensional array containing the index of the first
            occurrence of the minimum value; otherwise, a non-zero-dimensional array
            containing the indices of the minimum values. The returned array must have
            the default array index data type.

        Returns
        -------
        ret
            Array containing the indices of the minimum values across the specified
            axis.

        """
        return ivy.argmin(self._data, axis=axis, keepdims=keepdims, out=out)

    def nonzero(
        self: ivy.Array,
        /,
        *,
        as_tuple=True,
        size: int = None,
        fill_value: int = 0,
    ) -> Union[Tuple[ivy.Array], ivy.Array]:
        """
        ivy.Array instance method variant of ivy.nonzero. This method simply
        wraps the function, and so the docstring for ivy.nonzero also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array. Should have a numeric data type.
        as_tuple
            if True, the output is returned as a tuple of indices, one for each
            dimension of the input, containing the indices of the true elements in that
            dimension. If False, the coordinates are returned in a (N, ndim) array,
            where N is the number of true elements. Default = True.
        size
            if specified, the function will return an array of shape (size, ndim).
            If the number of non-zero elements is fewer than size, the remaining
            elements will be filled with fill_value. Default = None.
        fill_value
            when size is specified and there are fewer than size number of elements,
            the remaining elements in the output array will be filled with fill_value.
            Default = 0.

        Returns
        -------
        ret
            Array containing the indices of the non-zero values.

        """
        return ivy.nonzero(
            self._data, as_tuple=as_tuple, size=size, fill_value=fill_value
        )

    def where(
        self: ivy.Array,
        x1: ivy.Array,
        x2: ivy.Array,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.where. This method simply
        wraps the function, and so the docstring for ivy.where also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            Where True, yield x1, otherwise yield x2.
        x1
            input array. Should have a numeric data type.
        x2
            values from which to choose when condition is False.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            An array with elements from self where condition is True, and elements from
            x2 otherwise.

        """
        return ivy.where(self._data, x1._data, x2._data, out=out)

    def argwhere(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.argwhere. This method simply
        wraps the function, and so the docstring for ivy.argwhere also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array for which indices are desired
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            Indices for where the boolean array is True.

        """
        return ivy.argwhere(self._data, out=out)
