# global
from typing import Optional, Union, Sequence
import abc

# local
import ivy


class _ArrayWithUtility(abc.ABC):
    def all(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.all. This method simply
        wraps the function, and so the docstring for ivy.all also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis or axes along which to perform a logical AND reduction. By default, a
            logical AND reduction must be performed over the entire array. If a tuple of
            integers, logical AND reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default  ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            if a logical AND reduction was performed over the entire array, the returned
            array must be a zero-dimensional array containing the test result;
            otherwise, the returned array must be a non-zero-dimensional array
            containing the test results. The returned array must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.array([0, 1, 2])
        >>> y = x.all()
        >>> print(y)
        ivy.array(False)

        >>> x = ivy.array([[[0, 1], [0, 0]], [[1, 2], [3, 4]]])
        >>> y = x.all(axis=1)
        >>> print(y)
        ivy.array([[False, False],
               [ True,  True]])
        """
        return ivy.all(self._data, axis=axis, keepdims=keepdims, out=out)

    def any(
        self: ivy.Array,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        keepdims: bool = False,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.any. This method simply
        wraps the function, and so the docstring for ivy.any also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array.
        axis
            axis or axes along which to perform a logical OR reduction. By default, a
            logical OR reduction must be performed over the entire array. If a tuple of
            integers, logical OR reductions must be performed over multiple axes. A
            valid ``axis`` must be an integer on the interval ``[-N, N)``, where ``N``
            is the rank(number of dimensions) of ``self``. If an ``axis`` is specified
            as a negative integer, the function must determine the axis along which to
            perform a reduction by counting backward from the last dimension (where
            ``-1`` refers to the last dimension). If provided an invalid ``axis``, the
            function must raise an exception. Default: ``None``.
        keepdims
            If ``True``, the reduced axes (dimensions) must be included in the result as
            singleton dimensions, and, accordingly, the result must be compatible with
            the input array (see :ref:`broadcasting`). Otherwise, if ``False``, the
            reduced axes(dimensions) must not be included in the result.
            Default: ``False``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            if a logical OR reduction was performed over the entire array, the returned
            array must be a zero-dimensional array containing the test result;
            otherwise, the returned array must be a non-zero-dimensional array
            containing the test results. The returned array must have a data type of
            ``bool``.

        Examples
        --------
        >>> x = ivy.array([0, 1, 2])
        >>> y = x.any()
        >>> print(y)
        ivy.array(True)

        >>> x = ivy.array([[[0, 1], [0, 0]], [[1, 2], [3, 4]]])
        >>> y = x.any(axis=2)
        >>> print(y)
        ivy.array([[ True, False],
               [ True,  True]])
        """
        return ivy.any(self._data, axis=axis, keepdims=keepdims, out=out)
