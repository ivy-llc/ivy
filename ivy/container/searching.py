# global
from typing import Optional

# local
import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithSearching(ContainerBase):
    @staticmethod
    def static_argmax(
        x: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
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
        return ContainerBase.multi_map_in_static_method(
            "argmax", x, axis=axis, keepdims=keepdims, out=out
        )

    def argmax(
        self: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.argmax. This method simply
        wraps the function, and so the docstring for ivy.argmax also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
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
        return self.static_argmax(self, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def static_argmin(
        x: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.argmin. This method simply
        wraps the function, and so the docstring for ivy.argmin also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
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
        return ContainerBase.multi_map_in_static_method(
            "argmin", x, axis=axis, keepdims=keepdims, out=out
        )

    def argmin(
        self: ivy.Container,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.argmin. This method simply
        wraps the function, and so the docstring for ivy.argmin also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
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
        return self.static_argmin(self, axis=axis, keepdims=keepdims, out=out)
