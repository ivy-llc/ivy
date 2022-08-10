# global
from typing import Optional, Union

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithSearching(ContainerBase):
    @staticmethod
    def static_argmax(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
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
            a container containing the indices of the maximum values across the
            specified axis.

        """
        return ContainerBase.multi_map_in_static_method(
            "argmax", x, axis=axis, keepdims=keepdims, out=out
        )

    def argmax(
        self: ivy.Container,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
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
            a container containing the indices of the maximum values across the
            specified axis.

        """
        return self.static_argmax(self, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def static_argmin(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the indices of the minimum values across the
            specified axis.
        """
        return ContainerBase.multi_map_in_static_method(
            "argmin", x, axis=axis, keepdims=keepdims, out=out
        )

    def argmin(
        self: ivy.Container,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: Optional[bool] = False,
        out: Optional[ivy.Container] = None,
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the indices of the minimum values across the
            specified axis.

        """
        return self.static_argmin(self, axis=axis, keepdims=keepdims, out=out)

    @staticmethod
    def static_nonzero(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray]
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nonzero. This method simply
        wraps the function, and so the docstring for ivy.nonzero also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.

        Returns
        -------
        ret
            a container containing the indices of the nonzero values.

        """
        return ContainerBase.multi_map_in_static_method("nonzero", x)

    def nonzero(self: ivy.Container) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nonzero. This method simply
        wraps the function, and so the docstring for ivy.nonzero also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.

        Returns
        -------
        ret
            a container containing the indices of the nonzero values.

        """
        return self.static_nonzero(self)

    @staticmethod
    def static_where(
        condition: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.where. This method simply
        wraps the function, and so the docstring for ivy.where also applies
        to this method with minimal changes.

        Parameters
        ----------
        condition
            input array or container. Should have a boolean data type.
        x1
            input array or container. Should have a numeric data type.
        x2
            input array or container. Should have a numeric data type.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the values of x1 where condition is True, and x2
            where condition is False.

        """
        return ContainerBase.multi_map_in_static_method(
            "where", condition, x1, x2, out=out
        )

    def where(
        self: ivy.Container,
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.where. This method simply
        wraps the function, and so the docstring for ivy.where also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a boolean data type.
        x1
            input array or container. Should have a numeric data type.
        x2
            input array or container. Should have a numeric data type.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the values of x1 where condition is True, and x2
            where condition is False.

        """
        return self.static_where(self, x1, x2, out=out)
