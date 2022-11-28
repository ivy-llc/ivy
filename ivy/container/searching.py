# global
from numbers import Number
from typing import Optional, Union, List, Dict

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
        keepdims: bool = False,
        output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
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
            the maximum value of the flattened array. Deafult: ``None``.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the array.
        output_dtype
             Optional data type of the output array.
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
            "argmax",
            x,
            axis=axis,
            keepdims=keepdims,
            output_dtype=output_dtype,
            out=out,
        )

    def argmax(
        self: ivy.Container,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
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
            the maximum value of the flattened array. Deafult: ``None``.
        keepdims
            If this is set to True, the axes which are reduced are left in the result as
            dimensions with size one. With this option, the result will broadcast
            correctly against the array.
        output_dtype
            Optional output dtype of the container.
        out
            If provided, the result will be inserted into this array. It should be of
            the appropriate shape and dtype.

        Returns
        -------
        ret
            a container containing the indices of the maximum values across the
            specified axis.

        """
        return self.static_argmax(
            self, axis=axis, keepdims=keepdims, output_dtype=output_dtype, out=out
        )

    @staticmethod
    def static_argmin(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        output_dtype: Optional[Union[ivy.int32, ivy.int64]] = None,
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
        output_dtype
            An optional output_dtype from: int32, int64. Defaults to int64.
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
            "argmin",
            x,
            axis=axis,
            keepdims=keepdims,
            output_dtype=output_dtype,
            out=out,
        )

    def argmin(
        self: ivy.Container,
        /,
        *,
        axis: Optional[int] = None,
        keepdims: bool = False,
        output_dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
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
        output_dtype
            An optional output_dtype from: int32, int64. Defaults to int64.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            a container containing the indices of the minimum values across the
            specified axis.

        """
        return self.static_argmin(
            self, axis=axis, keepdims=keepdims, output_dtype=output_dtype, out=out
        )

    @staticmethod
    def static_nonzero(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        as_tuple: bool = True,
        size: Optional[int] = None,
        fill_value: Number = 0,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.nonzero. This method simply
        wraps the function, and so the docstring for ivy.nonzero also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input array or container. Should have a numeric data type.
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
            a container containing the indices of the nonzero values.

        """
        return ContainerBase.multi_map_in_static_method(
            "nonzero", x, as_tuple=as_tuple, size=size, fill_value=fill_value
        )

    def nonzero(
        self: ivy.Container,
        /,
        *,
        as_tuple: bool = True,
        size: Optional[int] = None,
        fill_value: Number = 0,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.nonzero. This method simply
        wraps the function, and so the docstring for ivy.nonzero also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array or container. Should have a numeric data type.
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
            a container containing the indices of the nonzero values.

        """
        return self.static_nonzero(
            self, as_tuple=as_tuple, size=size, fill_value=fill_value
        )

    @staticmethod
    def static_where(
        condition: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
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
        /,
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

    # Extra #
    # ----- #

    @staticmethod
    def static_argwhere(
        x: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.argwhere. This method
        simply wraps the function, and so the docstring for ivy.argwhere
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Boolean array, for which indices are desired.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Indices for where the boolean array is True.
        """
        return ContainerBase.multi_map_in_static_method(
            "argwhere",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def argwhere(
        self: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ):
        """
        ivy.Container instance method variant of ivy.argwhere. This method
        simply wraps the function, and so the docstring for ivy.argwhere
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Boolean array, for which indices are desired.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Indices for where the boolean array is True.

        Examples
        --------
        Using :class:`ivy.Container` instance method

        >>> x = ivy.Container(a=ivy.array([1, 2]), b=ivy.array([3, 4]))
        >>> res = x.argwhere()
        >>> print(res)
        {
            a: ivy.array([[0], [1]]),
            b: ivy.array([[0], [1]])
        }

        >>> x = ivy.Container(a=ivy.array([1, 0]), b=ivy.array([3, 4]))
        >>> res = x.argwhere()
        >>> print(res)
        {
            a: ivy.array([[0]]),
            b: ivy.array([[0], [1]])
        }

        """
        return self.static_argwhere(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
