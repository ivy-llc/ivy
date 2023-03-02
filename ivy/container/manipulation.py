# For Review
# global
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Dict,
    Iterable,
    Sequence,
)
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class _ContainerWithManipulation(ContainerBase):
    @staticmethod
    def static_concat(
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container], ...],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        /,
        *,
        axis: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.concat. This method simply
        wraps the function, and so the docstring for ivy.concat also applies to
        this method with minimal changes.
        """
        return ContainerBase.cont_multi_map_in_function(
            "concat",
            xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def concat(
        self: ivy.Container,
        /,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container], ...],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        *,
        axis: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.concat. This method simply wraps
        the function, and so the docstring for ivy.concat also applies to this method
        with minimal changes.
        """
        new_xs = xs.cont_copy() if ivy.is_ivy_container(xs) else xs.copy()
        new_xs.insert(0, self.cont_copy())
        return self.static_concat(
            new_xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_expand_dims(
        x: ivy.Container,
        /,
        *,
        axis: Union[int, Sequence[int]] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.expand_dims. This method simply
        wraps the function, and so the docstring for ivy.expand_dims also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            position where a new axis (dimension) of size one will be added. If an
            element of the container has the rank of ``N``, then the ``axis`` needs
            to be between ``[-N-1, N]``. Default: ``0``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``x``, but with the dimensions of
            its elements added by one in a given ``axis``.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1.]),
        ...                   b=ivy.array([3., 4.]),
        ...                   c=ivy.array([6., 7.]))
        >>> y = ivy.Container.static_expand_dims(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.],
                          [1.]]),
            b: ivy.array([[3.],
                          [4.]]),
            c: ivy.array([[6.],
                          [7.]])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]),
        ...                   c=ivy.array([6., 7., 8.]))
        >>> container_axis = ivy.Container(a=0, b=-1, c=(0,1))
        >>> y = ivy.Container.static_expand_dims(x, axis=container_axis)
        >>> print(y)
        {
            a: ivy.array([[0., 1., 2.]]),
            b: ivy.array([[3.],
                          [4.],
                          [5.]]),
            c: ivy.array([[[6., 7., 8.]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "expand_dims",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def expand_dims(
        self: ivy.Container,
        /,
        *,
        axis: Union[int, Sequence[int]] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.expand_dims. This method simply
        wraps the function, and so the docstring for ivy.expand_dims also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            position where a new axis (dimension) of size one will be added. If an
            element of the container has the rank of ``N``, the ``axis`` needs to
            be between ``[-N-1, N]``. Default: ``0``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self``, but with the dimensions of
            its elements added by one in a given ``axis``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0., 1.],
        ...                                [2., 3.]]),
        ...                   b=ivy.array([[4., 5.],
        ...                                [6., 7.]]))
        >>> y = x.expand_dims(axis=1)
        >>> print(y)
        {
            a: ivy.array([[[0., 1.]],
                          [[2., 3.]]]),
            b: ivy.array([[[4., 5.]],
                          [[6., 7.]]])
        }
        """
        return self.static_expand_dims(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_split(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        num_or_size_splits: Optional[Union[int, Sequence[int]]] = None,
        axis: int = 0,
        with_remainder: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> List[ivy.Container]:
        """
        ivy.Container static method variant of ivy.split. This method simply
        wraps the function, and so the docstring for ivy.split also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            array to be divided into sub-arrays.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers. Default
            is to divide into as many 1-dimensional arrays as the axis dimension.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.
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
            list of containers of sub-arrays.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2, 1, 5, 9]), b=ivy.array([3, 7, 2, 11]))
        >>> y = ivy.Container.static_split(x, num_or_size_splits=2)
        >>> print(y)
        [{
            a: ivy.array([2, 1]),
            b: ivy.array([3, 7])
        }, {
            a: ivy.array([5, 9]),
            b: ivy.array([2, 11])
        }]

        """
        return ContainerBase.cont_multi_map_in_function(
            "split",
            x,
            num_or_size_splits=num_or_size_splits,
            axis=axis,
            with_remainder=with_remainder,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def split(
        self: ivy.Container,
        /,
        *,
        num_or_size_splits: Optional[Union[int, Sequence[int]]] = None,
        axis: int = 0,
        with_remainder: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> List[ivy.Container]:
        """
        ivy.Container instance method variant of ivy.split. This method simply
        wraps the function, and so the docstring for ivy.split also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            array to be divided into sub-arrays.
        num_or_size_splits
            Number of equal arrays to divide the array into along the given axis if an
            integer. The size of each split element if a sequence of integers. Default
            is to divide into as many 1-dimensional arrays as the axis dimension.
        axis
            The axis along which to split, default is ``0``.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is ``False``.
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
            list of containers of sub-arrays.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([2, 1, 5, 9]), b=ivy.array([3, 7, 2, 11]))
        >>> y = x.split(num_or_size_splits=2)
        >>> print(y)
        [{
            a: ivy.array([2, 1]),
            b: ivy.array([3, 7])
        }, {
            a: ivy.array([5, 9]),
            b: ivy.array([2, 11])
        }]
        """
        return self.static_split(
            self,
            num_or_size_splits=num_or_size_splits,
            axis=axis,
            with_remainder=with_remainder,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_permute_dims(
        x: ivy.Container,
        /,
        axes: Tuple[int, ...],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.permute_dims. This method simply
        wraps the function, and so the docstring for ivy.permute_dims also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axes
            tuple containing a permutation of (0, 1, ..., N-1) where N is the number
            of axes (dimensions) of x.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self`` permuted along the given axes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), b=ivy.array([[3., 4., 5.]]))
        >>> y = ivy.Container.static_permute_dims(x, axes=(1, 0))
        >>> print(y)
        {
            a:ivy.array([[0.],[1.],[2.]]),
            b:ivy.array([[3.],[4.],[5.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "permute_dims",
            x,
            axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def permute_dims(
        self: ivy.Container,
        /,
        axes: Tuple[int, ...],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.permute_dims. This method simply
        wraps the function, and so the docstring for ivy.permute_dims also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        axes
            tuple containing a permutation of (0, 1, ..., N-1) where N is the number
            of axes (dimensions) of x.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``self`` permuted along the given axes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0., 1., 2.]]), b=ivy.array([[3., 4., 5.]]))
        >>> y = x.permute_dims(axes=(1, 0))
        >>> print(y)
        {
            a:ivy.array([[0.],[1.],[2.]]),
            b:ivy.array([[3.],[4.],[5.]])
        }
        """
        return self.static_permute_dims(
            self,
            axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_flip(
        x: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.flip. This method simply
        wraps the function, and so the docstring for ivy.flip also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            axis (or axes) along which to flip. If axis is None,
            all input array axes are flipped. If axis is negative,
            axis is counted from the last dimension. If provided more
            than one axis, only the specified axes. Default: None.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type and
            shape as ``x`` and whose elements, relative to ``x``, are reordered.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),
        ...                   b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container.static_flip(x)
        >>> print(y)
        {
            a: ivy.array([1, 0, -1]),
            b: ivy.array([4, 3, 2])
        }

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),
        ...                   b=ivy.array([2, 3, 4]))
        >>> y = ivy.Container.static_flip(x, axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 0, -1]),
            b: ivy.array([4, 3, 2])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "flip",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def flip(
        self: ivy.Container,
        /,
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.flip. This method simply wraps the
        function, and so the docstring for ivy.flip also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis (or axes) along which to flip. If axis is None,
            all input array axes are flipped. If axis is negative,
            axis is counted from the last dimension. If provided
            more than one axis, only the specified axes. Default: None.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        out
            optional output container, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type and
            shape as ``self`` and whose elements, relative to ``self``, are reordered.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),
        ...                   b=ivy.array([2, 3, 4]))
        >>> y = x.flip()
        >>> print(y)
        {
            a: ivy.array([1, 0, -1]),
            b: ivy.array([4, 3, 2])
        }

        >>> x = ivy.Container(a=ivy.array([-1, 0, 1]),
        ...                   b=ivy.array([2, 3, 4]))
        >>> y = x.flip(axis=0)
        >>> print(y)
        {
            a: ivy.array([1, 0, -1]),
            b: ivy.array([4, 3, 2])
        }
        """
        return self.static_flip(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_reshape(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        copy: Optional[bool] = None,
        out: Optional[ivy.Container] = None,
        order: str = "C",
        allowzero: bool = True,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.reshape. This method simply wraps the
        function, and so the docstring for ivy.reshape also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.

        shape
            The new shape should be compatible with the original shape.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.
        order
            Read the elements of x using this index order, and place the elements into
            the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'

        Returns
        -------
        ret
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]),
        ...                   b=ivy.array([0, 1, 2, 3, 4, 5]))
        >>> y = ivy.Container.static_reshape(x, (3,2))
        >>> print(y)
        {
            a: ivy.array([[0, 1],
                          [2, 3],
                          [4, 5]]),
            b: ivy.array([[0, 1],
                          [2, 3],
                          [4, 5]])
        }

        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]),
        ...                   b=ivy.array([0, 1, 2, 3, 4, 5]))
        >>> y = ivy.Container.static_reshape(x, (3,2), order='F')
        >>> print(y)
        {
            a: ivy.array([[0, 3],
                          [1, 4],
                          [2, 5]]),
            b: ivy.array([[0, 3],
                          [1, 4],
                          [2, 5]])
        }


        """
        return ContainerBase.cont_multi_map_in_function(
            "reshape",
            x,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            allowzero=allowzero,
            out=out,
            order=order,
        )

    def reshape(
        self: ivy.Container,
        /,
        shape: Union[ivy.Shape, ivy.NativeShape, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        copy: Optional[bool] = None,
        order: str = "C",
        allowzero: bool = True,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.reshape. This method
        simply wraps the function, and so the docstring for ivy.reshape also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container.
        shape
            The new shape should be compatible with the original shape.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.
        copy
            boolean indicating whether or not to copy the input array.
            If True, the function must always copy.
            If False, the function must never copy and must
            raise a ValueError in case a copy would be necessary.
            If None, the function must reuse existing memory buffer if possible
            and copy otherwise. Default: ``None``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        order
            Read the elements of the input container using this index order,
            and place the elements into the reshaped array using this index order.
            ‘C’ means to read / write the elements using C-like index order,
            with the last axis index changing fastest, back to the first axis index
            changing slowest.
            ‘F’ means to read / write the elements using Fortran-like index order, with
            the first index changing fastest, and the last index changing slowest.
            Note that the ‘C’ and ‘F’ options take no account of the memory layout
            of the underlying array, and only refer to the order of indexing.
            Default order is 'C'
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``self``
            and elements as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]),
        ...                   b=ivy.array([0, 1, 2, 3, 4, 5]))
        >>> y = x.reshape((2,3))
        >>> print(y)
        {
            a: ivy.array([[0, 1, 2],
                          [3, 4, 5]]),
            b: ivy.array([[0, 1, 2],
                          [3, 4, 5]])
        }

        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]),
        ...                   b=ivy.array([0, 1, 2, 3, 4, 5]))
        >>> y = x.reshape((2,3), order='F')
        >>> print(y)
        {
            a: ivy.array([[0, 2, 4],
                          [1, 3, 5]]),
            b: ivy.array([[0, 2, 4],
                          [1, 3, 5]])
        }
        """
        return self.static_reshape(
            self,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            allowzero=allowzero,
            out=out,
            order=order,
        )

    @staticmethod
    def static_roll(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        shift: Union[int, Tuple[int, ...], ivy.Container],
        *,
        axis: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positivclipe, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_roll(x, 1)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([5., 3., 4.])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]))
        >>> shift = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, shift)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([4., 5., 3.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "roll",
            x,
            shift,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def roll(
        self: ivy.Container,
        /,
        shift: Union[int, Sequence[int], ivy.Container],
        *,
        axis: Optional[Union[int, Sequence[int], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positive, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``self`` and whose
            elements, relative to ``self``, are shifted.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.roll(1)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([5., 3., 4.])
        }

        """
        return self.static_roll(
            self,
            shift,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_squeeze(
        x: ivy.Container,
        /,
        axis: Union[int, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.squeeze. This method simply
        wraps the function, and so the docstring for ivy.squeeze also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            input container.
        axis
            axis (or axes) to squeeze.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]),
        ...                   b=ivy.array([[[11.], [12.]]]))
        >>> y = ivy.Container.static_squeeze(x, 0)
        >>> print(y)
        {
            a: ivy.array([[10., 11.]]),
            b: ivy.array([[11., 12.]])
        }

        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]),
        ...                   b=ivy.array([[[11.], [12.]]]))
        >>> y = ivy.Container.static_squeeze(x, [0, 2])
        >>> print(y)
        {
            a: ivy.array([[10.], [11.]]),
            b: ivy.array([[11.], [12.]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "squeeze",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def squeeze(
        self: ivy.Container,
        /,
        axis: Union[int, Sequence[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.squeeze. This method simply wraps
        the function, and so the docstring for ivy.squeeze also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        axis
            axis (or axes) to squeeze.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]),
        ...                   b=ivy.array([[[11.], [12.]]]))
        >>> y = x.squeeze(2)
        >>> print(y)
        {
            a: ivy.array([[10., 11.]]),
            b: ivy.array([[11., 12.]])
        }

        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]),
        ...                   b=ivy.array([[[11.], [12.]]]))
        >>> y = x.squeeze(0)
        >>> print(y)
        {
            a: ivy.array([[10.], [11.]]),
            b: ivy.array([[11.], [12.]])
        }
        """
        return self.static_squeeze(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_stack(
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        /,
        *,
        axis: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stack. This method simply wraps the
        function, and so the docstring for ivy.stack also applies to this method
        with minimal changes.

        Parameters
        ----------
        xs
            Container with leaves to join. Each array leavve must have the same shape.
        axis
            axis along which the array leaves will be joined. More details can be found
            in the docstring for ivy.stack.

        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> z = ivy.Container.static_stack(x,axis = 1)
        >>> print(z)
        {
            a: ivy.array([[0, 2],
                        [1, 3]]),
            b: ivy.array([[4],
                        [5]])
        }

        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> z = ivy.Container.static_stack([x,y])
        >>> print(z)
        {
            a: ivy.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: ivy.array([[[4, 5]],
                        [[1, 0]]])
        }

        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> z = ivy.Container.static_stack([x,y],axis=1)
        >>> print(z)
        {
            a: ivy.array([[[0, 1],
                        [3, 2]],
                        [[2, 3],
                        [1, 0]]]),
            b: ivy.array([[[4, 5],
                        [1, 0]]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "stack",
            xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def stack(
        self: ivy.Container,
        /,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        *,
        axis: int = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.stack. This method
        simply wraps the function, and so the docstring for ivy.stack
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Container with leaves to join with leaves of other arrays/containers.
             Each array leave must have the same shape.
        xs
            Container with other leaves to join.
            Each array leave must have the same shape.
        axis
            axis along which the array leaves will be joined. More details can be found
            in the docstring for ivy.stack.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container with the results.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> x.stack([y])
        {
            a: ivy.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: ivy.array([[[4, 5]],
                        [[1, 0]]])
        }
        """
        new_xs = xs.cont_copy() if ivy.is_ivy_container(xs) else xs.copy()
        new_xs.insert(0, self.cont_copy())
        return self.static_stack(
            new_xs,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_repeat(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        repeats: Union[int, Iterable[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.repeat. This method simply wraps the
        function, and so the docstring for ivy.repeat also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_repeat(2)
        >>> print(y)
        {
            a: ivy.array([0., 0., 1., 1., 2., 2.]),
            b: ivy.array([3., 3., 4., 4., 5., 5.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "repeat",
            x,
            repeats,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def repeat(
        self: ivy.Container,
        /,
        repeats: Union[int, Iterable[int]],
        *,
        axis: Optional[Union[int, Sequence[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.repeat. This method
        simply wraps the function, and so the docstring for ivy.repeat
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input container.
        repeats
            The number of repetitions for each element. repeats is broadcast to fit the
            shape of the given axis.
        axis
            The axis along which to repeat values. By default, use the flattened input
            array, and return a flat output array.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The output container with repreated leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.repeat(2)
        >>> print(y)
        {
            a: ivy.array([0., 0., 1., 1., 2., 2.]),
            b: ivy.array([3., 3., 4., 4., 5., 5.])
        }
        """
        return self.static_repeat(
            self,
            repeats,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_tile(
        x: ivy.Container,
        /,
        repeats: Iterable[int],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.tile. This method simply
        wraps the function, and so the docstring for ivy.tile also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input Container.
        repeats
            The number of repetitions of x along each axis.
        out
            optional output array, for writing the result to. It must have
            a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The container output with tiled leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container.static_tile((2,3))
        >>> print(y)
        {
            a: ivy.array([[0,1,0,1,0,1],
                          [2,3,2,3,2,3],
                          [0,1,0,1,0,1],
                          [2,3,2,3,2,3]]),
            b: ivy.array([[4,5,4,5,4,5],
                          [4,5,4,5,4,5]])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "tile",
            x,
            repeats,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tile(
        self: ivy.Container,
        /,
        repeats: Iterable[int],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.tile. This method simply wraps the
        function, and so the docstring for ivy.tile also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container.
        repeats
            The number of repetitions of x along each axis.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The container output with tiled leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = x.tile((2,3))
        >>> print(y)
        {
            a: (<classivy.array.array.Array>shape=[4,6]),
            b: (<classivy.array.array.Array>shape=[2,6])
        }

        """
        return self.static_tile(
            self,
            repeats,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_constant_pad(
        x: ivy.Container,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.constant_pad. This method simply
        wraps the function, and so the docstring for ivy.constant_pad also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input container with leaves to pad.
        pad_width
            Number of values padded to the edges of each axis.
            Specified as ((before_1, after_1), … (before_N, after_N)), where N
            is number of axes of x.
        value
            The constant value to pad the array with.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            Output container with padded array leaves of rank equal to x with
            shape increased according to pad_width.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([1, 2, 3]), b = ivy.array([4, 5, 6]))
        >>> y = ivy.Container.static_constant_pad(x, pad_width = [[2, 3]])
        >>> print(y)
        {
            a: ivy.array([0, 0, 1, 2, 3, 0, 0, 0]),
            b: ivy.array([0, 0, 4, 5, 6, 0, 0, 0])
        }

        """
        return ContainerBase.cont_multi_map_in_function(
            "constant_pad",
            x,
            pad_width,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def constant_pad(
        self: ivy.Container,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.constant_pad. This method simply
        wraps the function, and so the docstring for ivy.constant_pad also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            Input container with leaves to pad.
        pad_width
            Number of values padded to the edges of each axis.
            Specified as ((before_1, after_1), … (before_N, after_N)), where N
            is number of axes of x.
        value
            The constant value to pad the array with.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            Output container with padded array leaves of rank equal to x with
            shape increased according to pad_width.

        Examples
        --------
        >>> x = ivy.Container(a = ivy.array([1, 2, 3]), b = ivy.array([4, 5, 6]))
        >>> y = x.constant_pad(pad_width = [[2, 3]])
        >>> print(y)
        {
            a: ivy.array([0, 0, 1, 2, 3, 0, 0, 0]),
            b: ivy.array([0, 0, 4, 5, 6, 0, 0, 0])
        }
        """
        return self.static_constant_pad(
            self,
            pad_width,
            value=value,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_zero_pad(
        x: ivy.Container,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.zero_pad. This method simply
        wraps the function, and so the docstring for ivy.zero_pad also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis. Specified as
            ((before_1, after_1), … (before_N, after_N)),
            where N is number of axes of x.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Padded array of rank equal to x with shape increased according to pad_width.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a = ivy.array([1., 2., 3.]), b = ivy.array([3., 4., 5.]))
        >>> y = ivy.zero_pad(x, pad_width = [[2, 3]])
        >>> print(y)
        {
            a: ivy.array([0., 0., 1., 2., 3., 0., 0., 0.]),
            b: ivy.array([0., 0., 3., 4., 5., 0., 0., 0.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "zero_pad",
            x,
            pad_width,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def zero_pad(
        self: ivy.Container,
        /,
        pad_width: Iterable[Tuple[int]],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.zero_pad. This method simply wraps
        the function, and so the docstring for ivy.zero_pad also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input array to pad.
        pad_width
            Number of values padded to the edges of each axis. Specified as
            ((before_1, after_1), … (before_N, after_N)),
            where N is number of axes of x.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            Padded array of rank equal to x with shape increased according to pad_width.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a = ivy.array([1., 2., 3.]), b = ivy.array([3., 4., 5.]))
        >>> y = x.zero_pad(pad_width = [[2, 3]])
        >>> print(y)
        {
            a: ivy.array([0., 0., 1., 2., 3., 0., 0., 0.]),
            b: ivy.array([0., 0., 3., 4., 5., 0., 0., 0.])
        }

        """
        return self.static_zero_pad(
            self,
            pad_width,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_swapaxes(
        x: ivy.Container,
        axis0: int,
        axis1: int,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.swapaxes. This method simply
        wraps the function, and so the docstring for ivy.swapaxes also applies to
        this method with minimal changes.

        Parameters
        ----------
        x
            Input container
        axis0
            First axis to be swapped.
        axis1
            Second axis to be swapped.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            x with its axes permuted.

        >>> a = ivy.array([[1, 2, 3], [4, 5, 6]])
        >>> b = ivy.array([[7, 8, 9], [10, 11, 12]])
        >>> x = ivy.Container(a = a, b = b)
        >>> y = x.swapaxes(0, 1)
        >>> print(y)
        {
            a: ivy.array([[1, 4],
                          [2, 5],
                          [3, 6]]),
            b: ivy.array([[7, 10],
                          [8, 11],
                          [9, 12]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "swapaxes",
            x,
            axis0,
            axis1,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def swapaxes(
        self: ivy.Container,
        axis0: int,
        axis1: int,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.swapaxes. This method simply wraps
        the function, and so the docstring for ivy.swapaxes also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container.
        axis0
            First axis to be swapped.
        axis1
            Second axis to be swapped.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            x with its axes permuted.

        Examples
        --------
        >>> a = ivy.array([[1, 2, 3], [4, 5, 6]])
        >>> b = ivy.array([[7, 8, 9], [10, 11, 12]])
        >>> x = ivy.Container(a = a, b = b)
        >>> y = x.swapaxes(0, 1)
        >>> print(y)
        {
            a: ivy.array([[1, 4],
                          [2, 5],
                          [3, 6]]),
            b: ivy.array([[7, 10],
                          [8, 11],
                          [9, 12]])
        }
        """
        return self.static_swapaxes(
            self,
            axis0,
            axis1,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_unstack(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        axis: int = 0,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.unstack. This method
        simply wraps the function, and so the docstring for ivy.unstack
        also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array or container to unstack.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            List of arrays, unpacked along specified dimensions, or containers
            with arrays unpacked at leaves

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> y = ivy.Container.static_unstack(x, axis=0)
        >>> print(y)
        [{
            a: ivy.array([[1, 2],
                         [3, 4]]),
            b: ivy.array([[9, 10],
                         [11, 12]])
        }, {
            a: ivy.array([[5, 6],
                         [7, 8]]),
             b: ivy.array([[13, 14],
                          [15, 16]])
        }]

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> y = ivy.Container.static_unstack(x, axis=1, keepdims=True)
        >>> print(y)
        [{
            a: ivy.array([[[1, 2]],
                         [[5, 6]]]),
            b: ivy.array([[[9, 10]],
                         [[13, 14]]])
        }, {
            a: ivy.array([[[3, 4]],
                         [[7, 8]]]),
            b: ivy.array([[[11, 12]],
                         [[15, 16]]])
        }]
        """
        return ContainerBase.cont_multi_map_in_function(
            "unstack",
            x,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def unstack(
        self: ivy.Container,
        /,
        *,
        axis: int = 0,
        keepdims: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.unstack. This method
        simply wraps the function, and so the docstring for ivy.unstack
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input container to unstack at leaves.
        axis
            Axis for which to unpack the array.
        keepdims
            Whether to keep dimension 1 in the unstack dimensions. Default is ``False``.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.

        Returns
        -------
        ret
            Containers with arrays unpacked at leaves

        Examples
        --------
        With one :class:`ivy.Container` instances:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                            b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> x.unstack(axis=0)
        [{
            a: ivy.array([[1, 2],
                         [3, 4]]),
            b: ivy.array([[9, 10],
                          [11, 12]])
        }, {
            a: ivy.array([[5, 6],
                          [7, 8]]),
            b: ivy.array([[13, 14],
                          [15, 16]])
        }]
        """
        return self.static_unstack(
            self,
            axis=axis,
            keepdims=keepdims,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_clip(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x_min: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        x_max: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.clip. This method simply wraps the
        function, and so the docstring for ivy.clip also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            Input array or container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_clip(x, 1., 5.)
        >>> print(y)
        {
            a: ivy.array([1., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }

        With multiple :class:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]),
        ...                   b=ivy.array([3., 4., 5.]))
        >>> x_min = ivy.Container(a=0, b=0)
        >>> x_max = ivy.Container(a=1, b=1)
        >>> y = ivy.Container.static_clip(x, x_min, x_max)
        >>> print(y)
        {
            a: ivy.array([0., 1., 1.]),
            b: ivy.array([1., 1., 1.])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "clip",
            x,
            x_min,
            x_max,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def clip(
        self: ivy.Container,
        x_min: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        x_max: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.clip. This method simply wraps the
        function, and so the docstring for ivy.clip also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
        key_chains
            The key-chains to apply or not apply the method to. Default is ``None``.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is ``True``.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is ``False``.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is ``False``.
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.clip(1,2)
        >>> print(y)
        {
            a: ivy.array([1., 1., 2.]),
            b: ivy.array([2., 2., 2.])
        }
        """
        return self.static_clip(
            self,
            x_min,
            x_max,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
