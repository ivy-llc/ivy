# For Review
# global
from typing import Optional, Union, List, Tuple, Dict, Iterable, Sequence
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithManipulation(ContainerBase):
    @staticmethod
    def static_concat(
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        /,
        *,
        axis: Optional[int] = 0,
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
        return ContainerBase.multi_map_in_static_method(
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
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        *,
        axis: Optional[int] = 0,
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
        new_xs = xs.copy()
        new_xs.insert(0, self.copy())
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
        axis: Union[int, Tuple[int], List[int]] = 0,
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of ``x``, but with the dimensions of
            its elements added by one in a given ``axis``.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1.]), \
                              b=ivy.array([3., 4.]), \
                              c=ivy.array([6., 7.]))
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

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]), \
                              c=ivy.array([6., 7., 8.]))
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
        return ContainerBase.multi_map_in_static_method(
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
        axis: Union[int, Tuple[int], List[int]] = 0,
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
        >>> x = ivy.Container(a=ivy.array([[0., 1.], \
                                           [2., 3.]]), \
                              b=ivy.array([[4., 5.], \
                                           [6., 7.]]))
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
        num_or_size_splits: Optional[Union[int, Iterable[int]]] = None,
        axis: int = 0,
        with_remainder: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Union[ivy.Container, List[ivy.Container]]:
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
            The axis along which to split, default is 0.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
            A container with list of sub-arrays.

        """
        return ContainerBase.multi_map_in_static_method(
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
        num_or_size_splits: Optional[Union[int, Iterable[int]]] = None,
        axis: int = 0,
        with_remainder: bool = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Union[ivy.Container, List[ivy.Container]]:
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
            The axis along which to split, default is 0.
        with_remainder
            If the tensor does not split evenly, then store the last remainder entry.
            Default is False.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains will
            be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied. Default
            is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.

        Returns
        -------
            A container with list of sub-arrays.

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
        """
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
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
        """
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
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
            and copy otherwise. Default: None.
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]), \
                              b=ivy.array([0, 1, 2, 3, 4, 5]))
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


        """
        return ContainerBase.multi_map_in_static_method(
            "reshape",
            x,
            shape,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            copy=copy,
            out=out,
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
            and copy otherwise. Default: None.
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``self``
            and elements as ``self``.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0, 1, 2, 3, 4, 5]), \
                              b=ivy.array([0, 1, 2, 3, 4, 5]))
        >>> y = x.reshape((2,3))
        >>> print(y)
        {
            a: ivy.array([[0, 1, 2],
                          [3, 4, 5]]),
            b: ivy.array([[0, 1, 2],
                          [3, 4, 5]])
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
            out=out,
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_roll(x, 1)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([5., 3., 4.])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> shift = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, shift)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([4., 5., 3.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
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
        """
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
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

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]), \
                              b=ivy.array([[[11.], [12.]]]))
        >>> y = x.squeeze(2)
        >>> print(y)
        {
            a: ivy.array([[10., 11.]]),
            b: ivy.array([[11., 12.]])
        }

        >>> x = ivy.Container(a=ivy.array([[[10.], [11.]]]), \
                              b=ivy.array([[[11.], [12.]]]))
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
        axis: Optional[int] = 0,
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
        """
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[int] = 0,
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
        """
        new_xs = xs.copy()
        new_xs.insert(0, self.copy())
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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
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
        return ContainerBase.multi_map_in_static_method(
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
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
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
        reps: Iterable[int],
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
        """
        return ContainerBase.multi_map_in_static_method(
            "tile",
            x,
            reps,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def tile(
        self: ivy.Container,
        /,
        reps: Iterable[int],
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
        """
        return self.static_tile(
            self,
            reps,
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
        """
        return ContainerBase.multi_map_in_static_method(
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
        """
        return ContainerBase.multi_map_in_static_method(
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
        """
        return ContainerBase.multi_map_in_static_method(
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
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_clip(x, 1., 5.)
        >>> print(y)
        {
            a: ivy.array([1., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> x_min = ivy.Container(a=1, b=-1)
        >>> x_max = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, x_min, x_max)
        >>> print(y)
        {
            a: ivy.array([1., 1., 1.]),
            b: ivy.array([-1., -1., -1.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
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
