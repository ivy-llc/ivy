# global
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Sequence,
    Tuple,
    Literal,
    Any,
    Callable,
    Iterable
)
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithManipulationExperimental(ContainerBase):
    @staticmethod
    def static_moveaxis(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.moveaxis. This method simply wraps
        the function, and so the docstring for ivy.moveaxis also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            The container with the arrays whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with moved axes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> ivy.Container.static_moveaxis(x, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "moveaxis",
            a,
            source,
            destination,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def moveaxis(
        self: ivy.Container,
        source: Union[int, Sequence[int]],
        destination: Union[int, Sequence[int]],
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.moveaxis. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            The container with the arrays whose axes should be reordered.
        source
            Original positions of the axes to move. These must be unique.
        destination
            Destination positions for each of the original axes.
            These must also be unique.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            Container including arrays with moved axes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.zeros((3, 4, 5)), b=ivy.zeros((2,7,6)))
        >>> x.moveaxis(, 0, -1).shape
        {
            a: (4, 5, 3)
            b: (7, 6, 2)
        }
        """
        return self.static_moveaxis(self, source, destination, out=out)

    @staticmethod
    def static_heaviside(
        x1: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.heaviside. This method simply wraps
        the function, and so the docstring for ivy.heaviside also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            input container including the arrays.
        x2
            values to use where the array is zero.
        out
            optional output container array, for writing the result to.

        Returns
        -------
        ret
            output container with element-wise Heaviside step function of each array.

        Examples
        --------
        With :class:`ivy.Array` input:
        >>> x1 = ivy.Container(a=ivy.array([-1.5, 0, 2.0]), b=ivy.array([3.0, 5.0])
        >>> x2 = ivy.Container(a=0.5, b=[1.0, 2.0])
        >>> ivy.Container.static_heaviside(x1, x2)
        {
            a: ivy.array([ 0. ,  0.5,  1. ])
            b: ivy.array([1.0, 1.0])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "heaviside",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def heaviside(
        self: ivy.Container,
        x2: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.heaviside. This method simply
        wraps the function, and so the docstring for ivy.heaviside also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input container including the arrays.
        x2
            values to use where the array is zero.
        out
            optional output container array, for writing the result to.

        Returns
        -------
        ret
            output container with element-wise Heaviside step function of each array.

        Examples
        --------
        With :class:`ivy.Array` input:
        >>> x1 = ivy.Container(a=ivy.array([-1.5, 0, 2.0]), b=ivy.array([3.0, 5.0])
        >>> x2 = ivy.Container(a=0.5, b=[1.0, 2.0])
        >>> x1.heaviside(x2)
        {
            a: ivy.array([ 0. ,  0.5,  1. ])
            b: ivy.array([1.0, 1.0])
        }
        """
        return self.static_heaviside(self, x2, out=out)

    @staticmethod
    def static_flipud(
        m: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.flipud. This method simply wraps
        the function, and so the docstring for ivy.flipud also applies to this method
        with minimal changes.

        Parameters
        ----------
        m
            the container with arrays to be flipped.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 0.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]), b=ivy.arange(4))
        >>> ivy.Container.static_flipud(m)
        {
            a: ivy.array(
                [[ 0.,  0.,  3.],
                 [ 0.,  2.,  0.],
                 [ 1.,  0.,  0.]]
            )
            b: ivy.array([3, 2, 1, 0])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "flipud",
            m,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def flipud(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.flipud. This method simply
        wraps the function, and so the docstring for ivy.flipud also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            the container with arrays to be flipped.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 0.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]), b=ivy.arange(4))
        >>> m.flipud()
        {
            a: ivy.array(
                [[ 0.,  0.,  3.],
                 [ 0.,  2.,  0.],
                 [ 1.,  0.,  0.]]
            )
            b: ivy.array([3, 2, 1, 0])
        }
        """
        return self.static_flipud(self, out=out)

    def vstack(
        self: ivy.Container,
        /,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        *,
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

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> x.vstack([y])
        {
            a: ivy.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: ivy.array([[[4, 5]],
                        [[1, 0]]])
        }
        """
        new_xs = xs.copy()
        new_xs.insert(0, self.copy())
        return self.static_vstack(
            new_xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_vstack(
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.stack. This method simply wraps the
        function, and so the docstring for ivy.vstack also applies to this method
        with minimal changes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> c = ivy.Container(a=[ivy.array([1,2,3]), ivy.array([0,0,0])],
                              b=ivy.arange(3))
        >>> ivy.Container.static_vstack(c)
        {
            a: ivy.array([[1, 2, 3],
                          [0, 0, 0]]),
            b: ivy.array([[0],
                          [1],
                          [2]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "vstack",
            xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hstack(
        self: ivy.Container,
        /,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.hstack. This method
        simply wraps the function, and so the docstring for ivy.hstack
        also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> x.hstack([y])
        {
            a: ivy.array([[0, 1, 3, 2],
                          [2, 3, 1, 0]]),
            b: ivy.array([[4, 5, 1, 0]])
        }
        """
        new_xs = xs.copy()
        new_xs.insert(0, self.copy())
        return self.static_hstack(
            new_xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hstack(
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hstack. This method simply wraps the
        function, and so the docstring for ivy.hstack also applies to this method
        with minimal changes.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> c = ivy.Container(a=[ivy.array([1,2,3]), ivy.array([0,0,0])])
        >>> ivy.Container.static_hstack(c)
        {
            a: ivy.array([1, 2, 3, 0, 0, 0])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "hstack",
            xs,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_rot90(
        m: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        k: Optional[int] = 1,
        axes: Optional[Tuple[int, int]] = (0, 1),
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.rot90.
        This method simply wraps the function, and so the docstring for
        ivy.rot90 also applies to this method with minimal changes.

        Parameters
        ----------
        m
            Input array of two or more dimensions.
        k
            Number of times the array is rotated by 90 degrees.
        axes
            The array is rotated in the plane defined by the axes. Axes must be
            different.
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
            Container with a rotated view of m.
            
        Examples
        --------
        >>> m = ivy.Container(a=ivy.array([[1,2], [3,4]]),\
                        b=ivy.array([[1,2,3,4],\
                                    [7,8,9,10]]))
        >>> ivy.Container.static_rot90(m)
        {
            a: ivy.array([[2, 4],
                          [1, 3]]),
            b: ivy.array([[4, 10],
                          [3, 9],
                          [2, 8],
                          [1, 7]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "rot90",
            m,
            k=k,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def rot90(
        self: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        k: Optional[int] = 1,
        axes: Optional[Tuple[int, int]] = (0, 1),
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.rot90.
        This method simply wraps the function, and so the docstring for
        ivy.rot90 also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of two or more dimensions.
        k
            Number of times the array is rotated by 90 degrees.
        axes
            The array is rotated in the plane defined by the axes. Axes must be
            different.
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
            Container with a rotated view of input array.

        Examples
        --------
        >>> m = ivy.Container(a=ivy.array([[1,2], [3,4]]),\
                        b=ivy.array([[1,2,3,4],\
                                    [7,8,9,10]]))
        >>> m.rot90()
        {
            a: ivy.array([[2, 4],
                          [1, 3]]),
            b: ivy.array([[4, 10],
                          [3, 9],
                          [2, 8],
                          [1, 7]])
        }
        """
        return self.static_rot90(
            self,
            k=k,
            axes=axes,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_top_k(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        k: int,
        /,
        *,
        axis: Optional[int] = -1,
        largest: Optional[bool] = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Tuple[ivy.Container, ivy.Container]] = None,
    ) -> Tuple[ivy.Container, ivy.Container]:
        """ivy.Container static method variant of ivy.top_k. This method simply wraps
        the function, and so the docstring for ivy.top_k also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            The container to compute top_k for.
        k
            Number of top elements to retun must not exceed the array size.
        axis
            The axis along which we must return the top elements default value is 1.
        largest
            If largest is set to False we return k smallest elements of the array.
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
            Default is ``False``
        out:
            Optional output tuple, for writing the result to. Must have two Container,
            with a shape that the returned tuple broadcast to.

        Returns
        -------
        ret
            a container with indices and values.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 2, -4]), b=ivy.array([4., 5., 0.]))
        >>> y = ivy.Container.static_top_k(x, 2)
        >>> print(y)
        {
            a: [
                values = ivy.array([ 2, -1]),
                indices = ivy.array([1, 0])
            ],
            b: [
                values = ivy.array([5., 4.]),
                indices = ivy.array([1, 0])
            ]
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "top_k",
            x,
            k,
            axis=axis,
            largest=largest,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def top_k(
        self: ivy.Container,
        k: int,
        /,
        *,
        axis: Optional[int] = -1,
        largest: Optional[bool] = True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[Tuple[ivy.Container, ivy.Container]] = None,
    ) -> Tuple[ivy.Container, ivy.Container]:
        """ivy.Container instance method variant of ivy.top_k. This method
        simply wraps the function, and so the docstring for ivy.top_k
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The container to compute top_k for.
        k
            Number of top elements to retun must not exceed the array size.
        axis
            The axis along which we must return the top elements default value is 1.
        largest
            If largest is set to False we return k smallest elements of the array.
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
            Default is ``False``
        out:
            Optional output tuple, for writing the result to. Must have two Container,
            with a shape that the returned tuple broadcast to.

        Returns
        -------
        ret
            a container with indices and values.

        Examples
        --------
        With :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([-1, 2, -4]), b=ivy.array([4., 5., 0.]))
        >>> y = x.top_k(2)
        >>> print(y)
        {
            a: [
                values = ivy.array([ 2, -1]),
                indices = ivy.array([1, 0])
            ],
            b: [
                values = ivy.array([5., 4.]),
                indices = ivy.array([1, 0])
            ]
        }
        """
        return self.static_top_k(
            self,
            k,
            axis=axis,
            largest=largest,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_fliplr(
        m: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.fliplr. This method simply wraps
        the function, and so the docstring for ivy.fliplr also applies to this method
        with minimal changes.

        Parameters
        ----------
        m
            the container with arrays to be flipped. Arrays must be at least 2-D.
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
            Default is ``False``
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 1.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]),\
                            b=ivy.array([[1, 2, 3],[4, 5, 6]]))
        >>> ivy.Container.static_fliplr(m)
        {
            a: ivy.array([[0, 0, 1],
                          [0, 2, 0],
                          [3, 0, 0]]),
            b: ivy.array([[3, 2, 1],
                          [6, 5, 4]])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "fliplr",
            m,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def fliplr(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.fliplr. This method simply
        wraps the function, and so the docstring for ivy.fliplr also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            the container with arrays to be flipped. Arrays must be at least 2-D.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the input container's array
            with elements order reversed along axis 1.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> m = ivy.Container(a=ivy.diag([1, 2, 3]),\
                            b=ivy.array([[1, 2, 3],[4, 5, 6]]))
        >>> m.fliplr()
        {
            a: ivy.array([[0, 0, 1],
                          [0, 2, 0],
                          [3, 0, 0]]),
            b: ivy.array([[3, 2, 1],
                          [6, 5, 4]])
        }
        """
        return self.static_fliplr(self, out=out)

    @staticmethod
    def static_i0(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.i0. This method simply wraps
        the function, and so the docstring for ivy.i0 also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            the container with array inputs.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays with the modified Bessel
            function evaluated at each of the elements of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array(4))
        >>> ivy.Container.static_i0(x)
        {
            a: ivy.array([1.26606588, 2.2795853 , 4.88079259])
            b: ivy.array(11.30192195)
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "i0",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def i0(
        self: ivy.Container,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.i0. This method simply
        wraps the function, and so the docstring for ivy.i0 also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            the container with array inputs.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays with the modified Bessel
            function evaluated at each of the elements of x.

        Examples
        --------
        With one :class:`ivy.Container` input:
        >>> x = ivy.Container(a=ivy.array([1, 2, 3]), b=ivy.array(4))
        >>> x.i0()
        {
            a: ivy.array([1.26606588, 2.2795853 , 4.88079259])
            b: ivy.array(11.30192195)
        }
        """
        return self.static_i0(self, out=out)

    @staticmethod
    def static_flatten(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.flatten. This method simply wraps the
        function, and so the docstring for ivy.flatten also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container to flatten at leaves.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            Container with arrays flattened at leaves.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ...                   b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> ivy.flatten(x)
        [{
            a: ivy.array([1, 2, 3, 4, 5, 6, 7, 8])
            b: ivy.array([9, 10, 11, 12, 13, 14, 15, 16])
        }]
        """
        return ContainerBase.multi_map_in_static_method(
            "flatten",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            start_dim=start_dim,
            end_dim=end_dim,
            out=out,
        )

    def flatten(
        self: ivy.Container,
        *,
        start_dim: Optional[int] = 0,
        end_dim: Optional[int] = -1,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.flatten also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input container to flatten at leaves.
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            Container with arrays flattened at leaves.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        ...                   b=ivy.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]]))
        >>> ivy.flatten(x)
        [{
            a: ivy.array([1, 2, 3, 4, 5, 6, 7, 8])
            b: ivy.array([9, 10, 11, 12, 13, 14, 15, 16])
        }]
        """
        return self.static_flatten(self, start_dim=start_dim, end_dim=end_dim, out=out)

    @staticmethod
    def static_pad(
        input: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return ContainerBase.multi_map_in_static_method(
            "pad",
            input,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs,
        )

    def pad(
        self: ivy.Container,
        pad_width: Union[Iterable[Tuple[int]], int],
        /,
        *,
        mode: Optional[
            Union[
                Literal[
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                    "empty",
                ],
                Callable,
            ]
        ] = "constant",
        stat_length: Optional[Union[Iterable[Tuple[int]], int]] = None,
        constant_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        end_values: Optional[Union[Iterable[Tuple[Number]], Number]] = 0,
        reflect_type: Optional[Literal["even", "odd"]] = "even",
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
        **kwargs: Optional[Any],
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.pad. This method simply
        wraps the function, and so the docstring for ivy.pad also applies to
        this method with minimal changes.
        """
        return self.static_pad(
            self,
            pad_width,
            mode=mode,
            stat_length=stat_length,
            constant_values=constant_values,
            end_values=end_values,
            reflect_type=reflect_type,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
            **kwargs,
        )
