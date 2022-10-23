# global
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Sequence,
    Tuple,
)


# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithManipulationExtensions(ContainerBase):
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
        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> ivy.Container.static_stack(x,axis = 1)
        {
            a: ivy.array([[0, 2],
                        [1, 3]]),
            b: ivy.array([[4],
                        [5]])
        }
        >>> x = ivy.Container(a=ivy.array([[0, 1], [2,3]]), b=ivy.array([[4, 5]]))
        >>> y = ivy.Container(a=ivy.array([[3, 2], [1,0]]), b=ivy.array([[1, 0]]))
        >>> ivy.Container.static_stack([x,y])
        {
            a: ivy.array([[[0, 1],
                        [2, 3]],
                        [[3, 2],
                        [1, 0]]]),
            b: ivy.array([[[4, 5]],
                        [[1, 0]]])
        }
        >>> ivy.Container.static_stack([x,y],axis=1)
        {
            a: ivy.array([[[0, 1],
                        [3, 2]],
                        [[2, 3],
                        [1, 0]]]),
            b: ivy.array([[[4, 5],
                        [1, 0]]])
        }
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
