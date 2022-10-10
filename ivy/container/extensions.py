# global
from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


class ContainerWithExtensions(ContainerBase):
    @staticmethod
    def static_sinc(
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
        ivy.Container static method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``x``. The returned
            container must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),\
                              b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = ivy.Container.static_sinc(x)
        >>> print(y)
        {
            a: ivy.array([0.636, -0.212, 0.127]),
            b: ivy.array([-0.090, 0.070, -0.057])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "sinc",
            x,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def sinc(
        self: ivy.Container,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.sinc. This method simply
        wraps the function, and so the docstring for ivy.sinc also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            input container whose elements are each expressed in radians.
            Should have a floating-point data type.
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
            a container containing the sinc of each element in ``self``.
            The returned container must have a floating-point data type
            determined by :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0.5, 1.5, 2.5]),\
                              b=ivy.array([3.5, 4.5, 5.5]))
        >>> y = x.sinc()
        >>> print(y)
        {
            a: ivy.array([0.637,-0.212,0.127]),
            b: ivy.array([-0.0909,0.0707,-0.0579])
        }
        """
        return self.static_sinc(
            self,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_flatten(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        start_dim: int,
        end_dim: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            start_dim=start_dim,
            end_dim=end_dim,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def flatten(
        self: ivy.Container,
        *,
        start_dim: int,
        end_dim: int,
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
    def static_lcm(
        x1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),\
                            b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),\
                            b=ivy.array([32, 40, 25, 13]))
        >>> ivy.Container.lcm(x1, x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return ContainerBase.multi_map_in_static_method(
            "lcm",
            x1,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lcm(
        self: ivy.Container,
        x2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        x1
            first input container.
        x2
            second input container.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            a container containing the the element-wise least common multiples
            of the arrays contained in x1 and x2.

        Examples
        --------
        >>> x1=ivy.Container(a=ivy.array([2, 3, 4]),\
                            b=ivy.array([6, 54, 62, 10]))
        >>> x2=ivy.Container(a=ivy.array([5, 8, 15]),\
                            b=ivy.array([32, 40, 25, 13]))
        >>> x1.lcm(x2)
        {
            a: ivy.array([10, 21, 60]),
            b: ivy.array([96, 1080, 1550, 130])
        }

        """
        return self.static_lcm(
            self,
            x2,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_hann_window(
        window_length: Union[int, ivy.Container],
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,        
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.hann_window. This method simply wraps 
        the function, and so the docstring for ivy.hann_window also applies to this 
        method with minimal changes.

        Parameters
        ----------
        window_length
            container including multiple window sizes.
        periodic
            If True, returns a window to be used as periodic function. 
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container that contains the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.Container.static_hann(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455]) 
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "hann_window",
            window_length,
            periodic,
            dtype,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def hann_window(
        self: ivy.Container,
        periodic: Optional[bool] = True,
        dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.hann_window. This method simply
        wraps the function, and so the docstring for ivy.hann_window also applies to 
        this method with minimal changes.

        Parameters
        ----------
        self
            input container with window sizes.
        periodic
            If True, returns a window to be used as periodic function. 
            If False, return a symmetric window.
        dtype
            The data type to produce. Must be a floating point type.
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            The container containing the Hann windows.

        Examples
        --------
        With one :class:`ivy.Container` input:

        >>> x = ivy.Container(a=3, b=5)
        >>> ivy.hann_window(x)
        {
            a: ivy.array([0.0000, 0.7500, 0.7500])
            b: ivy.array([0.0000, 0.3455, 0.9045, 0.9045, 0.3455]) 
        }
        """
        return self.static_hann_window(
            self,
            periodic,
            dtype,
            out=out)
