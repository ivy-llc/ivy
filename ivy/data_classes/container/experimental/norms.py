from ivy.data_classes.container.base import ContainerBase
from typing import Union, List, Dict, Optional, Tuple
import ivy


class _ContainerWithNormsExperimental(ContainerBase):
    @staticmethod
    def static_l2_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out=None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.l2_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.l2_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = ivy.Container.static_l2_normalize(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.16903085, 0.50709254, 0.84515423],
                          [0.44183609, 0.56807494, 0.69431382]]),
            b: ivy.array([[-0.70710677, -0.70710677],
                          [-0.89442718, -0.44721359]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "l2_normalize",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l2_normalize(
        self,
        axis: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out=None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l2_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.l2_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = x.static_l2_normalize(axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.16903085, 0.50709254, 0.84515423],
                          [0.44183609, 0.56807494, 0.69431382]]),
            b: ivy.array([[-0.70710677, -0.70710677],
                          [-0.89442718, -0.44721359]])
        }
        """
        return self.static_l2_normalize(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_batch_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """
        ivy.Container static method variant of ivy.batch_norm.
        This method simply wraps the function, and so the docstring
        for ivy.batch_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
            C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any
            shape broadcastble to the input shape.
        variance
            Variance array for the input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any shape
            broadcastble to the input shape.
        offset
            An offset array. If present, will be added to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal
            to the size of channel dimension C. If ``training=False`` then it can
            be of any shape broadcastble to the input shape.
        scale
            A scale array. If present, the scale is applied to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal to
            the size of channel dimension C. If ``training=False`` then it can be of
            any shape broadcastble to the input shape.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        out
            optional output arrays, for writing the result to.
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
             Tuple of containers containing
              the normalized input, running mean, and running variance.
        """
        return ContainerBase.cont_multi_map_in_function(
            "batch_norm",
            x,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def batch_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """
        ivy.Container instance method variant of ivy.batch_norm.
        This method simply wraps the function, and so the docstring
        for ivy.batch_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any
            shape broadcastble to the input shape.
        variance
            Variance array for the input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any shape
            broadcastble to the input shape.
        offset
            An offset array. If present, will be added to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal
            to the size of channel dimension C. If ``training=False`` then it can
            be of any shape broadcastble to the input shape.
        scale
            A scale array. If present, the scale is applied to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal to
            the size of channel dimension C. If ``training=False`` then it can be of
            any shape broadcastble to the input shape.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        out
            optional output array, for writing the result to.
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
             Tuple of containers containing
              the normalized input, running mean, and running variance.
        """
        return self.static_batch_norm(
            self,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_instance_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """
        ivy.Container static method variant of ivy.instance_norm.
        This method simply wraps the function, and so the docstring
        for ivy.instance_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any
            shape broadcastble to the input shape.
        variance
            Variance array for the input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any shape
            broadcastble to the input shape.
        offset
            An offset array. If present, will be added to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal
            to the size of channel dimension C. If ``training=False`` then it can
            be of any shape broadcastble to the input shape.
        scale
            A scale array. If present, the scale is applied to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal to
            the size of channel dimension C. If ``training=False`` then it can be of
            any shape broadcastble to the input shape.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        out
            optional output arrays, for writing the result to.
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
             Tuple of containers
              containing the normalized input, running mean, and running variance.
        """
        return ContainerBase.cont_multi_map_in_function(
            "instance_norm",
            x,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def instance_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        variance: Union[ivy.NativeArray, ivy.Array, ivy.Container],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """
        ivy.Container instance method variant of ivy.instance_norm.
        This method simply wraps the function, and so the docstring
        for ivy.instance_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any
            shape broadcastble to the input shape.
        variance
            Variance array for the input's normalization. If ``training=True``
            then it must be one dimensional with size equal to the size of
            channel dimension C. If ``training=False`` then it can be of any shape
            broadcastble to the input shape.
        offset
            An offset array. If present, will be added to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal
            to the size of channel dimension C. If ``training=False`` then it can
            be of any shape broadcastble to the input shape.
        scale
            A scale array. If present, the scale is applied to the normalized input.
            If ``training=True`` then it must be one dimensional with size equal to
            the size of channel dimension C. If ``training=False`` then it can be of
            any shape broadcastble to the input shape.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        out
            optional output array, for writing the result to.
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
             Tuple of containers containing
              the normalized input, running mean, and running variance.
        """
        return self.static_instance_norm(
            self,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_lp_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        p: float = 2,
        axis: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.lp_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.lp_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            The input container with leaves to be normalized.
        p
            The order of the norm.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = ivy.Container.static_lp_normalize(x, p=1, axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.12500000, 0.37500000, 0.62500000],
                          [0.27500000, 0.35000000, 0.42500000]]),
            b: ivy.array([[-1.0000000, -1.0000000],
                          [-0.5000000, -0.2500000]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "lp_normalize",
            x,
            p=p,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def lp_normalize(
        self,
        p: float = 2,
        axis: Optional[int] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l2_normalize.
        This method simply wraps the function, and so the
        docstring for ivy.l2_normalize also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
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
            a container containing the normalized leaves.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]])))
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]]))
        >>> y = x.static_lp_normalize(axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.16903085, 0.50709254, 0.84515423],
                          [0.44183609, 0.56807494, 0.69431382]]),
            b: ivy.array([[-0.70710677, -0.70710677],
                          [-0.89442718, -0.44721359]])
        }
        """
        return self.static_lp_normalize(
            self,
            p=p,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )
