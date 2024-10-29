from ivy.data_classes.container.base import ContainerBase
from typing import Union, List, Dict, Optional, Tuple
import ivy


class _ContainerWithNormsExperimental(ContainerBase):
    @staticmethod
    def static_l1_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.l1_normalize. This method
        simply wraps the function, and so the docstring for ivy.l1_normalize
        also applies to this method with minimal changes.

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
        >>> y = ivy.Container.static_l1_normalize(x, axis=1)
        >>> print(y)
        {
            a: ivy.array([[0.1, 0.3, 0.5],
                          [0.35, 0.45, 0.55]]),
            b: ivy.array([[-0.5, -0.5],
                          [-0.5, -0.25]])
        }
        """
        return ContainerBase.cont_multi_map_in_function(
            "l1_normalize",
            x,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def l1_normalize(
        self: ivy.Container,
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l1_normalize. This
        method simply wraps the function, and so the docstring for
        ivy.l1_normalize also applies to this method with minimal changes.

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        axis
            The axis along which to normalize.
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples).
            Default is False.
        out
            Optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container containing the normalized leaves.
        """
        return self.static_l1_normalize(
            self,
            axis=axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_l2_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.l2_normalize. This method
        simply wraps the function, and so the docstring for ivy.l2_normalize
        also applies to this method with minimal changes.

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
        self: ivy.Container,
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l2_normalize. This
        method simply wraps the function, and so the docstring for
        ivy.l2_normalize also applies to this method with minimal changes.

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
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]))
        >>> y = x.l2_normalize(axis=1)
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
        mean: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        variance: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: Union[bool, ivy.Container] = False,
        eps: Union[float, ivy.Container] = 1e-5,
        momentum: Union[float, ivy.Container] = 1e-1,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """ivy.Container static method variant of ivy.batch_norm. This method
        simply wraps the function, and so the docstring for ivy.batch_norm also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of default shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
            C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. It can be of any shape
            braodcastable to (N,*S,C).
        variance
            Variance array used for input's normalization. It can be of any shape
            braodcastable to (N,*S,C).
        offset
            An offset array. If present, will be added to the normalized input.
            It can be of any shape broadcastable to (N,*S,C).
        scale
            A scale array. If present, the scale is applied to the normalized input.
            It can be of any shape broadcastable to (N,*S,C).
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
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
            data_format=data_format,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def batch_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        variance: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: Union[bool, ivy.Container] = False,
        eps: Union[float, ivy.Container] = 1e-5,
        momentum: Union[float, ivy.Container] = 1e-1,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """ivy.Container instance method variant of ivy.batch_norm. This method
        simply wraps the function, and so the docstring for ivy.batch_norm also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of default shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array used for input's normalization. It can be of any shape
            braodcastable to (N,*S,C).
        variance
            Variance array used for input's normalization. It can be of any shape
            braodcastable to (N,*S,C).
        offset
            An offset array. If present, will be added to the normalized input.
            It can be of any shape broadcastable to (N,*S,C).
        scale
            A scale array. If present, the scale is applied to the normalized input.
            It can be of any shape broadcastable to (N,*S,C).
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
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
        return self.static_batch_norm(
            self,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            data_format=data_format,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_instance_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        variance: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: Union[bool, ivy.Container] = False,
        eps: Union[float, ivy.Container] = 1e-5,
        momentum: Union[float, ivy.Container] = 1e-1,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """ivy.Container static method variant of ivy.instance_norm. This
        method simply wraps the function, and so the docstring for
        ivy.instance_norm also applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of shape default (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array of size C used for input's normalization.
        variance
            Variance array of size C used for input's normalization.
        offset
            An offset array of size C. If present, will be added
             to the normalized input.
        scale
            A scale array of size C. If present, the scale
             is applied to the normalized input.
        training
            If true, calculate and use the mean and variance of `x`.
             Otherwise, use the provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
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
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def instance_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        mean: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        variance: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        training: Union[bool, ivy.Container] = False,
        eps: Union[float, ivy.Container] = 1e-5,
        momentum: Union[float, ivy.Container] = 1e-1,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[
            Tuple[
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
                Union[ivy.Array, ivy.Container],
            ]
        ] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> Tuple[ivy.Container, ivy.Container, ivy.Container]:
        """ivy.Container instance method variant of ivy.instance_norm. This
        method simply wraps the function, and so the docstring for
        ivy.instance_norm also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of shape default (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        mean
            Mean array of size C used for input's normalization.
        variance
            Variance array of size C used for input's normalization.
        offset
            An offset array of size C. If present, will be added
            to the normalized input.
        scale
            A scale array of size C. If present, the scale is
            applied to the normalized input.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
        eps
            A small float number to avoid dividing by 0.
        momentum
             the value used for the running_mean and running_var computation.
              Default value is 0.1.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
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
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_group_norm(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        num_groups: Union[int, ivy.Container] = 1,
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        eps: Union[float, ivy.Container] = 1e-5,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.group_norm. This method
        simply wraps the function, and so the docstring for ivy.group_norm also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
            Input array of default shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
            C corresponds to the channel dimension.
        num_groups
            number of groups to separate the channels into
        offset
            An offset array of size C. If present, will be added
            to the normalized input.
        scale
            A scale array of size C. If present, the scale is
            applied to the normalized input.
        eps
            A small float number to avoid dividing by 0.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
        out
            optional output arrays, for writing the result to.

        Returns
        -------
        ret
            The normalized array.
        """
        return ContainerBase.cont_multi_map_in_function(
            "group_norm",
            x,
            num_groups,
            scale=scale,
            offset=offset,
            eps=eps,
            out=out,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def group_norm(
        self: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        num_groups: Union[int, ivy.Container] = 1,
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array, ivy.Container]] = None,
        eps: Union[float, ivy.Container] = 1e-5,
        data_format: Union[str, ivy.Container] = "NSC",
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.group_norm. This method
        simply wraps the function, and so the docstring for ivy.group_norm also
        applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input array of default shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
            C corresponds to the channel dimension.
        num_groups
            number of groups to separate the channels into
        offset
            An offset array of size C. If present, will be added
            to the normalized input.
        scale
            A scale array of size C. If present, the scale is
            applied to the normalized input.
        eps
            A small float number to avoid dividing by 0.
        data_format
            The ordering of the dimensions in the input, one of "NSC" or "NCS",
            where N is the batch dimension, S represents any number of spatial
            dimensions and C is the channel dimension. Default is "NSC".
        out
            optional output arrays, for writing the result to.

        Returns
        -------
        ret
            The normalized array.
        """
        return self.static_group_norm(
            self,
            num_groups,
            scale=scale,
            offset=offset,
            eps=eps,
            out=out,
            data_format=data_format,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    @staticmethod
    def static_lp_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        p: Union[float, ivy.Container] = 2,
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container static method variant of ivy.lp_normalize. This method
        simply wraps the function, and so the docstring for ivy.lp_normalize
        also applies to this method with minimal changes.

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
        self: ivy.Container,
        p: Union[float, ivy.Container] = 2,
        axis: Optional[Union[int, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str], ivy.Container]] = None,
        to_apply: Union[bool, ivy.Container] = True,
        prune_unapplied: Union[bool, ivy.Container] = False,
        map_sequences: Union[bool, ivy.Container] = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """ivy.Container instance method variant of ivy.l2_normalize. This
        method simply wraps the function, and so the docstring for
        ivy.l2_normalize also applies to this method with minimal changes.

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
        >>> x = ivy.Container(a=ivy.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]]),
        ...                    b=ivy.array([[-1., -1.], [-1., -0.5]]))
        >>> y = x.lp_normalize(axis=1)
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
