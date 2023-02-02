from ivy.container.base import ContainerBase
from typing import Union, List, Dict, Optional
import ivy


class ContainerWithNormsExperimental(ContainerBase):
    @staticmethod
    def static_l2_normalize(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        axis: int = None,
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
        axis=None,
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
    def static_instance_norm(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        scale: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        eps: float = 1e-05,
        momentum: Optional[float] = 0.1,
        data_format: str = "NCHW",
        running_mean: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        running_stddev: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ):
        """ivy.Container static method variant of ivy.instance_norm.
        This method simply wraps the function, and so the
        docstring for ivy.instance_norm also applies to this method
        with minimal changes..

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        scale
            Scale parameter for the normalization.
        bias
            Bias parameter for the normalization.
        eps
            Small constant to avoid division by zero.
        momentum
            Momentum parameter for running statistics
        data_format
            Format of the input data, either 'NCHW' or 'NHWC'.
        running_mean
            The running mean of the input container.
        running_stddev
            The running standard deviation of the input container.
        affine
            Whether to use affine transformation for the output.
        track_running_stats
            Whether to track the running statistics of the input container.
        out
            Optional output container, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized container.
            OR
            The normalized container, Running mean container, Running stddev container

        Examples
        --------
        With :class:`track_running_stats=False`:
        ret : The normalized container.

        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
                              b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> bias = ivy.Container(a=ivy.array([0.4, 1.5, 1]),
        ...                      b=ivy.array([1.2, 2.4, 1.5, 3.7, 2.2]))
        >>> scale = ivy.Container(a=ivy.array([0.2, 0.5, 1]),
        ...                       b=ivy.array([0.2, 0.4, 0.5, 0.7, 1.8]))
        >>> ivy.Container.static_instance_norm(x, scale=scale, bias=bias,
        ...                                    data_format='NCHW', affine=True,
        ...                                    track_running_stats=False)
        {
            a: ivy.array([[[[0.68283635],[0.25858182],[0.25858182]],
                           [[1.14645457],[2.20709086],[1.14645457]],
                           [[0.29290909],[0.29290909],[2.41418171]]]]),
            b: ivy.array([[[[1.59998751],[1.10000312],[1.10000312],
                            [1.10000312],[1.10000312]],
                            [[2.20000625],[3.19997501],[2.20000625],
                            [2.20000625],[2.20000625]],
                            [[1.25000787],[1.25000787],[2.49996877],
                            [1.25000787],[1.25000787]],
                            [[3.35001087],[3.35001087],[3.35001087],
                            [5.09995651],[3.35001087]],
                            [[1.30002821],[1.30002821],[1.30002821],
                            [1.30002821],[5.79988766]]]])
        }

        With :class:`track_running_stats=True`:
        ret : normalized container, Running mean container, Running stddev container

        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
        ...                   b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> bias = ivy.Container(a=ivy.array([0.4, 1.5, 1]),
        ...                      b=ivy.array([1.2, 2.4, 1.5, 3.7, 2.2]))
        >>> scale = ivy.Container(a=ivy.array([0.2, 0.5, 1]),
        ...                       b=ivy.array([0.2, 0.4, 0.5, 0.7, 1.8]))
        >>> ivy.Container.static_instance_norm(x, scale=scale, bias=bias,
        ...                                    data_format='NCHW', affine=True,
        ...                                    track_running_stats=True)
        [{
            a: ivy.array([[[[0.68283635],[0.25858182],[0.25858182]],
                           [[1.14645457],[2.20709086],[1.14645457]],
                           [[0.29290909],[0.29290909],[2.41418171]]]]),
            b: ivy.array([[[[1.59998751],[1.10000312],[1.10000312],
                            [1.10000312],[1.10000312]],
                            [[2.20000625],[3.19997501],[2.20000625],
                            [2.20000625],[2.20000625]],
                            [[1.25000787],[1.25000787],[2.49996877],
                            [1.25000787],[1.25000787]],
                            [[3.35001087],[3.35001087],[3.35001087],
                            [5.09995651],[3.35001087]],
                            [[1.30002821],[1.30002821],[1.30002821],
                            [1.30002821],[5.79988766]]]])
        }, {
            a: ivy.array([[[[0.30000001]],[[0.30000001]],[[0.30000001]]]]),
            b: ivy.array([[[[0.17999999]],[[0.17999999]],[[0.17999999]],
                            [[0.17999999]],[[0.17999999]]]])
        }, {
            a: ivy.array([[[[0.52426404]],[[0.52426404]],[[0.52426404]]]]),
            b: ivy.array([[[[0.46000001]],[[0.46000001]],[[0.45999998]],
                            [[0.45999998]],[[0.45999998]]]])
        }]
        """
        return ContainerBase.cont_multi_map_in_function(
            "instance_norm",
            x,
            scale=scale,
            bias=bias,
            eps=eps,
            momentum=momentum,
            data_format=data_format,
            running_mean=running_mean,
            running_stddev=running_stddev,
            affine=affine,
            track_running_stats=track_running_stats,
            out=out,
        )

    def instance_norm(
        self,
        scale: Optional[Union[ivy.Container, ivy.Array, ivy.NativeArray]] = None,
        bias: Optional[Union[ivy.Container, ivy.Array, ivy.NativeArray]] = None,
        eps: float = 1e-05,
        momentum: Optional[float] = 0.1,
        data_format: str = "NCHW",
        running_mean: Optional[Union[ivy.Container, ivy.Array, ivy.NativeArray]] = None,
        running_stddev: Optional[
            Union[ivy.Container, ivy.Array, ivy.NativeArray]
        ] = None,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = False,
        out: Optional[ivy.Array] = None,
    ):
        """ivy.Container instance method variant of ivy.instance_norm.
        This method simply wraps the function, and so the
        docstring for ivy.instance_norm also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            The input container with leaves to be normalized.
        scale
            Scale parameter for the normalization.
        bias
            Bias parameter for the normalization.
        eps
            Small constant to avoid division by zero.
        momentum
            Momentum parameter for running statistics
        data_format
            Format of the input data, either 'NCHW' or 'NHWC'.
        running_mean
            The running mean of the input container.
        running_stddev
            The running standard deviation of the input container.
        affine
            Whether to use (Scale, Bias) transformation for the output.
        track_running_stats
            Whether to track the running statistics of the input container.
        out
            Optional output container, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized container.
            OR
            The normalized container, Running mean container, Running stddev container

        Examples
        --------
        With :class:`track_running_stats=False`:
        ret : The normalized container.

        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
                              b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> bias = ivy.Container(a=ivy.array([0.4, 1.5, 1]),
        ...                      b=ivy.array([1.2, 2.4, 1.5, 3.7, 2.2]))
        >>> scale = ivy.Container(a=ivy.array([0.2, 0.5, 1]),
        ...                       b=ivy.array([0.2, 0.4, 0.5, 0.7, 1.8]))
        >>> ivy.Container.static_instance_norm(x, scale=scale, bias=bias,
        ...                                    data_format='NCHW',affine=True,
        ...                                    track_running_stats=False)
        {
            a: ivy.array([[[[0.68283635],[0.25858182],[0.25858182]],
                           [[1.14645457],[2.20709086],[1.14645457]],
                           [[0.29290909],[0.29290909],[2.41418171]]]]),
            b: ivy.array([[[[1.59998751],[1.10000312],[1.10000312],
                            [1.10000312],[1.10000312]],
                            [[2.20000625],[3.19997501],[2.20000625],
                            [2.20000625],[2.20000625]],
                            [[1.25000787],[1.25000787],[2.49996877],
                            [1.25000787],[1.25000787]],
                            [[3.35001087],[3.35001087],[3.35001087],
                            [5.09995651],[3.35001087]],
                            [[1.30002821],[1.30002821],[1.30002821],
                            [1.30002821],[5.79988766]]]])
        }

        With :class:`track_running_stats=True`:
        ret : normalized container, Running mean container, Running stddev container

        >>> x = ivy.Container(a = ivy.eye(3, 3).reshape((1, 3, 3, 1)),
                              b = ivy.eye(5, 5).reshape((1, 5, 5, 1)))
        >>> bias = ivy.Container(a=ivy.array([0.4, 1.5, 1]),
        ...                      b=ivy.array([1.2, 2.4, 1.5, 3.7, 2.2]))
        >>> scale = ivy.Container(a=ivy.array([0.2, 0.5, 1]),
        ...                       b=ivy.array([0.2, 0.4, 0.5, 0.7, 1.8]))
        >>> ivy.Container.static_instance_norm(x, scale=scale, bias=bias,
        ...                                    data_format='NCHW',affine=True,
        ...                                    track_running_stats=True)
        [{
            a: ivy.array([[[[0.68283635],[0.25858182],[0.25858182]],
                           [[1.14645457],[2.20709086],[1.14645457]],
                           [[0.29290909],[0.29290909],[2.41418171]]]]),
            b: ivy.array([[[[1.59998751],[1.10000312],[1.10000312],
                            [1.10000312],[1.10000312]],
                            [[2.20000625],[3.19997501],[2.20000625],
                            [2.20000625],[2.20000625]],
                            [[1.25000787],[1.25000787],[2.49996877],
                            [1.25000787],[1.25000787]],
                            [[3.35001087],[3.35001087],[3.35001087],
                            [5.09995651],[3.35001087]],
                            [[1.30002821],[1.30002821],[1.30002821],
                            [1.30002821],[5.79988766]]]])
        }, {
            a: ivy.array([[[[0.30000001]],[[0.30000001]],[[0.30000001]]]]),
            b: ivy.array([[[[0.17999999]],[[0.17999999]],[[0.17999999]],
                            [[0.17999999]],[[0.17999999]]]])
        }, {
            a: ivy.array([[[[0.52426404]],[[0.52426404]],[[0.52426404]]]]),
            b: ivy.array([[[[0.46000001]],[[0.46000001]],
                            [[0.45999998]],[[0.45999998]],[[0.45999998]]]])
        }]
        """
        return self.static_instance_norm(
            self,
            scale=scale,
            bias=bias,
            eps=eps,
            momentum=momentum,
            data_format=data_format,
            running_mean=running_mean,
            running_stddev=running_stddev,
            affine=affine,
            track_running_stats=track_running_stats,
            out=out,
        )
