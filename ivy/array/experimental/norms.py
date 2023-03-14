# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithNormsExperimental(abc.ABC):
    def l2_normalize(
        self: ivy.Array,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Normalizes the array to have unit L2 norm.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis along which to normalize. If ``None``, the whole array
            is normalized.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized array.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> x.l2_normalize(axis=1)
        ivy.array([[0.4472, 0.8944],
                   [0.6, 0.8]])
        """
        return ivy.l2_normalize(self, axis=axis, out=out)

    def instance_norm(
        self: ivy.Array,
        /,
        *,
        scale: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        bias: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        eps: float = 1e-05,
        momentum: float = 0.1,
        data_format: str = "NCHW",
        running_mean: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        running_stddev: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
        affine: bool = True,
        track_running_stats: bool = False,
        out: Optional[ivy.Array] = None,
    ):
        """Applies Instance Normalization over a 4D input along C dimension.

        Parameters
        ----------
        self
            Input array.
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
            The running mean of the input array.
        running_stddev
            The running standard deviation of the input array.
        affine
            Whether to use affine transformation for the output.
        track_running_stats
            Whether to track the running statistics of the input array.
        out
            Optional output array, for writing the result to. It must
            have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized array.
            OR
            The normalized array, Running mean, Running stddev

        Examples
        --------
        With :class:`track_running_stats=False`:
        ret : The normalized array.

        >>> x = ivy.eye(3, 3).reshape((1, 3, 3, 1))
        >>> ivy.instance_norm(x, scale=[2., 1, 0.5], bias=[2., 1, 0.5],
        ...                   data_format='NCHW', affine=True,
        ...                   track_running_stats=False)
        ivy.array([[[[4.82836342],[0.58581817],[0.58581817]],
                [[0.29290909],[2.41418171],[0.29290909]],
                [[0.14645454],[0.14645454],[1.20709085]]]])

        With :class:`track_running_stats=True`:
        ret : The normalized array, Running mean, Running stddev.

        >>> x = ivy.eye(3, 3).reshape((1, 3, 3, 1))
        >>> ivy.instance_norm(x, scale=[2., 1, 0.5], bias=[2., 1, 0.5],
        ...                   data_format='NCHW',affine=True,
        ...                   track_running_stats=False)
        (ivy.array([[[[4.82836342],[0.58581817],[0.58581817]],
                [[0.29290909],[2.41418171],[0.29290909]],
                [[0.14645454],[0.14645454],[1.20709085]]]]),
         ivy.array([[[[0.30000001]],[[0.30000001]],[[0.30000001]]]]),
         ivy.array([[[[0.52426404]],[[0.52426404]],[[0.52426404]]]]))
        """
        return ivy.instance_norm(
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

    def lp_normalize(
        self: ivy.Array,
        /,
        *,
        p: float = 2,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """Normalizes the array to have Lp norm.

        Parameters
        ----------
        self
            Input array.
        p
            p-norm to use for normalization.
        axis
            Axis along which to normalize. If ``None``, the whole array
            is normalized.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized array.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> x.lp_normalize(p=2, axis=1)
        ivy.array([[0.4472, 0.8944],
               [0.6, 0.8]])
        """
        return ivy.lp_normalize(self, p=p, axis=axis, out=out)
