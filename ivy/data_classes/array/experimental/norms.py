# global
import abc
from typing import Optional, Union, Tuple

# local
import ivy


class _ArrayWithNormsExperimental(abc.ABC):
    def l1_normalize(
        self: ivy.Array,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Normalize the array to have unit L1 norm.

        Parameters
        ----------
        self
            Input array.
        axis
            Axis or axes along which to normalize. If ``None``,
            the whole array is normalized.
        out
            Optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The normalized array.

        Examples
        --------
        >>> x = ivy.array([[1., 2.], [3., 4.]])
        >>> x.l1_normalize(axis=1)
        ivy.array([[0.3333, 0.6667],
                   [0.4286, 0.5714]])
        """
        return ivy.l1_normalize(self, axis=axis, out=out)

    def l2_normalize(
        self: ivy.Array,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Normalize the array to have unit L2 norm.

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

    def batch_norm(
        self: Union[ivy.NativeArray, ivy.Array],
        mean: Union[ivy.NativeArray, ivy.Array],
        variance: Union[ivy.NativeArray, ivy.Array],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        data_format: str = "NSC",
        out: Optional[Tuple[ivy.Array, ivy.Array, ivy.Array]] = None,
    ) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.batch_norm. This method simply wraps
        the function, and so the docstring for ivy.batch_norm also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            Input array of default shape (N, *S, C), where N is the batch dimension,
            *S corresponds to any number of spatial dimensions and
             C corresponds to the channel dimension.
        training
            If true, calculate and use the mean and variance of `x`. Otherwise, use the
            provided `mean` and `variance`.
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

        Returns
        -------
        ret
             Tuple of arrays containing the
             normalized input, running mean, and running variance.
        """
        return ivy.batch_norm(
            self._data,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            data_format=data_format,
            out=out,
        )

    def instance_norm(
        self: Union[ivy.NativeArray, ivy.Array],
        mean: Union[ivy.NativeArray, ivy.Array],
        variance: Union[ivy.NativeArray, ivy.Array],
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        training: bool = False,
        eps: float = 1e-5,
        momentum: float = 1e-1,
        data_format: str = "NSC",
        out: Optional[Tuple[ivy.Array, ivy.Array, ivy.Array]] = None,
    ) -> Tuple[ivy.Array, ivy.Array, ivy.Array]:
        """
        ivy.Array instance method variant of ivy.instance_norm. This method simply wraps
        the function, and so the docstring for ivy.instance_norm also applies to this
        method with minimal changes.

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
            optional output array, for writing the result to.

        Returns
        -------
        ret
             Tuple of array containing
              the normalized input, running mean, and running variance.
        """
        return ivy.instance_norm(
            self._data,
            mean,
            variance,
            scale=scale,
            offset=offset,
            training=training,
            eps=eps,
            momentum=momentum,
            out=out,
            data_format=data_format,
        )

    def group_norm(
        self: Union[ivy.NativeArray, ivy.Array],
        num_groups: int = 1,
        /,
        *,
        offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
        eps: Optional[float] = 1e-5,
        data_format: Optional[str] = "NSC",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.group_norm. This method simply wraps
        the function, and so the docstring for ivy.group_norm also applies to this
        method with minimal changes.

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
        return ivy.group_norm(
            self._data,
            num_groups,
            scale=scale,
            offset=offset,
            eps=eps,
            out=out,
            data_format=data_format,
        )

    def lp_normalize(
        self: ivy.Array,
        /,
        *,
        p: float = 2,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        Normalize the array to have Lp norm.

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
