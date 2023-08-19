# global
import abc

from typing import Optional, Union

# local
import ivy


class _ArrayWithLossesExperimental(abc.ABC):
    def gaussian_nll_loss(
        self: ivy.Array,
        variance: Union[ivy.Array, ivy.NativeArray],
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        full: Union[bool, ivy.Container] = False,
        reduction: Optional[Union[str, ivy.Array]] = "mean",
        epsilon: Optional[Union[float, ivy.Array]] = 1e-7,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gaussian_nll_loss. This method simply
        wraps the function, and so the docstring for ivy.gaussian_nll_loss also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing the true values.
        variance
            input array containing the log-variances of the Gaussian distributions.
        target
            input array containing the true values.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'sum'``.
        epsilon
            a float specifying the amount of smoothing when calculating the loss.
            Default: ``1e-7``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The Gaussian negative log likelihood loss between the given distributions.

        Examples
        --------
        >>> mean = ivy.array([0.0, 1.0, -1.0])
        >>> variance = ivy.array([0.0, 0.0, 0.0])
        >>> target = ivy.array([1.0, 1.0, -1.0])
        >>> loss = target.gaussian_nll_loss(mean, variance, target)
        >>> print(loss)
        ivy.array([0.5, 0.5, 0.])
        """
        return ivy.gaussian_nll_loss(
            self._data,
            variance,
            target,
            full=full,
            reduction=reduction,
            epsilon=epsilon,
            out=out,
        )
