# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithLossesExperimental(abc.ABC):
    def l1_loss(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: str = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.l1_loss. This method simply wraps the
        function, and so the docstring for ivy.l1_loss also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array.
        pred
            input array containing the predicted values.
        reduction
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed.
            ``'none'``: No reduction will be applied to the output. Default: ``'mean'``.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The L1 loss between the input array and the predicted values.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([0.7, 1.8, 2.9])
        >>> z = x.l1_loss(y)
        >>> print(z)
        ivy.array(0.30000000000000004)
        """
        return ivy.l1_loss(self._data, pred, reduction=reduction, out=out)