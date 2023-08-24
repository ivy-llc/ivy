# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithLossesExperimental(abc.ABC):
    def l1_loss(
        self: Union[ivy.Array, ivy.NativeArray],
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Optional[str] = "mean",
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
        target
            input array containing the targeted values.
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
            The L1 loss between the input array and the targeticted values.

        Examples
        --------
        >>> x = ivy.array([1.0, 2.0, 3.0])
        >>> y = ivy.array([0.7, 1.8, 2.9])
        >>> z = x.l1_loss(y)
        >>> print(z)
        ivy.array(0.20000000000000004)
        """
        return ivy.l1_loss(self._data, target, reduction=reduction, out=out)

    def smooth_l1_loss(
        self: ivy.Array,
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        beta: Optional[float] = 1.0,
        reduction: Optional[str] = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy. smooth_l1_loss. This method simply
        wraps the function, and so the docstring for ivy.smooth_l1_loss also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        target
            input array containing targeted labels.
        beta
            A float specifying the beta value for
            the smooth L1 loss. Default: 1.0.
        reduction
            Reduction method for the loss.
            Options are 'none', 'mean', or 'sum'.
            Default: 'mean'.
        out
            Optional output array, for writing the result to.
            It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The smooth L1 loss between the given labels.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3, 4])
        >>> y = ivy.array([2, 2, 2, 2])
        >>> z = x.smooth_l1_loss(y, beta=0.5)
        >>> print(z)
        ivy.array(0.8125)
        """
        return ivy.smooth_l1_loss(
            self._data, target, beta=beta, reduction=reduction, out=out
        )
