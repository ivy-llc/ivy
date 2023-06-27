# global
import abc
import ivy
from typing import Optional, Union


class _ArrayWithLossesExperimental(abc.ABC):
    def mse_loss(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        reduction: str = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.mse_loss. This method simply wraps
        the function, and so the docstring for ivy.mse_loss also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        pred
            input array containing the predicted labels.
        axis
            the axis along which to compute the mean squared error. If axis is `None`,
            the mean squared error will be computed over all dimensions.
            Default: `None`.
        reduction
            specifies the reduction to apply to the output. Options are 'mean', 'sum',
            or 'none'. Default: 'mean'.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The mean squared error loss between the true and predicted labels.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([2, 4, 6])
        >>> z = x.mse_loss(y)
        >>> print(z)
        ivy.array(3.0)
        """
        return ivy.mse_loss(
            self._data, pred, axis=axis, reduction=reduction, out=out
        )

    def mae_loss(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: Optional[int] = None,
        reduction: str = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.mae_loss. This method simply wraps
        the function, and so the docstring for ivy.mae_loss also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        pred
            input array containing the predicted labels.
        axis
            the axis along which to compute the mean absolute error. If axis is `None`,
            the mean absolute error will be computed over all dimensions.
            Default: `None`.
        reduction
            specifies the reduction to apply to the output. Options are 'mean', 'sum',
            or 'none'. Default: 'mean'.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            The mean absolute error loss between the true and predicted labels.

        Examples
        --------
        >>> x = ivy.array([1, 2, 3])
        >>> y = ivy.array([2, 4, 6])
        >>> z = x.mae_loss(y)
        >>> print(z)
        ivy.array(2.0)
        """
        return ivy.mae_loss(
            self._data, pred, axis=axis, reduction=reduction, out=out
        )
