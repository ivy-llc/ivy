# global
import abc
from typing import Optional, Union

# local
import ivy


class _ArrayWithLosses(abc.ABC):
    def cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        reduction: str = "sum",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.cross_entropy. This method simply
        wraps the function, and so the docstring for ivy.cross_entropy also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        pred
            input array containing the predicted labels.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``,
            the cross-entropy will be computed along the last dimension.
            Default: ``-1``.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The cross-entropy loss between the given distributions.

        Examples
        --------
        >>> x = ivy.array([0, 0, 1, 0])
        >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
        >>> z = x.cross_entropy(y)
        >>> print(z)
        ivy.array(1.3862944)
        """
        return ivy.cross_entropy(
            self._data, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
        )

    def binary_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        epsilon: float = 1e-7,
        reduction: str = "none",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.binary_cross_entropy. This method
        simply wraps the function, and so the docstring for ivy.binary_cross_entropy
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        pred
            input array containing Predicted labels.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The binary cross entropy between the given distributions.

        Examples
        --------
        >>> x = ivy.array([1 , 1, 0])
        >>> y = ivy.array([0.7, 0.8, 0.2])
        >>> z = x.binary_cross_entropy(y)
        >>> print(z)
        ivy.array([0.357, 0.223, 0.223])
        """
        return ivy.binary_cross_entropy(
            self._data, pred, epsilon=epsilon, reduction=reduction, out=out
        )

    def sparse_cross_entropy(
        self: ivy.Array,
        pred: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        axis: int = -1,
        epsilon: float = 1e-7,
        reduction: str = "sum",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sparse_cross_entropy. This method
        simply wraps the function, and so the docstring for ivy.sparse_cross_entropy
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing the true labels as logits.
        pred
            input array containing the predicted labels as logits.
        axis
            the axis along which to compute the cross-entropy. If axis is ``-1``, the
            cross-entropy will be computed along the last dimension. Default: ``-1``.
            epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied.
            Default: ``1e-7``.
        epsilon
            a float in [0.0, 1.0] specifying the amount of smoothing when calculating
            the loss. If epsilon is ``0``, no smoothing will be applied. Default:
            ``1e-7``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The sparse cross-entropy loss between the given distributions.

        Examples
        --------
        >>> x = ivy.array([1 , 1, 0])
        >>> y = ivy.array([0.7, 0.8, 0.2])
        >>> z = x.sparse_cross_entropy(y)
        >>> print(z)
        ivy.array([0.223, 0.223, 0.357])
        """
        return ivy.sparse_cross_entropy(
            self._data, pred, axis=axis, epsilon=epsilon, reduction=reduction, out=out
        )
