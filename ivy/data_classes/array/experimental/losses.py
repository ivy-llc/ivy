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
            input array containing true labels.
        target
            input array containing targeted labels.
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

    def log_poisson_loss(
        self: Union[ivy.Array, ivy.NativeArray],
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        compute_full_loss: bool = False,
        axis: int = -1,
        reduction: str = "none",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.log_poisson_loss. This method simply
        wraps the function, and so the docstring for ivy.l1_loss also applies to this
        method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        target
            input array containing targeted labels.
        compute_full_loss
            whether to compute the full loss. If false, a constant term is dropped
            in favor of more efficient optimization. Default: ``False``.
        axis
            the axis along which to compute the log-likelihood loss. If axis is ``-1``,
            the log-likelihood loss will be computed along the last dimension.
            Default: ``-1``.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'none'``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The binary log-likelihood loss between the given distributions.


        Examples
        --------
        >>> x = ivy.array([0, 0, 1, 0])
        >>> y = ivy.array([0.25, 0.25, 0.25, 0.25])
        >>> loss = x.log_poisson_loss(y)
        >>> print(loss)
        ivy.array([1.28402555, 1.28402555, 1.03402555, 1.28402555])

        >>> z = ivy.array([0.1, 0.1, 0.7, 0.1])
        >>> loss = x.x.log_poisson_loss(z, reduction='mean')
        >>> print(loss)
        ivy.array(1.1573164)
        """
        return ivy.log_poisson_loss(
            self._data,
            target,
            compute_full_loss=compute_full_loss,
            axis=axis,
            reduction=reduction,
            out=out,
        )

    def huber_loss(
        self: ivy.Array,
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Optional[str] = "mean",
        delta: Optional[float] = 1.0,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of huber_loss. This method simply wraps the
        function, and so the docstring for huber_loss also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        target
            input array containing targeted labels.
        reduction : str, optional
            The type of reduction to apply to the loss.
            Possible values are "mean" (default)
            and "sum".
        delta
            The threshold parameter that determines the point where the loss transitions
            from squared error to absolute error. Default is 1.0.
        out
            Optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The Huber loss between the true and predicted values.

        Examples
        --------
        >>> true = ivy.array([2, 4, 7, 1])
        >>> pred = ivy.array([2.5, 3.5, 8, 0.8])
        >>> loss = true.huber_loss(pred, delta=1.0)
        >>> print(loss)
        ivy.array([0.125, 0.125, 0.5  , 0.125])
        """
        return ivy.huber_loss(
            self._data, target, reduction=reduction, delta=delta, out=out
        )

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

    def soft_margin_loss(
        self: ivy.Array,
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Optional[str] = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.soft_margin_loss. This method simply
        wraps the function, and so the docstring for ivy.soft_margin_loss also applies
        to this method with minimal changes.

        Parameters
        ----------
        self
            input array containing true labels.
        target
            input array containing targeted labels.
        reduction
            ``'none'``: No reduction will be applied to the output.
            ``'mean'``: The output will be averaged.
            ``'sum'``: The output will be summed. Default: ``'sum'``.
        out
            optional output array, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The soft margin loss between the true and targeticted labels.

        Examples
        --------
        >>> x = ivy.array([1, 1, 0])
        >>> y = ivy.array([0.7, 0.8, 0.2])
        >>> z = x.soft_margin_loss(y)
        >>> print(z)
        ivy.array([0.35667497, 0.22314353, 1.60943791])
        """
        return ivy.soft_margin_loss(self._data, target, reduction=reduction, out=out)

    def kl_div(
        self: ivy.Array,
        target: Union[ivy.Array, ivy.NativeArray],
        /,
        *,
        reduction: Optional[str] = "mean",
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.kl_div. This method simply wraps the
        function, and so the docstring for ivy.kl_div also applies to this method with
        minimal changes.

        Parameters
        ----------
        self
            Array containing input probability distribution.
        target
            Array contaiing target probability distribution.
        reduction
            'none': No reduction will be applied to the output.
            'mean': The output will be averaged.
            'batchmean': The output will be divided by batch size.
            'sum': The output will be summed.
            Default: 'mean'.
        out
            Optional output array, for writing the result to.
            It must have a shape that the inputs broadcast to.

        Returns
        -------
        ret
            The Kullback-Leibler divergence loss between the two input arrays.

        Examples
        --------
        >>> input = ivy.array([0.2, 0.8], [0.5, 0.5])
        >>> target = ivy.array([0.6, 0.4], [0.3, 0.7])
        >>> output_array = input.kl_div(target)
        >>> print(output_array)
        ivy.array(0.0916)
        """
        return ivy.kl_div(self._data, target, reduction=reduction, out=out)
